import cv2
import numpy as np
from ultralytics import YOLO
import glob
import os
import matplotlib.pyplot as plt
import time
import statistics

# With image saving
class BrickPoseEstimator:
    def __init__(self, model_path):
        # Load YOLO model
        self.yolo = YOLO(model_path)

        # Original camera parameters (2K: 1440x2560)
        self.original_camera_matrix = np.array([
            [2067.0886, 0.0, 1235.7012],
            [0.0, 2065.7244, 727.0466],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([7.1634, -88.0136, -0.00047, -0.00045, 219.1458, 6.9479, -86.5780, 216.1373], dtype=np.float32)

        self._last_distances = []
        self._last_centers = []

    def scale_camera_matrix(self, image_height, image_width):
        # Calculate scale factor
        height_scale = image_height / 1440
        width_scale = image_width / 2560

        # Scale camera matrix
        scaled_matrix = self.original_camera_matrix.copy()
        scaled_matrix[0, 0] *= width_scale   # fx
        scaled_matrix[1, 1] *= height_scale  # fy
        scaled_matrix[0, 2] *= width_scale   # cx
        scaled_matrix[1, 2] *= height_scale  # cy

        return scaled_matrix

    def process_image(self, image):
        # RGB画像を1280x720にリサイズ
        image_resized = cv2.resize(image, (1280, 720))

        # Scale camera matrix
        self.camera_matrix = self.scale_camera_matrix(image_resized.shape[0], image_resized.shape[1])

        # YOLO inference
        results = self.yolo(image_resized)[0]
        print(f"Image size: {image_resized.shape[1]}x{image_resized.shape[0]}")
        print(f"Scaled camera matrix:\n{self.camera_matrix}")

        self._last_distances = []
        self._last_centers = []

        if results.keypoints is not None:
            print(f"Detections found: {len(results.keypoints.data)}")

            for idx, (kpts, cls) in enumerate(zip(results.keypoints.data, results.boxes.cls)):
                class_id = int(cls)
                label = results.names[class_id]

                # Filter Bricks with 4 keypoints
                if label == 'Bricks' and len(kpts) == 4:
                    # Check confidence scores
                    conf_scores = kpts[:, 2]
                    if all(conf > 0.5 for conf in conf_scores):
                        keypoints = kpts[:, :2].cpu().numpy()

                        try:
                            # Pose estimation
                            distance, tvec, rvec, center, is_front = self.estimate_pose(keypoints)
                            self._last_distances.append(distance / 1000.0)  # Convert to meters
                            self._last_centers.append(center)

                            # Draw results (optional) - 削除
                            # image_resized = self.draw_results(image_resized, keypoints, distance, tvec, rvec, is_front)
                        except Exception as e:
                            print(f"Error in pose estimation: {str(e)}")

        return image_resized

    def estimate_pose(self, keypoints_2d):
        def get_brick_parameters(points_2d, camera_matrix, dist_coeffs):
            # Define dimensions for front and side
            front = {'half_w': 500, 'half_h': 690}  # Front: width 100cm, height 138cm
            side = {'half_w': 610, 'half_h': 690}   # Side: width 122cm, height 138cm

            def get_3d_points(params):
                return np.float32([
                    [-params['half_w'], -params['half_h'], 0],
                    [ params['half_w'], -params['half_h'], 0],
                    [ params['half_w'],  params['half_h'], 0],
                    [-params['half_w'],  params['half_h'], 0]
                ])

            def compute_error(params):
                points_3d = get_3d_points(params)
                success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs)
                if not success:
                    return float('inf'), None, None

                projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
                error = np.mean(np.linalg.norm(points_2d - projected_points.reshape(-1, 2), axis=1))
                return error, rvec, tvec

            # Calculate errors for front and side
            front_error, front_rvec, front_tvec = compute_error(front)
            side_error, side_rvec, side_tvec = compute_error(side)

            # Select the one with smaller error
            if front_error < side_error:
                return front['half_w'], front['half_h'], front_rvec, front_tvec, True
            else:
                return side['half_w'], side['half_h'], side_rvec, side_tvec, False

        image_points = np.array(keypoints_2d, dtype=np.float32)
        half_w, half_h, rvec, tvec, is_front = get_brick_parameters(image_points, self.camera_matrix, self.dist_coeffs)
        distance = np.linalg.norm(tvec)

        # Calculate center point
        center = np.mean(image_points, axis=0)

        return distance, tvec, rvec, center, is_front

    def get_last_distances(self):
        """Return the list of estimated distances (in meters)"""
        return self._last_distances

    def get_last_centers(self):
        """Return the list of estimated center points"""
        return self._last_centers

def evaluate_distance_estimation(folder_path):
    """
    Evaluate the accuracy of the distance calculation program using the dataset.

    Args:
        folder_path (str): Path to the folder containing RGB images, depth images, and point cloud data
    """
    # Initialize BrickPoseEstimator
    model_path = "YOLO11/weight/2024-12-10-pose-yolo11m-500epoch_2024-11-27_vertical_aug.pt"
    estimator = BrickPoseEstimator(model_path)

    # Get file paths of RGB images
    rgb_files = glob.glob(os.path.join(folder_path, "**/rgb_frame_*.png"), recursive=True)
    
    # Initialize dictionary to store errors by distance ranges
    distance_errors = {}

    for rgb_path in rgb_files:
        # Read corresponding depth image
        depth_path = rgb_path.replace("rgb_frame_", "depth_frame_")
        if not os.path.exists(depth_path):
            continue
            
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        if depth_image is None:
            continue

        # Read RGB image
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            continue

        # Perform distance estimation
        result_image = estimator.process_image(rgb, depth_image.shape)
        estimated_distances = estimator.get_last_distances()
        centers = estimator.get_last_centers()

        if not estimated_distances:
            continue

        for estimated_distance, center in zip(estimated_distances, centers):
            try:
                print(f"Image: {os.path.basename(rgb_path)}")
                print(f"Estimated distance: {estimated_distance:.3f}m")             
            except ValueError as e:
                print(f"Error: {e}")

    # Collect statistical data
    statistical_results = []

    if distance_errors:
        # Save statistics for each distance range
        stats_output_path = os.path.join(folder_path, 'statistics.txt')
        with open(stats_output_path, 'w') as f:
            f.write("Evaluation Results by Distance:\n")
            for distance, errors in sorted(distance_errors.items()):
                mean = np.mean(errors)
                std = np.std(errors)
                count = len(errors)
                result_line = f"Distance {distance:.1f}m - Mean error: {mean:.3f}m, Std: {std:.3f}m, Count: {count}"
                print(result_line)
                f.write(result_line + '\n')
                statistical_results.append((distance, mean, std, count))
    return statistical_results

if __name__ == "__main__":
    root_folder = "dataset/pointscloud/rotated"
    total_time = 0
    image_count = 0
    processing_times = []  # 処理時間を保存するリスト
    model_path = "YOLO11/weight/2024-12-10-pose-yolo11l-500epoch_2024-11-27_vertical_aug.pt"
    estimator = BrickPoseEstimator(model_path)

    # サブフォルダを繰り返し処理
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            images_folder = os.path.join(subfolder_path, "images")
            if os.path.isdir(images_folder):
                for image_file in os.listdir(images_folder):
                    image_path = os.path.join(images_folder, image_file)
                    if os.path.isfile(image_path):
                        # 画像を読み込む
                        rgb = cv2.imread(image_path)
                        if rgb is None:
                            continue

                        start_time = time.time()
                        # process_image メソッドを呼び出し
                        result_image = estimator.process_image(rgb)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        total_time += elapsed_time
                        processing_times.append(elapsed_time)  # 処理時間をリストに追加
                        image_count += 1

                        # 推定された距離を取得して出力
                        estimated_distances = estimator.get_last_distances()
                        for distance in estimated_distances:
                            print(f"Estimated distance: {distance:.3f} meters")

    # 画像1枚あたりの平均処理時間を計算
    if image_count > 0:
        average_time_per_image = total_time / image_count
        std_dev_time = statistics.stdev(processing_times)  # 標準偏差を計算
        print(f"Average processing time per image: {average_time_per_image:.3f} seconds")
        print(f"Standard deviation of processing times: {std_dev_time:.3f} seconds")
    else:
        print("No images found.")