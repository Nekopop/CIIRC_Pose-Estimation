import cv2
import numpy as np
from ultralytics import YOLO
import glob
import os
import matplotlib.pyplot as plt

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

    def process_image(self, image, depth_image_shape):
        # Resize RGB image to the same resolution as the depth image
        image_resized = cv2.resize(image, (depth_image_shape[1], depth_image_shape[0]))

        # Scale camera matrix
        self.camera_matrix = self.scale_camera_matrix(depth_image_shape[0], depth_image_shape[1])

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

                            # Draw results (optional)
                            image_resized = self.draw_results(image_resized, keypoints, distance, tvec, rvec, is_front)
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

    def draw_results(self, image, keypoints, distance, tvec, rvec, is_front):
        # Draw keypoints and lines
        color = (0, 255, 0) if is_front else (255, 0, 0)  # Front: green, Side: blue
        for pt in keypoints:
            x, y = map(int, pt)
            cv2.circle(image, (x, y), 5, color, -1)

        for i in range(len(keypoints)):
            pt1 = tuple(map(int, keypoints[i]))
            pt2 = tuple(map(int, keypoints[(i + 1) % len(keypoints)]))
            cv2.line(image, pt1, pt2, color, 2)

        # Calculate center point
        center_x = int(np.mean(keypoints[:, 0]))
        center_y = int(np.mean(keypoints[:, 1]))

        # Draw center point
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

        # Prepare text
        text = f"{distance / 1000.0:.3f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Text position
        text_x = center_x - text_width // 2
        text_y = center_y - 15

        # Draw background rectangle
        padding = 4
        cv2.rectangle(image,
                      (text_x - padding, text_y - text_height - padding),
                      (text_x + text_width + padding, text_y + padding),
                      (0, 0, 0),
                      -1)

        # Draw text
        cv2.putText(image, text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness)

        # Draw normal vector
        # Rodrigues' rotation formula to convert rvec to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        # Calculate the end point of the normal vector
        normal_vector = np.dot(rotation_matrix, tvec).flatten()
        end_point = (center_x + int(normal_vector[0]), center_y + int(normal_vector[1]))
        # Draw the normal vector as a line
        cv2.line(image, (center_x, center_y), end_point, (0, 255, 255), 2)

        return image

    def get_last_distances(self):
        """Return the list of estimated distances (in meters)"""
        return self._last_distances

    def get_last_centers(self):
        """Return the list of estimated center points"""
        return self._last_centers

def calculate_actual_distance(depth_image, center, camera_matrix):
    """
    Calculate the actual distance from the depth image and camera parameters.

    Args:
        depth_image (np.ndarray): Depth image
        center (np.ndarray): Estimated center point coordinates
        camera_matrix (np.ndarray): Camera matrix

    Returns:
        float: Distance from the camera (in meters)
    """
    cx, cy = int(center[0]), int(center[1])

    # Check if indices are within image bounds
    if cx < 0 or cx >= depth_image.shape[1] or cy < 0 or cy >= depth_image.shape[0]:
        raise ValueError("Center point is out of image bounds")

    depth = depth_image[cy, cx] / 1000.0  # Convert depth value to meters

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cxo = camera_matrix[0, 2]
    cyo = camera_matrix[1, 2]

    x = (center[0] - cxo) * depth / fx
    y = (center[1] - cyo) * depth / fy
    z = depth

    return np.linalg.norm([x, y, z])

import os
import numpy as np

def evaluate_distance_estimation(folder_path):
    """
    Evaluate the accuracy of the distance calculation program using the dataset.

    Args:
        folder_path (str): Path to the folder containing RGB images, depth images, and point cloud data
    """
    # Initialize BrickPoseEstimator
    model_path = "YOLO11/weight/2024-12-10-pose-yolo11l-500epoch(2024-11-27_vertical_aug).pt"
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
                # Calculate actual distance from depth image
                actual_distance = calculate_actual_distance(depth_image, center, estimator.camera_matrix)
                
                # Calculate error
                error = abs(estimated_distance - actual_distance)
                
                # Round actual distance to nearest 0.5m for grouping
                distance_bucket = round(actual_distance * 2) / 2
                if distance_bucket not in distance_errors:
                    distance_errors[distance_bucket] = []
                distance_errors[distance_bucket].append(error)

                print(f"Image: {os.path.basename(rgb_path)}")
                print(f"Estimated distance: {estimated_distance:.3f}m")
                print(f"Actual distance: {actual_distance:.3f}m")
                print(f"Error: {error:.3f}m")
                print("-" * 40)
                
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
    total_results = []

    # Iterate over subfolders
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Evaluate and collect statistical results
            stats = evaluate_distance_estimation(subfolder_path)
            total_results.extend(stats)

    # Compute total results and save to file
    if total_results:
        total_stats_output_path = os.path.join(root_folder, 'total_statistics.txt')
        with open(total_stats_output_path, 'w') as f:
            f.write("Total Evaluation Results:\n")
            distances = sorted(set([d for d, mean, std, count in total_results]))
            for distance in distances:
                distance_means = [mean for d, mean, std, count in total_results if d == distance]
                distance_stds = [std for d, mean, std, count in total_results if d == distance]
                distance_counts = [count for d, mean, std, count in total_results if d == distance]
                overall_mean = np.mean(distance_means)
                overall_std = np.mean(distance_stds)
                total_count = sum(distance_counts)
                result_line = f"Distance {distance:.1f}m - Overall Mean error: {overall_mean:.3f}m, Overall Std: {overall_std:.3f}m, Total Count: {total_count}"
                print(result_line)
                f.write(result_line + '\n')