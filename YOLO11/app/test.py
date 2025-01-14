import cv2
import numpy as np
from ultralytics import YOLO
import random
import glob
import os
import argparse

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

        self.original_height = 1440
        self.original_width = 2560

        self.dist_coeffs = np.array([7.1634, -88.0136, -0.00047, -0.00045, 219.1458, 6.9479, -86.5780, 216.1373], dtype=np.float32)

        # Center the Bricks model (match YOLO keypoint order)
        half_w, half_h = 500, 690
        self.model_points = np.array([
            [-half_w, half_h, 0],   # Point 0: Bottom-Left - Corresponds to YOLO's 1st keypoint
            [half_w, half_h, 0],    # Point 1: Bottom-Right - Corresponds to YOLO's 2nd keypoint
            [half_w, -half_h, 0],   # Point 2: Top-Right - Corresponds to YOLO's 3rd keypoint
            [-half_w, -half_h, 0]   # Point 3: Top-Left - Corresponds to YOLO's 4th keypoint
        ], dtype=np.float32)

    def scale_camera_matrix(self, image_height, image_width):
        # Calculate scale factors
        height_scale = image_height / self.original_height
        width_scale = image_width / self.original_width

        # Scale camera matrix
        scaled_matrix = self.original_camera_matrix.copy()
        scaled_matrix[0, 0] *= width_scale   # fx
        scaled_matrix[1, 1] *= height_scale  # fy
        scaled_matrix[0, 2] *= width_scale   # cx
        scaled_matrix[1, 2] *= height_scale  # cy

        return scaled_matrix

    def process_image(self, image):
        # Maintain aspect ratio with fixed HD resolution (1280x720)
        h, w = image.shape[:2]
        aspect_ratio = w / h

        if (aspect_ratio > 1280 / 720):
            new_width = 1280
            new_height = int(1280 / aspect_ratio)
        else:
            new_height = 720
            new_width = int(720 * aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height))

        # Scale camera matrix
        height, width = resized_image.shape[:2]
        self.camera_matrix = self.scale_camera_matrix(height, width)

        # YOLO inference
        results = self.yolo(resized_image)[0]
        print(f"Image size: {width}x{height}")
        print(f"Scaled camera matrix:\n{self.camera_matrix}")

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
                            distance, tvec, rvec, center, is_front, normal_vector = self.estimate_pose(keypoints)
                            distance_m = distance / 1000  # Convert distance to meters
                            print(f"Object {idx}: Distance = {distance_m:.4f}m, Center = {center}, Normal Vector = {normal_vector}")

                            # Draw results
                            resized_image = self.draw_results(resized_image, keypoints, distance, tvec, rvec, normal_vector, is_front)
                        except Exception as e:
                            print(f"Error in pose estimation: {str(e)}")

        return resized_image

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

        # Calculate normal vector
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        normal_vector = rotation_matrix[:, 2]  # Z-axis of the rotation matrix

        return distance, tvec, rvec, center, is_front, normal_vector

    def draw_results(self, image, keypoints, distance, tvec, rvec, normal_vector, is_front):
        # Define colors for front and side
        front_color = (0, 255, 0)  # Green
        side_color = (255, 0, 0)   # Blue

        # Select color based on the face type
        color = front_color if is_front else side_color

        # Draw keypoints
        for point in keypoints:
            cv2.circle(image, tuple(point.astype(int)), 5, color, -1)

        # Draw bones (lines between keypoints)
        for i in range(len(keypoints)):
            start_point = tuple(keypoints[i].astype(int))
            end_point = tuple(keypoints[(i + 1) % len(keypoints)].astype(int))
            cv2.line(image, start_point, end_point, color, 2)

        # Calculate the center point of the brick
        center = np.mean(keypoints, axis=0).astype(int)

        # Draw distance near the center point
        distance_m = distance / 1000  # Convert distance to meters
        text = f"{distance_m:.4f}m"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x, text_y = center[0] + 10, center[1] - 10
        cv2.rectangle(image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), -1)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw normal vector
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        normal_vector = rotation_matrix[:, 2]  # Z-axis of the rotation matrix

        # Calculate the endpoint of the normal vector for visualization
        normal_length = 100  # Length of the normal vector to be drawn
        endpoint = center + (normal_vector[:2] * normal_length).astype(int)

        # Draw the normal vector
        cv2.arrowedLine(image, tuple(center), tuple(endpoint), (0, 255, 255), 2)  # Yellow color in BGR

        # Calculate the start and end points for the normal vector line
        start_point = tuple(keypoints[0].astype(int))
        end_point = (start_point[0] + int(normal_vector[0] * 100), start_point[1] + int(normal_vector[1] * 100))

        return image

def main():
    parser = argparse.ArgumentParser(description="Brick Pose Estimation")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--input", required=True, help="Path to input file or directory")
    args = parser.parse_args()

    input_path = args.input
    estimator = BrickPoseEstimator(args.model)

    if os.path.isdir(input_path):
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_files = [input_path]

    for i, file_path in enumerate(image_files):
        image = cv2.imread(file_path)

        if image is None:
            print(f"Failed to read: {file_path}")
            continue

        result_image = estimator.process_image(image)

        # Create output filename
        output_path = f"results/result_{i:02d}.jpg"
        os.makedirs("results", exist_ok=True)

        # Save result
        cv2.imwrite(output_path, result_image)
        print(f"Processed {file_path} -> {output_path}")

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
