import cv2
import numpy as np
from ultralytics import YOLO
import random
import glob
import os

class BrickPoseEstimator:
    def __init__(self, model_path):
        # Load YOLOv11 model
        self.yolo = YOLO(model_path)
        
        # Original camera parameters (2K: 1440x2560)
        self.original_camera_matrix = np.array([
            [2067.0886, 0.0, 1235.7012],
            [0.0, 2065.7244, 727.0466],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.original_height = 1440
        self.original_width = 2560
        
        self.dist_coeffs = np.array([7.1634, -88.0136, -0.00047, -0.00045], dtype=np.float32)
        
        # Center the Bricks model (match YOLOv11 keypoint order)
        half_w, half_h = 500, 690
        self.model_points = np.array([
            [-half_w, half_h, 0],   # Point 0: Bottom-Left - Corresponds to YOLOv11's 1st keypoint
            [half_w, half_h, 0],    # Point 1: Bottom-Right - Corresponds to YOLOv11's 2nd keypoint
            [half_w, -half_h, 0],   # Point 2: Top-Right - Corresponds to YOLOv11's 3rd keypoint
            [-half_w, -half_h, 0]   # Point 3: Top-Left - Corresponds to YOLOv11's 4th keypoint
        ], dtype=np.float32)

    def scale_camera_matrix(self, image_height, image_width):
        # Calculate scale factors
        height_scale = image_height / self.original_height
        width_scale = image_width / self.original_width
        
        # Scale camera matrix
        scaled_matrix = self.original_camera_matrix.copy()
        scaled_matrix[0,0] *= width_scale   # fx
        scaled_matrix[1,1] *= height_scale  # fy
        scaled_matrix[0,2] *= width_scale   # cx
        scaled_matrix[1,2] *= height_scale  # cy
        
        return scaled_matrix

    def process_image(self, image):
        # Maintain aspect ratio with fixed height of 640
        h, w = image.shape[:2]
        scale = 640 / h
        new_width = int(w * scale)
        resized_image = cv2.resize(image, (new_width, 640))
        
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
                            distance, tvec, rvec = self.estimate_pose(keypoints)
                            
                            # Draw results
                            resized_image = self.draw_results(resized_image, keypoints, distance, tvec, rvec)
                        except Exception as e:
                            print(f"Error in pose estimation: {str(e)}")
        
        return resized_image

    def estimate_pose(self, keypoints_2d):
        image_points = np.array(keypoints_2d, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            print("Pose estimation failed")  # Debug information
        distance = np.linalg.norm(tvec)
        return distance, tvec, rvec

    def draw_results(self, image, keypoints, distance, tvec, rvec):
        # Draw keypoints and bones
        for pt in keypoints:
            x, y = map(int, pt)
            cv2.circle(image, (x,y), 5, (0,255,0), -1)
        
        for i in range(len(keypoints)):
            pt1 = tuple(map(int, keypoints[i]))
            pt2 = tuple(map(int, keypoints[(i + 1) % len(keypoints)]))
            cv2.line(image, pt1, pt2, (0,255,0), 2)
        
        # Calculate center
        center_x = int(np.mean([pt[0] for pt in keypoints]))
        center_y = int(np.mean([pt[1] for pt in keypoints]))
        
        # Draw origin
        cv2.circle(image, (center_x, center_y), 5, (255,0,0), -1)
        
        # Prepare text
        text = f"{distance:.1f}mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Calculate text size for background
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
                   (255, 255, 255),  # White text
                   thickness)
        
        return image

# Example usage
def main():
    model_path = "YOLO11/weight/2024-11-28(2024-11-27_vertical_aug).pt"
    estimator = BrickPoseEstimator(model_path)
    
    # Get all rgb.png files
    rgb_files = glob.glob("dataset/dataset_720p/**/rgb.png", recursive=True)
    
    # Take first 10 images
    selected_files = rgb_files[:10]
    
    for i, file_path in enumerate(selected_files):
        # Read and process image
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
        
        # Display result
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Result", 800, 600)
        cv2.imshow("Result", result_image)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()