import cv2
import os
import glob

def resize_to_720p(image):
    # Target height is 720p
    target_height = 720
    h, w = image.shape[:2]
    # Calculate new width maintaining aspect ratio
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    # Resize image
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def process_dataset():
    # Input and output directories
    input_dir = "dataset/rosbag2_2024_03_07-15_00_42_0-reversed"
    output_dir = "dataset_720p"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all rgb.png files in subdirectories
    rgb_files = glob.glob(os.path.join(input_dir, "**/rgb.png"), recursive=True)
    
    for file_path in rgb_files:
        # Read image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to read: {file_path}")
            continue
            
        # Resize image
        resized = resize_to_720p(image)
        
        # Create output path
        relative_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save resized image
        cv2.imwrite(output_path, resized)
        print(f"Processed: {file_path}")

if __name__ == "__main__":
    process_dataset()