import os
import cv2

def flip_images(input_folder, output_folder):
    """
    Flip images vertically and save them to a new folder, preserving the directory structure.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save flipped images.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                # Construct output path while preserving directory structure
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)

                # Read the image
                image = cv2.imread(input_path)
                if image is not None:
                    # Flip the image vertically
                    flipped_image = cv2.flip(image, 0)

                    # Save the flipped image
                    cv2.imwrite(output_path, flipped_image)
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"Failed to read: {input_path}")

if __name__ == "__main__":
    # Input folder containing images
    input_folder = "dataset/rosbag2_2024_03_07-15_00_42_0"
    # Output folder to save flipped images
    output_folder = "dataset/rosbag2_2024_03_07-15_00_42_0-reversed"
    
    flip_images(input_folder, output_folder)
