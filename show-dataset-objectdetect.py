import cv2
import os
import argparse
from typing import List, Tuple
import numpy as np

def load_image_paths(input_dir: str) -> List[str]:
    """
    Load image paths from the input directory.

    Args:
        input_dir (str): Path to the input directory containing 'images' and 'labels'.

    Returns:
        List[str]: List of sorted image paths.
    """
    image_dir = os.path.join(input_dir, "images")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"'images' directory not found in {input_dir}")
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
    return images

def parse_annotations(label_path: str) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Parse YOLOv5 object detection annotations from a label file.

    Args:
        label_path (str): Path to the label file.

    Returns:
        List[Tuple[int, List[Tuple[float, float]]]]: Parsed annotations with class ID and bounding polygon.
    """
    annotations = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(float(parts[0]))
            polygon = []
            for i in range(1, len(parts), 2):  # Read polygon points (x, y pairs)
                x = float(parts[i])
                y = float(parts[i + 1])
                polygon.append((x, y))
            annotations.append((class_id, polygon))
    return annotations

def display_images_with_annotations(input_dir: str):
    """
    Display images with object detection annotations.

    Args:
        input_dir (str): Path to the input directory containing 'images' and 'labels'.
    """
    image_paths = load_image_paths(input_dir)
    label_dir = os.path.join(input_dir, "labels")
    total_images = len(image_paths)

    if total_images == 0:
        print("No images found in the 'images' directory.")
        return

    current_index = 0

    while True:
        # Load the current image
        image_path = image_paths[current_index]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            break

        # Get corresponding label file
        label_path = os.path.join(label_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            annotations = parse_annotations(label_path)
            height, width = image.shape[:2]

            for class_id, polygon in annotations:
                # Denormalize polygon points and draw them
                points = [(int(x * width), int(y * height)) for x, y in polygon]
                points_np = np.array(points, dtype=np.int32)

                # Draw the polygon
                cv2.polylines(image, [points_np], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow polygon

                # Add class ID label near the first point
                if points:
                    x, y = points[0]
                    cv2.putText(image, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add text for navigation info
        label_text = f"Image {current_index + 1} of {total_images}"
        cv2.putText(image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the image
        cv2.imshow("Image Viewer with Annotations", image)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break
        elif key == ord('o'):  # 'o' for previous image
            current_index = (current_index - 1) % total_images
        elif key == ord('p'):  # 'p' for next image
            current_index = (current_index + 1) % total_images
        elif key == ord('k'):  # 'k' for first image
            current_index = 0
        elif key == ord('l'):  # 'l' for last image
            current_index = total_images - 1

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Image Viewer with Annotation Navigation")
    parser.add_argument("inputdir", type=str, help="Input directory containing 'images' and 'labels' folders")
    args = parser.parse_args()

    try:
        display_images_with_annotations(args.inputdir)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
