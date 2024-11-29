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

def parse_annotations(label_path: str) -> List[Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, int]]]]:
    """
    Parse YOLOv5 keypoint annotations from a label file.

    Args:
        label_path (str): Path to the label file.

    Returns:
        List[Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, int]]]]: Parsed annotations with
        class ID, bounding box, and keypoints.
    """
    annotations = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(float(parts[0]))
            bbox = tuple(map(float, parts[1:5]))  # (x, y, w, h)
            keypoints = []
            for i in range(5, len(parts), 3):  # Start after bbox and read keypoints
                kp_x = float(parts[i])
                kp_y = float(parts[i + 1])
                visibility = int(float(parts[i + 2]))
                keypoints.append((kp_x, kp_y, visibility))
            annotations.append((class_id, bbox, keypoints))
    return annotations

def display_images_with_annotations(input_dir: str):
    """
    Display images with keypoint annotations.

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

            for _, bbox, keypoints in annotations:
                # Draw bounding box
                x_center, y_center, w, h = bbox
                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)  # Cyan box

                # Draw keypoints and collect all keypoints for the polygon
                polygon_points = []
                for kp_x, kp_y, visibility in keypoints:
                    x, y = int(kp_x * width), int(kp_y * height)
                    polygon_points.append((x, y))  # Include all keypoints regardless of visibility
                    if visibility == 0:  # Not labeled (pink)
                        color = (255, 0, 255)
                    elif visibility == 1:  # Labeled but not visible (blue)
                        color = (255, 0, 0)
                    elif visibility == 2:  # Labeled and visible (green)
                        color = (0, 255, 0)
                    cv2.circle(image, (x, y), 5, color, -1)

                # Draw polygon using all keypoints
                if len(polygon_points) > 2:
                    polygon_points_np = np.array(polygon_points, dtype=np.int32)
                    cv2.polylines(image, [polygon_points_np], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow polygon

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