import os
import argparse
from typing import List, Tuple
import cv2


def parse_annotations(label_path: str) -> List[Tuple[int, List[Tuple[float, float, int]]]]:
    """
    Parse YOLOv5 keypoint annotations from a label file.
    Args:
        label_path (str): Path to the label file.
    Returns:
        List[Tuple[int, List[Tuple[float, float, int]]]]: Parsed annotations with class ID and keypoints.
    """
    annotations = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(float(parts[0]))
            keypoints = []
            for i in range(5, len(parts), 3):  # Start after bbox and read keypoints
                kp_x = float(parts[i])
                kp_y = float(parts[i + 1])
                visibility = int(float(parts[i + 2]))
                keypoints.append((kp_x, kp_y, visibility))
            annotations.append((class_id, keypoints))
    return annotations


def extract_polygon_points_normalized(
    keypoints: List[Tuple[float, float, int]], image_width: int, image_height: int
) -> List[Tuple[float, float]]:
    """
    Extract normalized polygon points from keypoints.
    Args:
        keypoints (List[Tuple[float, float, int]]): List of keypoints with normalized coordinates and visibility.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    Returns:
        List[Tuple[float, float]]: List of normalized polygon points.
    """
    polygon_points = []
    for kp_x, kp_y, _ in keypoints:
        normalized_x = kp_x * image_width / image_width
        normalized_y = kp_y * image_height / image_height
        polygon_points.append((normalized_x, normalized_y))
    return polygon_points


def process_folder(input_dir: str, output_dir: str):
    """
    Recursively process all subfolders and convert keypoint annotations to YOLO format.
    Args:
        input_dir (str): Root directory of the YOLO dataset.
        output_dir (str): Root directory to save the converted annotations.
    """
    for root, dirs, files in os.walk(input_dir):
        if "images" in dirs and "labels" in dirs:  # Check if the current folder has 'images' and 'labels'
            image_dir = os.path.join(root, "images")
            label_dir = os.path.join(root, "labels")
            output_label_dir = os.path.join(output_dir, os.path.relpath(label_dir, input_dir))
            os.makedirs(output_label_dir, exist_ok=True)

            for image_file in sorted(os.listdir(image_dir)):
                if not image_file.endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_path = os.path.join(image_dir, image_file)
                label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))
                if not os.path.exists(label_path):
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                height, width = image.shape[:2]
                annotations = parse_annotations(label_path)

                output_label_path = os.path.join(output_label_dir, os.path.basename(label_path))
                with open(output_label_path, "w") as output_file:
                    for class_id, keypoints in annotations:
                        # Extract normalized polygon points
                        polygon_points = extract_polygon_points_normalized(keypoints, width, height)
                        if len(polygon_points) >= 4:  # Ensure at least 4 points for a polygon
                            # Flatten points into normalized x1 y1 x2 y2 ... format
                            flat_points = [coord for point in polygon_points[:4] for coord in point]
                            flat_points_str = " ".join(f"{p:.6f}" for p in flat_points)  # 6 decimal places
                            # Write to output file
                            output_file.write(f"{class_id} {flat_points_str}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset keypoint annotations to object detection format.")
    parser.add_argument("input_dir", type=str, help="Root directory of the YOLO dataset.")
    parser.add_argument("output_dir", type=str, help="Root directory to save the converted annotations.")
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
