import os
import argparse
import json
import cv2
import shutil
from typing import List, Dict, Tuple

def load_image_paths(input_dir: str) -> List[str]:
    """
    Load image paths from the input directory's 'images' subdirectory.

    Args:
        input_dir (str): Path to the input directory containing 'images' and 'labels'.

    Returns:
        List[str]: List of sorted image paths.
    """
    image_dir = input_dir
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"'images' directory not found in {input_dir}")
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
    return images


def parse_annotations(label_path: str) -> List[Dict]:
    """
    Parse YOLOv5 annotations from a label file.

    Args:
        label_path (str): Path to the label file.

    Returns:
        List[Dict]: List of annotations for each object in the image.
    """
    annotations = []
    with open(label_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(float(parts[0]))
            bbox = list(map(float, parts[1:5]))  # (x, y, w, h)
            keypoints = []
            for i in range(5, len(parts), 3):
                kp_x = float(parts[i])
                kp_y = float(parts[i + 1])
                visibility = int(float(parts[i + 2]))
                keypoints.append((kp_x, kp_y, visibility))
            annotations.append({
                "class_id": class_id,
                "bbox": bbox,
                "keypoints": keypoints
            })
    return annotations


def convert_to_coco_format(image_paths: List[str], label_paths: List[str], output_dir: str, dataset_type: str) -> Dict:
    """
    Convert the dataset into COCO format.

    Args:
        image_paths (List[str]): List of image file paths.
        label_paths (List[str]): List of label file paths.
        output_dir (str): Directory to save the converted annotations.
        dataset_type (str): Type of dataset (e.g., 'train', 'valid', 'test').

    Returns:
        Dict: COCO formatted dataset.
    """
    images = []
    annotations = []
    categories = []
    annotation_id = 1
    image_id = 1

    # Define categories (adjust according to your dataset)
    categories = [{"id": 0, "name": "class_name"}]  # Replace with actual class names if needed

    for image_path, label_path in zip(image_paths, label_paths):
        # Get image info
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        height, width = image.shape[:2]

        # Create image entry
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }
        images.append(image_info)

        # Parse annotations
        image_annotations = parse_annotations(label_path)

        for ann in image_annotations:
            # Bounding box [x_min, y_min, x_max, y_max]
            x_center, y_center, w, h = ann["bbox"]
            x_min = (x_center - w / 2) * width
            y_min = (y_center - h / 2) * height
            x_max = (x_center + w / 2) * width
            y_max = (y_center + h / 2) * height

            # Keypoints (flattened x, y coordinates of keypoints)
            keypoints = []
            for kp_x, kp_y, visibility in ann["keypoints"]:
                keypoints.extend([kp_x * width, kp_y * height, visibility])

            # Add annotation
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann["class_id"],
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "segmentation": [],  # Empty segmentation (optional for keypoints)
                "keypoints": keypoints,
                "num_keypoints": len(ann["keypoints"]),
                "iscrowd": 0
            }
            annotations.append(annotation_info)
            annotation_id += 1

        image_id += 1

    # COCO format final output
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save COCO formatted annotations to a JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"_annotations.coco.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_data, json_file)

    return coco_data


def copy_images(image_paths: List[str], output_dir: str, dataset_type: str):
    """
    Copy image files to the output directory.

    Args:
        image_paths (List[str]): List of image file paths.
        output_dir (str): The root output directory.
        dataset_type (str): Type of dataset (e.g., 'train', 'valid', 'test').
    """
    output_image_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(output_image_dir, exist_ok=True)

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_image_dir, image_name)
        shutil.copy(image_path, output_image_path)  # Copy image to output directory


def process_dataset(input_dir: str, output_dir: str):
    """
    Process the entire dataset and convert it to COCO format.

    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to the output directory for COCO dataset.
    """
    for dataset_type in ["train", "valid", "test"]:
        dataset_input_dir = os.path.join(input_dir, dataset_type)
        if not os.path.exists(dataset_input_dir):
            print(f"Skipping {dataset_type} - directory not found.")
            continue

        # Load image and label paths for the dataset
        print(f"Processing dataset directory: {dataset_input_dir}")
        image_paths = load_image_paths(os.path.join(dataset_input_dir, "images"))
        label_dir = os.path.join(dataset_input_dir, "labels")
        label_paths = [
            os.path.join(label_dir, os.path.basename(image_path).replace(".jpg", ".txt").replace(".png", ".txt"))
            for image_path in image_paths
        ]

        # Copy images to the output directory
        print(f"Copying images for {dataset_type} dataset...")
        copy_images(image_paths, output_dir, dataset_type)

        # Convert to COCO format and save to output directory
        print(f"Converting {dataset_type} dataset...")
        convert_to_coco_format(image_paths, label_paths, os.path.join(output_dir, dataset_type), dataset_type)
        print(f"{dataset_type} conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv5 dataset to COCO format.")
    parser.add_argument("input_dir", type=str, help="Root directory of the YOLO dataset.")
    parser.add_argument("output_dir", type=str, help="Root directory to save the converted annotations.")
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
