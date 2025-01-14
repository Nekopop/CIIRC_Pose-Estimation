import os
import json
import glob
import cv2
import shutil
from datetime import datetime
import yaml
from tqdm import tqdm

def create_coco_directory_structure(output_base_dir):
    """Create COCO dataset directory structure"""
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'annotations'), exist_ok=True)

def parse_yolo_pose_label(label_line, img_width, img_height):
    """Convert YOLOv8 pose labels to COCO format"""
    parts = label_line.strip().split()
    class_id = int(float(parts[0]))
    
    # Convert bounding box from YOLO format to COCO format
    x_center = float(parts[1]) * img_width
    y_center = float(parts[2]) * img_height
    width = float(parts[3]) * img_width
    height = float(parts[4]) * img_height
    x = x_center - width / 2
    y = y_center - height / 2
    
    # Convert keypoints
    keypoints = []
    for i in range(5, len(parts), 3):
        x_kp = float(parts[i]) * img_width
        y_kp = float(parts[i + 1]) * img_height
        v = int(float(parts[i + 2]))
        keypoints.extend([x_kp, y_kp, v])
    
    return {
        "category_id": class_id,
        "bbox": [x, y, width, height],
        "keypoints": keypoints,
        "num_keypoints": len(keypoints) // 3,
        "area": width * height,
        "iscrowd": 0
    }

def convert_and_copy_dataset(input_base_dir, output_base_dir):
    """Convert and copy the dataset to COCO format"""
    splits = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }
    
    for yolo_split, coco_split in splits.items():
        print(f"\nProcessing {yolo_split} split...")
        input_img_dir = os.path.join(input_base_dir, yolo_split, 'images')
        input_lbl_dir = os.path.join(input_base_dir, yolo_split, 'labels')
        output_img_dir = os.path.join(output_base_dir, coco_split, 'images')
        
        # Initialize COCO format data structure
        coco_data = {
            "info": {"description": f"Pallets and QRCode Dataset - {coco_split}",
                    "version": "1.0", "year": 2024},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 0,
                    "name": "Pallets",
                    "keypoints": ["corner1", "corner2", "corner3", "corner4"],
                    "skeleton": [[0,1], [1,2], [2,3], [3,0]]
                },
                {
                    "id": 1,
                    "name": "QRCode",
                    "keypoints": ["corner1", "corner2", "corner3", "corner4"],
                    "skeleton": [[0,1], [1,2], [2,3], [3,0]]
                }
            ]
        }
        
        image_id = 0
        annotation_id = 0
        
        # Process image files
        image_files = glob.glob(os.path.join(input_img_dir, '*.*'))
        for image_file in tqdm(image_files, desc="Converting images"):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Corresponding label file
            label_file = os.path.join(input_lbl_dir, os.path.splitext(os.path.basename(image_file))[0] + '.txt')
            if not os.path.exists(label_file):
                continue
            
            try:
                # Read the image and get dimensions
                img = cv2.imread(image_file)
                if img is None:
                    print(f"Warning: Could not read image {image_file}")
                    continue
                    
                height, width = img.shape[:2]
                
                # Copy the image
                img_filename = os.path.basename(image_file)
                output_img_path = os.path.join(output_img_dir, img_filename)
                shutil.copy2(image_file, output_img_path)
                
                # Add image information to COCO data
                image_info = {
                    "id": image_id,
                    "file_name": img_filename,
                    "width": width,
                    "height": height
                }
                coco_data["images"].append(image_info)
                
                # Process annotations
                with open(label_file, 'r') as f:
                    for line in f:
                        ann = parse_yolo_pose_label(line, width, height)
                        ann["id"] = annotation_id
                        ann["image_id"] = image_id
                        coco_data["annotations"].append(ann)
                        annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        # Save annotation JSON
        json_path = os.path.join(output_base_dir, 'annotations', f'instances_{coco_split}.json')
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Completed {coco_split} split: {image_id} images and {annotation_id} annotations")

def main():
    input_base_dir = "dataset/YOLOv11_rotated"
    output_base_dir = "coco_dataset"
    
    create_coco_directory_structure(output_base_dir)
    convert_and_copy_dataset(input_base_dir, output_base_dir)
    print("\nConversion completed successfully!")

if __name__ == "__main__":
    main()
