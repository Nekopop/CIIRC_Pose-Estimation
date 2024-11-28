import os
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def rotate_annotations(label_lines, image_shape):
    rotated_labels = []
    for line in label_lines:
        values = list(map(float, line.strip().split()))
        cls_id = int(values[0])
        
        # Bounding box
        bbox = values[1:5]  # [x_center, y_center, width, height]
        
        # Keypoints
        keypoints = values[5:]  # [x1, y1, v1, x2, y2, v2, ...]

        # Transformation for 90-degree left rotation
        x_center_rotated = bbox[1]            # y_center
        y_center_rotated = 1.0 - bbox[0]      # 1 - x_center
        width_rotated = bbox[3]               # height
        height_rotated = bbox[2]              # width

        # Rotate keypoints
        rotated_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i:i+3]
            # Apply 90-degree left rotation
            x_rotated = y
            y_rotated = 1.0 - x
            rotated_keypoints.extend([x_rotated, y_rotated, v])

        # Create new label line
        rotated_values = [cls_id, x_center_rotated, y_center_rotated, 
                         width_rotated, height_rotated] + rotated_keypoints
        rotated_line = ' '.join(map(lambda x: f"{x:.6f}", rotated_values))
        rotated_labels.append(rotated_line)

    return rotated_labels

def display_random_images(image_dir, label_dir, num_images=10):
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    plt.figure(figsize=(15, 10))
    for i, image_file in enumerate(selected_files):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_dir, f"{base_filename}.txt")

        # Load image
        image = cv2.imread(image_file)
        h, w = image.shape[:2]

        # Load labels
        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Plot annotations
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Skip invalid lines
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))

            x_center = coords[0] * w
            y_center = coords[1] * h
            width_bbox = coords[2] * w
            height_bbox = coords[3] * h

            # Draw bounding box (uncomment if needed)
            x_min = x_center - width_bbox / 2
            y_min = y_center - height_bbox / 2
            rect = plt.Rectangle((x_min, y_min), width_bbox, height_bbox,
                                 linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)

            keypoints = coords[4:]
            num_keypoints = int(len(keypoints) / 3)
            x_points = []
            y_points = []
            for k in range(num_keypoints):
                xk = keypoints[k * 3] * w
                yk = keypoints[k * 3 + 1] * h
                vk = keypoints[k * 3 + 2]
                if vk != 0:
                    x_points.append(xk)
                    y_points.append(yk)
            plt.scatter(x_points, y_points, c='r', s=10)

        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    input_image_dir = "dataset/YOLOv11//test/images"
    input_label_dir = "dataset/YOLOv11/test/labels"
    output_image_dir = "dataset/YOLOv11_rotated/test/images"
    output_label_dir = "dataset/YOLOv11_rotated/test/labels"

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_image_dir, "*.jpg"))

    for image_file in image_files:
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(input_label_dir, f"{base_filename}.txt")

        # Load image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to read image {image_file}")
            continue

        # Load labels
        if not os.path.exists(label_file):
            print(f"Label file does not exist {label_file}")
            continue

        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        if base_filename.startswith("000"):
            # Rotate image 90 degrees counterclockwise
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Update labels
            rotated_labels = rotate_annotations(label_lines, image.shape)

            # Save rotated image and labels
            output_image_path = os.path.join(output_image_dir, f"{base_filename}.jpg")
            output_label_path = os.path.join(output_label_dir, f"{base_filename}.txt")

            cv2.imwrite(output_image_path, rotated_image)

            # Preserve the original label format and save
            with open(output_label_path, 'w') as f:
                for label in rotated_labels:
                    f.write(f"{label.strip()}\n")
        else:
            # Copy image as is
            output_image_path = os.path.join(output_image_dir, f"{base_filename}.jpg")
            output_label_path = os.path.join(output_label_dir, f"{base_filename}.txt")

            cv2.imwrite(output_image_path, image)

            # Preserve the original label format and copy
            with open(output_label_path, 'w') as f:
                for line in label_lines:
                    f.write(line.strip() + '\n')

        print(f"Processed {image_file}")

    # Display 10 random images with keypoints
    display_random_images(output_image_dir, output_label_dir)

if __name__ == "__main__":
    main()

    