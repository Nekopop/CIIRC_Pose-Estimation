import os
import glob
import shutil
import random
import matplotlib.pyplot as plt
import cv2

def rotate_labels(label_lines):
    rotated_labels = []
    for line in label_lines:
        values = line.strip().split()
        # Parse class ID as integer
        cls_id = int(values[0])
        # Parse bounding box coordinates as floats
        bbox = list(map(float, values[1:5]))
        # Parse keypoints (coordinates as floats, visibility as integers)
        keypoints = []
        i = 5
        while i < len(values):
            x = float(values[i])
            y = float(values[i + 1])
            v = int(values[i + 2])
            keypoints.extend([x, y, v])
            i += 3

        # 90-degree left rotation transformation
        x_center_rotated = bbox[1]          # y_center
        y_center_rotated = 1.0 - bbox[0]    # 1 - x_center
        width_rotated = bbox[3]             # height
        height_rotated = bbox[2]            # width

        # Rotate keypoints
        rotated_keypoints = []
        for j in range(0, len(keypoints), 3):
            xk, yk, vk = keypoints[j], keypoints[j + 1], keypoints[j + 2]
            # Apply 90-degree left rotation
            x_rotated = yk
            y_rotated = 1.0 - xk
            rotated_keypoints.extend([x_rotated, y_rotated, vk])

        # Prepare rotated values
        rotated_values = [cls_id, x_center_rotated, y_center_rotated,
                          width_rotated, height_rotated] + rotated_keypoints

        # Format values while preserving data types
        formatted_values = [str(cls_id)]  # Class ID as integer
        formatted_values += [f"{v:.6f}" for v in [x_center_rotated, y_center_rotated, width_rotated, height_rotated]]
        for k in range(0, len(rotated_keypoints), 3):
            xk = rotated_keypoints[k]
            yk = rotated_keypoints[k + 1]
            vk = int(rotated_keypoints[k + 2])  # Visibility as integer
            formatted_values.extend([f"{xk:.6f}", f"{yk:.6f}", str(vk)])

        rotated_line = ' '.join(formatted_values)
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
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                label_lines = f.readlines()
        else:
            label_lines = []

        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Plot annotations
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Skip invalid lines
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            x_center = coords[0] * w
            y_center = coords[1] * h
            width_bbox = coords[2] * w
            height_bbox = coords[3] * h

            # Draw bounding box
            x_min = x_center - width_bbox / 2
            y_min = y_center - height_bbox / 2
            rect = plt.Rectangle((x_min, y_min), width_bbox, height_bbox,
                                 linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)

            keypoints = coords[4:]
            num_keypoints = int(len(keypoints) / 3)
            for k in range(num_keypoints):
                xk = keypoints[k * 3] * w
                yk = keypoints[k * 3 + 1] * h
                vk = int(keypoints[k * 3 + 2])  # Visibility
                if vk > 0:
                    plt.plot(xk, yk, 'ro')

    plt.show()

def main():
    input_image_dir = 'dataset/YOLOv11/original/test/images'      # Replace with your input images directory
    input_label_dir = 'dataset/YOLOv11/original/test/labels'      # Replace with your input labels directory
    output_image_dir = 'rotated/images'    # Replace with your output images directory
    output_label_dir = 'rotated/labels'    # Replace with your output labels directory

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Get all image files (process all images)
    all_image_files = glob.glob(os.path.join(input_image_dir, "*.jpg"))

    # Only process image files starting with '000'
    rotated_image_files = [f for f in all_image_files if os.path.basename(f).startswith('000')]

    for image_file in all_image_files:
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(input_label_dir, f"{base_filename}.txt")

        if image_file in rotated_image_files:
            # Load image
            image = cv2.imread(image_file)
            h, w = image.shape[:2]

            # Rotate image
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Rotate labels
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label_lines = f.readlines()
                rotated_label_lines = rotate_labels(label_lines)
            else:
                rotated_label_lines = []

            # Save rotated image and labels
            rotated_image_file = os.path.join(output_image_dir, f"{base_filename}_rotated.jpg")
            rotated_label_file = os.path.join(output_label_dir, f"{base_filename}_rotated.txt")

            cv2.imwrite(rotated_image_file, rotated_image)
            with open(rotated_label_file, 'w') as f:
                f.write('\n'.join(rotated_label_lines))
        else:
            # Copy image and label without modification
            output_image_file = os.path.join(output_image_dir, f"{base_filename}.jpg")
            shutil.copyfile(image_file, output_image_file)

            if os.path.exists(label_file):
                output_label_file = os.path.join(output_label_dir, f"{base_filename}.txt")
                shutil.copyfile(label_file, output_label_file)

    # Optionally display random images to verify
    display_random_images(output_image_dir, output_label_dir)

if __name__ == "__main__":
    main()