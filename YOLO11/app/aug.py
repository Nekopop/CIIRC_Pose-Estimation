import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def augment_image(image, label_lines):
    augmented_images = []
    augmented_labels = []

    # Add the original image and labels
    augmented_images.append(image)
    augmented_labels.append(label_lines)

    # Horizontal flip
    flipped_image = cv2.flip(image, 1)
    flipped_labels = flip_annotations(label_lines)

    augmented_images.append(flipped_image)
    augmented_labels.append(flipped_labels)

    return augmented_images, augmented_labels

def save_augmented_images(images, labels, base_filename, output_image_dir, output_label_dir):
    for i, (img, lbls) in enumerate(zip(images, labels)):
        output_image_path = os.path.join(output_image_dir, f"{base_filename}_aug_{i}.jpg")
        output_label_path = os.path.join(output_label_dir, f"{base_filename}_aug_{i}.txt")

        # Save the image
        cv2.imwrite(output_image_path, img)

        # Save the labels
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(lbls))

def display_augmented_images(images, labels):
    plt.figure(figsize=(15, 10))
    num_images = len(images)
    for i in range(num_images):
        img = images[i]
        lbls = labels[i]
        h, w = img.shape[:2]
        plt.subplot(3, 5, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Plot the annotations
        for line in lbls:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Skip invalid lines
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))

            x_center = coords[0] * w
            y_center = coords[1] * h
            width_bbox = coords[2] * w
            height_bbox = coords[3] * h

            # Draw the bounding box (uncomment if needed)
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

def flip_annotations(label_lines):
    flipped_labels = []
    for line in label_lines:
        parts = line.strip().split()
        cls_id = parts[0]
        coords = list(map(float, parts[1:]))

        # Extract bounding box and keypoints
        x_center, y_center, width, height = coords[:4]
        keypoints = coords[4:]

        # Flip the bounding box horizontally
        x_center_flipped = 1.0 - x_center
        
        # Rearrange keypoint order
        num_keypoints = int(len(keypoints) / 3)
        flipped_keypoints = []
        
        # Keypoint order: [0,1,2,3] → [1,0,3,2]
        keypoint_order = [1,0,3,2]  # New order
        
        for new_idx in keypoint_order:
            old_x = keypoints[new_idx * 3]
            old_y = keypoints[new_idx * 3 + 1]
            old_v = keypoints[new_idx * 3 + 2]
            
            # Flip x-coordinate
            new_x = 1.0 - old_x
            
            flipped_keypoints.extend([new_x, old_y, old_v])

        # Reconstruct flipped coordinates
        flipped_coords = [x_center_flipped, y_center, width, height] + flipped_keypoints
        flipped_line = ' '.join([cls_id] + [f"{c:.6f}" for c in flipped_coords])
        flipped_labels.append(flipped_line)
    
    return flipped_labels

def display_comparison_images(original_images, original_labels, flipped_images, flipped_labels, num_pairs=5):
    # Randomly select images to display
    num_images = min(num_pairs, len(original_images))
    indices = random.sample(range(len(original_images)), num_images)
    
    plt.figure(figsize=(15, 3*num_images))
    for idx, i in enumerate(indices):
        # Original image
        plt.subplot(num_images, 2, 2*idx + 1)
        plt.imshow(cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB))
        
        # Plot keypoints
        h, w = original_images[i].shape[:2]
        for line in original_labels[i]:
            parts = line.strip().split()
            coords = list(map(float, parts[1:]))
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
                    plt.annotate(str(k), (xk, yk), xytext=(5, 5), textcoords='offset points')
            plt.scatter(x_points, y_points, c='r', s=30)
        plt.title('Original')
        plt.axis('off')
        
        # Flipped image
        plt.subplot(num_images, 2, 2*idx + 2)
        plt.imshow(cv2.cvtColor(flipped_images[i], cv2.COLOR_BGR2RGB))
        
        # Plot keypoints
        h, w = flipped_images[i].shape[:2]
        for line in flipped_labels[i]:
            parts = line.strip().split()
            coords = list(map(float, parts[1:]))
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
                    plt.annotate(str(k), (xk, yk), xytext=(5, 5), textcoords='offset points')
            plt.scatter(x_points, y_points, c='r', s=30)
        plt.title('Flipped')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    input_image_dir = "dataset/YOLOv11_rotated/test/images"
    input_label_dir = "dataset/YOLOv11_rotated/test/labels"
    output_image_dir = "dataset/YOLOv11_augmented/vertical_flip/test/images"
    output_label_dir = "dataset/YOLOv11_augmented/vertical_flip/test/labels"

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_image_dir, "*.jpg"))

    all_augmented_images = []
    all_augmented_labels = []

    all_original_images = []
    all_original_labels = []
    all_flipped_images = []
    all_flipped_labels = []

    for image_file in image_files:
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(input_label_dir, f"{base_filename}.txt")

        # Load the image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to read image {image_file}")
            continue

        # Load the label
        if not os.path.exists(label_file):
            print(f"Label file does not exist {label_file}")
            continue

        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        # Create only horizontally flipped images and labels
        flipped_image = cv2.flip(image, 1)
        flipped_labels = flip_annotations(label_lines)

        # Save the flipped image and labels
        output_image_path = os.path.join(output_image_dir, f"{base_filename}_flip.jpg")
        output_label_path = os.path.join(output_label_dir, f"{base_filename}_flip.txt")

        cv2.imwrite(output_image_path, flipped_image)

        with open(output_label_path, 'w') as f:
            f.write('\n'.join(flipped_labels))

        print(f"Processed {image_file}")

        # Collect results for display (only the first 15 images)
        if len(all_augmented_images) < 15:
            all_augmented_images.append(flipped_image)
            all_augmented_labels.append(flipped_labels)

        # Collect for comparison
        all_original_images.append(image)
        all_original_labels.append(label_lines)
        all_flipped_images.append(flipped_image)
        all_flipped_labels.append(flipped_labels)

    # Visualize original and augmented images with keypoints
    display_augmented_images(all_augmented_images, all_augmented_labels)

    # Display comparison between original and flipped images
    display_comparison_images(all_original_images, all_original_labels, all_flipped_images, all_flipped_labels)

if __name__ == "__main__":
    main()
