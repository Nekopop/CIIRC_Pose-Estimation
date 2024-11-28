import os
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def display_random_images(image_dir, label_dir, num_images=15):
    # Get all image files in the directory
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    plt.figure(figsize=(15, 10))
    for i, image_file in enumerate(selected_files):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_dir, f"{base_filename}.txt")

        # Load the image
        image = cv2.imread(image_file)
        h, w = image.shape[:2]

        # Load the label
        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        plt.subplot(3, 5, i + 1)
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

            # Draw the bounding box (uncomment if needed)
            # x_min = x_center - width_bbox / 2
            # y_min = y_center - height_bbox / 2
            # rect = plt.Rectangle((x_min, y_min), width_bbox, height_bbox,
            #                      linewidth=1, edgecolor='g', facecolor='none')
            # plt.gca().add_patch(rect)

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

        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    image_dir = "dataset/YOLOv11/1127_vertical_aug/train/images"
    label_dir = "dataset/YOLOv11/1127_vertical_aug/train/labels"
    display_random_images(image_dir, label_dir)

if __name__ == "__main__":
    main()
