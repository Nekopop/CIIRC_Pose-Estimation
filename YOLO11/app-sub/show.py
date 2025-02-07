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

def display_all_images(image_dir, label_dir):
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    total_images = len(image_files)
    
    for idx, image_file in enumerate(image_files, 1):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_dir, f"{base_filename}.txt")

        image = cv2.imread(image_file)
        h, w = image.shape[:2]

        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        for line in label_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls_id = parts[0]
            coords = list(map(float, parts[1:]))

            # バウンディングボックスの描画
            x_center = coords[0] * w
            y_center = coords[1] * h
            width_bbox = coords[2] * w
            height_bbox = coords[3] * h

            x_min = int(x_center - width_bbox / 2)
            y_min = int(y_center - height_bbox / 2)
            x_max = int(x_center + width_bbox / 2)
            y_max = int(y_center + height_bbox / 2)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # キーポイントの描画
            keypoints = coords[4:]  # バウンディングボック���後のすべての座標がキーポイント
            for k in range(0, len(keypoints), 3):
                x = int(keypoints[k] * w)
                y = int(keypoints[k + 1] * h)
                v = keypoints[k + 2]
                
                if v > 0:  # 可視のキーポイントのみ描画
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                    cv2.putText(image, str(k//3), (x+5, y+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 画像番号を表示
        cv2.putText(image, f'Image: {idx}/{total_images}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Image Viewer', image)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

def main():
    image_dir = "dataset/YOLOv11/1127_vertical_aug/train/images"
    label_dir = "dataset/YOLOv11/1127_vertical_aug/train/labels"
    display_random_images(image_dir, label_dir)
    display_all_images(image_dir, label_dir)

if __name__ == "__main__":
    main()
