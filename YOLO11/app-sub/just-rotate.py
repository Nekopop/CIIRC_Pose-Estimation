import os
from PIL import Image

def rotate_image(image_path, save_path):
    with Image.open(image_path) as img:
        rotated_img = img.rotate(180)
        rotated_img.save(save_path)

def process_directory(directory, base_dir):
    for entry in os.scandir(directory):
        if entry.is_dir():
            process_directory(entry.path, base_dir)
        elif entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            relative_path = os.path.relpath(entry.path, base_dir)
            save_path = os.path.join(base_dir, 'rotated', relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            rotate_image(entry.path, save_path)

if __name__ == "__main__":
    directory = r"dataset_720p"
    process_directory(directory, directory)