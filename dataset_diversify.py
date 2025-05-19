import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

def get_augmentations():
    return A.Compose([
        A.Rotate(limit=30, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.3),
    ])

def augment_image(image, transform):
    augmented = transform(image=image)
    return augmented["image"]

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(image, path):
    Image.fromarray(image).save(path)

def augment_dataset(dataset_path, augment_count=5):
    transform = get_augmentations()
    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        print(f"Augmenting class: {class_name}")
        image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files):
            img_path = os.path.join(class_folder, img_file)
            image = load_image(img_path)
            if image is None:
                continue

            for i in range(augment_count):
                augmented = augment_image(image, transform)
                base_name = os.path.splitext(img_file)[0]
                new_filename = f"{base_name}_aug{i}.jpg"
                save_path = os.path.join(class_folder, new_filename)
                save_image(augmented, save_path)

if __name__ == "__main__":
    dataset_dir = "dataset"  # Change if needed
    augment_dataset(dataset_dir, augment_count=7)
