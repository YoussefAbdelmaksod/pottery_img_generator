# Dataset Split Script for Pottery Images
import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        images = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        for split, split_imgs in splits.items():
            split_dir = os.path.join(output_dir, split, label)
            os.makedirs(split_dir, exist_ok=True)
            for fname in split_imgs:
                src = os.path.join(label_dir, fname)
                dst = os.path.join(split_dir, fname)
                shutil.copy2(src, dst)
    print(f"Dataset split into train/val/test in {output_dir}")

if __name__ == "__main__":
    split_dataset("data/pottery_augmented", "data/pottery_split")
