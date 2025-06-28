# Dataset Augmentation Script for Pottery Images
import os
from PIL import Image, ImageOps, ImageEnhance
import random

def augment_image(img):
    """Apply random augmentations to an image."""
    # Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    img = img.rotate(angle)
    # Random color jitter
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    # Random brightness
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img

def augment_dataset(input_dir, output_dir, num_augments=2):
    os.makedirs(output_dir, exist_ok=True)
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        out_label_dir = os.path.join(output_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(label_dir, fname)
                try:
                    img = Image.open(src_path).convert('RGB')
                    for i in range(num_augments):
                        aug_img = augment_image(img)
                        aug_name = f"aug_{i}_" + fname
                        aug_img.save(os.path.join(out_label_dir, aug_name))
                except Exception as e:
                    print(f"Error augmenting {src_path}: {e}")
    print(f"Augmented dataset saved to {output_dir}")

if __name__ == "__main__":
    augment_dataset("data/pottery", "data/pottery_augmented", num_augments=3)
