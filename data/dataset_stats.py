# Dataset Statistics and Quality Check Script
import os
from PIL import Image
from collections import Counter

def dataset_stats(input_dir):
    label_counts = Counter()
    image_sizes = []
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(label_dir, fname)
                try:
                    img = Image.open(path)
                    image_sizes.append(img.size)
                    label_counts[label] += 1
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    if image_sizes:
        widths, heights = zip(*image_sizes)
        print(f"Image size (WxH) - min: {min(widths)}x{min(heights)}, max: {max(widths)}x{max(heights)}, avg: {sum(widths)//len(widths)}x{sum(heights)//len(heights)}")
    else:
        print("No images found.")

if __name__ == "__main__":
    dataset_stats("data/pottery_split/train")
