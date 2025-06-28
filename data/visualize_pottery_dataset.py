# Visualize and analyze the pottery dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def get_image_paths_and_labels(root_dir):
    image_paths = []
    labels = []
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if os.path.isdir(label_dir):
            for fname in os.listdir(label_dir):
                if fname.endswith('.png'):
                    image_paths.append(os.path.join(label_dir, fname))
                    labels.append(label)
    return image_paths, labels

def plot_image_grid(image_paths, n=12):
    plt.figure(figsize=(12, 6))
    for i, img_path in enumerate(image_paths[:n]):
        img = Image.open(img_path)
        plt.subplot(3, 4, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(os.path.dirname(img_path)))
    plt.tight_layout()
    plt.show()

def plot_label_distribution(labels):
    counter = Counter(labels)
    plt.figure(figsize=(8, 4))
    plt.bar(counter.keys(), counter.values())
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    root = "data/pottery"
    image_paths, labels = get_image_paths_and_labels(root)
    print(f"Found {len(image_paths)} images across {len(set(labels))} labels.")
    plot_label_distribution(labels)
    plot_image_grid(image_paths)
