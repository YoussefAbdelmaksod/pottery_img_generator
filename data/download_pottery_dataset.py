# Download and prepare a public pottery dataset from Hugging Face
from datasets import load_dataset
import os

def download_pottery_dataset(dataset_name="yassamina/pottery_dataset", split="train", output_dir="data/pottery"):
    """
    Download pottery images and metadata from Hugging Face Datasets.
    """
    ds = load_dataset(dataset_name, split=split)
    os.makedirs(output_dir, exist_ok=True)
    for i, item in enumerate(ds):
        img = item.get('image', None)
        label = item.get('label', 'unknown')
        if img is not None:
            label_dir = os.path.join(output_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            img_path = os.path.join(label_dir, f"pottery_{i}.png")
            img.save(img_path)
    print(f"Downloaded {len(ds)} images to {output_dir}")

if __name__ == "__main__":
    download_pottery_dataset()
