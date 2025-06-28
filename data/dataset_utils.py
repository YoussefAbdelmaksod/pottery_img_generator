import os
from PIL import Image
import shutil
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

def _process_image(src_path: str, dst_path: str, image_size: Tuple[int, int]) -> Tuple[str, str]:
    try:
        img = Image.open(src_path).convert('RGB')
        img = img.resize(image_size)
        img.save(dst_path)
        label = os.path.basename(os.path.dirname(dst_path))
        return dst_path, label
    except Exception as e:
        logging.error(f"Error processing {src_path}: {e}")
        return None, None

def curate_pottery_dataset(source_dir, output_dir, image_size=(512, 512), export_labels=True, num_workers=8):
    """
    Curate and preprocess a pottery image dataset for production use.
    - Validates input directories and image files.
    - Copies and resizes images from source_dir to output_dir, organized by label.
    - Uses parallel processing for speed.
    - Optionally exports a CSV with image paths and labels.
    
    Args:
        source_dir (str): Path to the raw dataset directory.
        output_dir (str): Path to save the curated dataset.
        image_size (tuple): Target image size (width, height).
        export_labels (bool): Whether to export a CSV of image paths and labels.
        num_workers (int): Number of parallel workers for processing.
    """
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    os.makedirs(output_dir, exist_ok=True)
    label_map = {}
    tasks = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for label in os.listdir(source_dir):
            label_dir = os.path.join(source_dir, label)
            if not os.path.isdir(label_dir):
                continue
            out_label_dir = os.path.join(output_dir, label)
            os.makedirs(out_label_dir, exist_ok=True)
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(label_dir, fname)
                    dst_path = os.path.join(out_label_dir, fname)
                    tasks.append(executor.submit(_process_image, src_path, dst_path, image_size))
        for future in as_completed(tasks):
            img_path, label = future.result()
            if img_path and label:
                label_map[img_path] = label
    if export_labels and label_map:
        csv_path = os.path.join(output_dir, 'labels.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'label'])
            for img_path, label in label_map.items():
                writer.writerow([img_path, label])
        logging.info(f"Exported labels to {csv_path}")
    logging.info(f"Curated dataset at {output_dir} with {len(label_map)} images.")
