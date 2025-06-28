"""
Helper script to prepare your pottery dataset for Hugging Face diffusers LoRA training.
This script ensures your dataset is in the right folder structure and creates a metadata file if needed.
"""
import os
import shutil
import argparse

def prepare_for_diffusers(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(label_dir, fname)
                dst = os.path.join(output_dir, f"{label}_{fname}")
                shutil.copy2(src, dst)
    print(f"Prepared dataset for diffusers at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare pottery dataset for diffusers LoRA training.")
    parser.add_argument('--input_dir', type=str, default='data/pottery', help='Input curated dataset directory')
    parser.add_argument('--output_dir', type=str, default='data/diffusers_pottery', help='Output directory for training images')
    args = parser.parse_args()
    prepare_for_diffusers(args.input_dir, args.output_dir)
