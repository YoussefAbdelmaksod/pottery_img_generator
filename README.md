# Pottery Image Generation with Stable Diffusion

This project leverages state-of-the-art text-to-image diffusion models (Stable Diffusion and variants) to generate high-resolution, realistic images of pottery. It is designed for designers, archaeologists, educators, artists, and 3D modelers.

---

## Project Structure

```
project-root/
├── main.py                      # CLI entry point for image generation
├── pipeline/
│   └── text2img.py              # Core text-to-image pipeline
├── data/
│   ├── download_pottery_dataset.py   # Download public pottery datasets
│   ├── dataset_utils.py              # Curation, preprocessing, CSV export
│   ├── augment_pottery_dataset.py    # Data augmentation
│   ├── split_pottery_dataset.py      # Train/val/test split
│   ├── dataset_stats.py              # Dataset statistics and quality checks
│   └── prepare_for_diffusers.py      # Prepare dataset for LoRA training
├── models/
│   ├── lora_utils.py                 # LoRA fine-tuning utilities (custom)
│   └── train_lora.py                 # (Legacy) LoRA training script
├── pottery_gradio_app.py        # Web UI for image generation
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Dev/test dependencies
├── train_dreambooth_lora.py     # Official Hugging Face LoRA training script
├── .github/
│   └── copilot-instructions.md  # Copilot workspace instructions
├── .vscode/
│   └── tasks.json               # VS Code tasks
├── LICENSE
├── README.md
└── tests/
    └── test_text2img.py         # Basic pipeline test
```

---

## Features
- Generate pottery images from natural language prompts
- Support for vessel types, materials, textures, and cultural styles
- Style consistency (e.g., Japanese Raku, Greek, Egyptian, etc.)
- Prompt engineering and LoRA fine-tuning ready
- Dataset curation, augmentation, splitting, and statistics utilities
- Web UI for non-technical users
- Basic tests and dev setup

---

## Getting Started

1. **Set up environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Download or curate a dataset**
   - Download: `python data/download_pottery_dataset.py`
   - Curate: `python data/dataset_utils.py` (see script for usage)
   - Augment: `python data/augment_pottery_dataset.py`
   - Split: `python data/split_pottery_dataset.py`
   - Stats: `python data/dataset_stats.py`
3. **Generate images**
   ```bash
   python main.py --prompt "A Japanese Raku style vase, studio lighting" --style "Japanese Raku" --material "glazed ceramic" --perspective "side view" --output pottery.png
   ```
4. **Web UI**
   ```bash
   python pottery_gradio_app.py
   ```
5. **LoRA Fine-Tuning (Cloud recommended for low VRAM)**
   - Prepare: `python data/prepare_for_diffusers.py --input_dir data/pottery --output_dir data/diffusers_pottery`
   - Download official script: `wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py`
   - Run training (see README for recommended command)

---

## CLI Options (main.py)
- `--prompt`: Text prompt for pottery image (required)
- `--output`: Output image file (default: output.png)
- `--model`: Model name or path (default: runwayml/stable-diffusion-v1-5)
- `--style`: Pottery style (e.g., Japanese Raku, Greek)
- `--material`: Material/texture (e.g., glazed ceramic, earthenware)
- `--perspective`: Camera perspective (e.g., side view, top-down)
- `--guidance_scale`: Diffusion guidance scale (default: 7.5)
- `--seed`: Random seed for reproducibility
- `--negative_prompt`: Negative prompt for image generation

---

## Portability & Collaboration
- **Yes, you can share this project with a partner.**
- All code, scripts, and instructions are included for setup, training, and usage on another machine.
- For LoRA fine-tuning, your partner should have a GPU with at least 8GB VRAM (or use cloud services like Colab/AWS).
- All dependencies are listed in `requirements.txt` and `requirements-dev.txt`.
- Dataset scripts and utilities are self-contained and documented.
- The project is MIT licensed—free for commercial and research use.

---

## Advanced Usage
- **Dataset Augmentation:** `python data/augment_pottery_dataset.py`
- **Dataset Splitting:** `python data/split_pottery_dataset.py`
- **Dataset Statistics:** `python data/dataset_stats.py`
- **Testing:** `pytest`
- **Web UI:** `python pottery_gradio_app.py`
- **LoRA Training:** See above and Hugging Face docs for latest best practices.

---

## License
This project is licensed under the MIT License. See LICENSE for details.

---

## Notes
- You will need a GPU for best performance.
- For custom fine-tuning, see `data/` and `models/` folders.
- For LoRA training, cloud GPU is recommended for low VRAM laptops.
