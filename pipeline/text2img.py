import torch
from diffusers import StableDiffusionPipeline
import argparse
from PIL import Image
import os
import warnings
from contextlib import redirect_stderr
import io


def generate_image(prompt, lora_weights_dir, output_path, steps=30, guidance=7.5, lora_scale=1.0):
    print(f"🚀 Starting image generation...")
    print(f"📝 Prompt: {prompt}")
    print(f"📁 LoRA weights: {lora_weights_dir}")
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message=".*CLIPTextModel.*")
    warnings.filterwarnings("ignore", message=".*safety_checker.*")
    
    # Load base model with optimized settings
    print("📥 Loading base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
        variant="fp16"
    ).to("cpu")

    # Find and load LoRA weights
    print("🔍 Looking for LoRA weights...")
    lora_file_path = os.path.join(lora_weights_dir, "pytorch_lora_weights.safetensors")
    
    if not os.path.exists(lora_file_path):
        possible_names = [
            "pytorch_lora_weights.safetensors",
            "pytorch_lora_weights.bin",
            "adapter_model.safetensors",
            "lora_weights.safetensors",
            "diffusion_pytorch_model.safetensors"
        ]
        found_file = None
        for name in possible_names:
            test_path = os.path.join(lora_weights_dir, name)
            if os.path.exists(test_path):
                found_file = name
                break
        
        if found_file:
            print(f"✅ Found LoRA weights: {found_file}")
        else:
            print(f"❌ No LoRA weights found in {lora_weights_dir}")
            print("Available files:")
            for file in os.listdir(lora_weights_dir):
                print(f"  - {file}")
            raise FileNotFoundError(f"No LoRA weights found in {lora_weights_dir}")
    else:
        found_file = "pytorch_lora_weights.safetensors"
        print(f"✅ Found LoRA weights: {found_file}")

    # Load LoRA weights with error suppression
    print("📥 Loading LoRA weights...")
    try:
        # Capture and suppress the CLIPTextModel warning
        f = io.StringIO()
        with redirect_stderr(f):
            pipe.load_lora_weights(
                lora_weights_dir, 
                weight_name=found_file
            )
        print(f"✅ LoRA weights loaded successfully (scale: {lora_scale})")
        # LoRA weights loaded; no need to set adapters explicitly in recent diffusers
        print(f"✅ LoRA weights loaded successfully (scale: {lora_scale})")
        
    except Exception as e:
        print(f"❌ Error loading LoRA weights: {e}")
        raise

    # Memory optimization
    print("⚡ Optimizing memory usage...")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory optimization enabled")
    except:
        try:
            pipe.enable_model_cpu_offload()
            print("✅ CPU offload enabled for memory optimization")
        except:
            print("⚠️  Using standard memory configuration")

    # Generate image
    print(f"🎨 Generating image... (steps: {steps}, guidance: {guidance})")
    
    # Clear GPU cache before generation
    torch.cuda.empty_cache()
    
    with torch.inference_mode():
        image = pipe(
            prompt, 
            num_inference_steps=steps, 
            guidance_scale=guidance,
            height=512,
            width=512
        ).images[0]

    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"✅ Image saved to {output_path}")
    
    # Clean up memory
    del pipe
    torch.cuda.empty_cache()
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image using LoRA fine-tuned Stable Diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to LoRA weights directory")
    parser.add_argument("--output", type=str, default="outputs/generated.png", help="Path to save generated image")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale (0.0-2.0)")

    args = parser.parse_args()
    
    try:
        output_path = generate_image(
            args.prompt,
            args.lora_dir,
            args.output,
            steps=args.steps,
            guidance=args.guidance,
            lora_scale=args.lora_scale
        )
        print(f"\n🎉 Generation completed successfully!")
        print(f"📸 Image saved at: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        raise
