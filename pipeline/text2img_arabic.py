import torch
from diffusers import StableDiffusionPipeline
import argparse
from PIL import Image
import os
import warnings
from contextlib import redirect_stderr
import io
import re
from googletrans import Translator


def detect_and_translate_prompt(prompt, target_lang='en'):
    """
    Detect if prompt contains Arabic/non-English text and translate if needed
    """
    translator = Translator()
    
    # Check if prompt contains Arabic characters
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    has_arabic = bool(arabic_pattern.search(prompt))
    
    if has_arabic:
        try:
            print(f"🔍 Detected Arabic text in prompt")
            print(f"📝 Original prompt: {prompt}")
            
            # Translate to English
            translated = translator.translate(prompt, src='ar', dest=target_lang)
            translated_text = translated.text
            
            print(f"🔄 Translated to English: {translated_text}")
            return translated_text, True
            
        except Exception as e:
            print(f"⚠️  Translation failed: {e}")
            print(f"🔄 Using original prompt: {prompt}")
            return prompt, False
    else:
        # Check if it's another non-English language
        try:
            detected = translator.detect(prompt)
            if detected.lang != 'en' and detected.confidence > 0.5:
                print(f"🔍 Detected {detected.lang} text (confidence: {detected.confidence:.2f})")
                print(f"📝 Original prompt: {prompt}")
                
                translated = translator.translate(prompt, dest=target_lang)
                translated_text = translated.text
                
                print(f"🔄 Translated to English: {translated_text}")
                return translated_text, True
        except:
            pass
    
    return prompt, False


def generate_image(prompt, lora_weights_dir, output_path, steps=30, guidance=7.5, lora_scale=1.0, translate=True):
    print(f"🚀 Starting image generation...")
    
    # Translate prompt if needed
    original_prompt = prompt
    if translate:
        prompt, was_translated = detect_and_translate_prompt(prompt)
        if was_translated:
            print(f"✅ Using translated prompt for generation")
    
    print(f"📝 Final prompt: {prompt}")
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
    ).to("cuda")

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
                weight_name=found_file,
                adapter_name="custom_lora"
            )
        
        # Set LoRA scale
        pipe.set_adapters(["custom_lora"], adapter_weights=[lora_scale])
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
    parser.add_argument("--no_translate", action="store_true", help="Disable automatic translation")

    args = parser.parse_args()
    
    try:
        output_path = generate_image(
            args.prompt,
            args.lora_dir,
            args.output,
            steps=args.steps,
            guidance=args.guidance,
            lora_scale=args.lora_scale,
            translate=not args.no_translate
        )
        print(f"\n🎉 Generation completed successfully!")
        print(f"📸 Image saved at: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        raise
