#!/usr/bin/env python3
"""
Pottery Image Generator CLI Tool

A command-line interface for the Pottery Image Generator API.
"""

import argparse
import requests
import json
import base64
import sys
import os
from PIL import Image
import io
from datetime import datetime

class PotteryGeneratorCLI:
    def __init__(self, api_base, token=None):
        self.api_base = api_base.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.api_base}/api/health", headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ API Health Check:")
                print(f"   Status: {data.get('status')}")
                print(f"   Model Loaded: {data.get('model_loaded')}")
                print(f"   Device: {data.get('device')}")
                print(f"   LoRA Loaded: {data.get('lora_loaded')}")
                print(f"   PyTorch Version: {data.get('torch_version')}")
                return True
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return False
    
    def generate_image(self, prompt, negative_prompt=None, steps=20, guidance_scale=7.5, 
                      width=512, height=512, seed=None, output_file=None):
        """Generate a pottery image"""
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "blurry, bad quality, distorted, plastic, artificial",
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        print(f"🎨 Generating pottery image...")
        print(f"   Prompt: {prompt}")
        print(f"   Steps: {steps}, Guidance: {guidance_scale}")
        print(f"   Size: {width}x{height}")
        
        try:
            response = requests.post(
                f"{self.api_base}/api/generate",
                headers=self.headers,
                json=payload,
                timeout=300  # 5 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Save image
                if not output_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_prompt = "_".join(safe_prompt.split())
                    output_file = f"pottery_{safe_prompt}_{timestamp}.png"
                
                # Decode and save image
                image_data = result["image"].split(",")[1]  # Remove data:image/png;base64,
                image_bytes = base64.b64decode(image_data)
                
                with open(output_file, 'wb') as f:
                    f.write(image_bytes)
                
                print(f"✅ Image generated successfully!")
                print(f"   Saved as: {output_file}")
                print(f"   Used seed: {result.get('seed', 'Random')}")
                
                return output_file, result
                
            else:
                print(f"❌ Generation Error: {response.status_code}")
                print(f"   {response.text}")
                return None, None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def batch_generate(self, prompts, **kwargs):
        """Generate multiple images"""
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🏺 Generating image {i}/{len(prompts)}")
            output_file, result = self.generate_image(prompt, **kwargs)
            if output_file:
                results.append((output_file, result))
            else:
                print(f"❌ Failed to generate image for: {prompt}")
        
        print(f"\n✅ Batch generation completed: {len(results)}/{len(prompts)} successful")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Generate pottery images using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s health
  %(prog)s generate "elegant ceramic vase with blue patterns"
  %(prog)s generate "rustic pot" --steps 25 --guidance 8.0 --output my_pot.png
  %(prog)s batch prompts.txt --steps 20
        """
    )
    
    parser.add_argument('--api-url', default='https://youssefabdelmaksod-pottery-img-generator.hf.space',
                       help='API base URL')
    parser.add_argument('--token', help='Hugging Face token (if Space is private)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check API health')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a single image')
    generate_parser.add_argument('prompt', help='Text description of the pottery')
    generate_parser.add_argument('--negative-prompt', help='What to avoid in the image')
    generate_parser.add_argument('--steps', type=int, default=20, help='Inference steps (10-50)')
    generate_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (1-20)')
    generate_parser.add_argument('--width', type=int, default=512, help='Image width')
    generate_parser.add_argument('--height', type=int, default=512, help='Image height')
    generate_parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    generate_parser.add_argument('--output', help='Output filename')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Generate multiple images from prompts file')
    batch_parser.add_argument('prompts_file', help='Text file with one prompt per line')
    batch_parser.add_argument('--negative-prompt', help='What to avoid in images')
    batch_parser.add_argument('--steps', type=int, default=20, help='Inference steps (10-50)')
    batch_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (1-20)')
    batch_parser.add_argument('--width', type=int, default=512, help='Image width')
    batch_parser.add_argument('--height', type=int, default=512, help='Image height')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = PotteryGeneratorCLI(args.api_url, args.token)
    
    if args.command == 'health':
        cli.health_check()
        
    elif args.command == 'generate':
        cli.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
            output_file=args.output
        )
        
    elif args.command == 'batch':
        if not os.path.exists(args.prompts_file):
            print(f"❌ File not found: {args.prompts_file}")
            return
        
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            print(f"❌ No prompts found in {args.prompts_file}")
            return
        
        print(f"📋 Found {len(prompts)} prompts in {args.prompts_file}")
        
        cli.batch_generate(
            prompts=prompts,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height
        )

if __name__ == "__main__":
    main()
