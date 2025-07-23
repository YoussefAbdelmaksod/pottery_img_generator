from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
import base64
import io
import os
from PIL import Image
import uuid

app = Flask(__name__)
CORS(app)

# Global pipeline variable
pipe = None

def load_model():
    global pipe
    
    # Load base Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"  # or your base model
    
    # Use CPU-compatible settings for Hugging Face Spaces
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load your LoRA weights - update path to match your structure
    pipe.load_lora_weights("./lora-output")  # Path to your LoRA files
    
    # Use DPM solver for faster generation
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to appropriate device
    pipe = pipe.to(device)
    print(f"Using {device} for inference")
    
    if device == "cuda":
        pipe.enable_memory_efficient_attention()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": pipe is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/api/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', 'blurry, bad quality, distorted')
        num_inference_steps = data.get('steps', 20)
        guidance_scale = data.get('guidance_scale', 7.5)
        width = data.get('width', 512)
        height = data.get('height', 512)
        seed = data.get('seed', None)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Set seed for reproducible results
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image with device-appropriate autocast
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if device_type == "cuda":
            with torch.autocast(device_type):
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                ).images[0]
        else:
            # CPU inference without autocast
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
        
        # Convert to base64 for JSON response
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            "prompt": prompt,
            "image": f"data:image/png;base64,{img_str}",
            "seed": seed,
            "parameters": {
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate_file', methods=['POST'])
def generate_image_file():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', 'blurry, bad quality, distorted')
        num_inference_steps = data.get('steps', 20)
        guidance_scale = data.get('guidance_scale', 7.5)
        width = data.get('width', 512)
        height = data.get('height', 512)
        seed = data.get('seed', None)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate image with device-appropriate autocast
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if device_type == "cuda":
            with torch.autocast(device_type):
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height
                ).images[0]
        else:
            # CPU inference without autocast
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]
        
        # Save to temporary file
        filename = f"generated_{uuid.uuid4()}.png"
        filepath = f"/tmp/{filename}"
        image.save(filepath)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch_generate', methods=['POST'])
def batch_generate():
    try:
        data = request.get_json()
        prompts = data.get('prompts', [])
        negative_prompt = data.get('negative_prompt', 'blurry, bad quality, distorted')
        num_inference_steps = data.get('steps', 20)
        guidance_scale = data.get('guidance_scale', 7.5)
        
        if not prompts:
            return jsonify({"error": "No prompts provided"}), 400
        
        results = []
        for i, prompt in enumerate(prompts):
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            if device_type == "cuda":
                with torch.autocast(device_type):
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    ).images[0]
            else:
                # CPU inference without autocast
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            results.append({
                "prompt": prompt,
                "image": f"data:image/png;base64,{img_str}"
            })
        
        return jsonify({"results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    app.run(host='0.0.0.0', port=7860)
