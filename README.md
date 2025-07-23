---
title: Pottery Image Generator
emoji: 🏺
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# 🏺 Pottery Image Generator

A fine-tuned Stable Diffusion model specialized in generating high-quality pottery and ceramic images. This model uses LoRA (Low-Rank Adaptation) to create beautiful, detailed pottery designs based on text prompts.

## 🚀 Quick Start

### Public Access
If the Space is public, you can directly access the API:
```bash
curl https://youssefabdelmaksod-pottery-img-generator.hf.space/api/health
```

### Private Access (Authentication Required)
If the Space is private, include your Hugging Face token:
```bash
curl -H "Authorization: Bearer YOUR_HF_TOKEN" \
  https://youssefabdelmaksod-pottery-img-generator.hf.space/api/health
```

## 📖 API Documentation

### Base URL
```
https://youssefabdelmaksod-pottery-img-generator.hf.space
```

### Authentication
For private Spaces, include the authorization header:
```
Authorization: Bearer YOUR_HUGGING_FACE_TOKEN
```

---

## 🔗 API Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Pottery Image Generator API",
  "endpoints": {
    "health": "/api/health",
    "generate": "/api/generate (POST)",
    "generate_file": "/api/generate_file (POST)",
    "batch_generate": "/api/batch_generate (POST)"
  },
  "status": "running"
}
```

---

### 2. Health Check
**GET** `/api/health`

Check the API status and model readiness.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "torch_version": "2.7.1+cu126",
  "lora_loaded": true
}
```

---

### 3. Generate Image (Base64)
**POST** `/api/generate`

Generate a pottery image and return it as base64-encoded data.

**Request Body:**
```json
{
  "prompt": "a beautiful ceramic vase with blue patterns",
  "negative_prompt": "blurry, bad quality, distorted",
  "steps": 20,
  "guidance_scale": 7.5,
  "width": 512,
  "height": 512,
  "seed": 42
}
```

**Parameters:**
- `prompt` (string, required): Description of the pottery you want to generate
- `negative_prompt` (string, optional): What to avoid in the image (default: "blurry, bad quality, distorted")
- `steps` (integer, optional): Number of inference steps (default: 20, range: 1-50)
- `guidance_scale` (float, optional): How closely to follow the prompt (default: 7.5, range: 1-20)
- `width` (integer, optional): Image width in pixels (default: 512)
- `height` (integer, optional): Image height in pixels (default: 512)
- `seed` (integer, optional): Random seed for reproducible results

**Response:**
```json
{
  "prompt": "a beautiful ceramic vase with blue patterns",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...",
  "seed": 42,
  "parameters": {
    "steps": 20,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512
  }
}
```

---

### 4. Generate Image (File Download)
**POST** `/api/generate_file`

Generate a pottery image and return it as a downloadable file.

**Request Body:** Same as `/api/generate`

**Response:** PNG image file download

---

### 5. Batch Generate
**POST** `/api/batch_generate`

Generate multiple pottery images from different prompts.

**Request Body:**
```json
{
  "prompts": [
    "ceramic bowl with geometric patterns",
    "terracotta pot with floral designs",
    "modern minimalist vase"
  ],
  "negative_prompt": "blurry, bad quality",
  "steps": 20,
  "guidance_scale": 7.5
}
```

**Response:**
```json
{
  "results": [
    {
      "prompt": "ceramic bowl with geometric patterns",
      "image": "data:image/png;base64,..."
    },
    {
      "prompt": "terracotta pot with floral designs", 
      "image": "data:image/png;base64,..."
    }
  ]
}
```

---

## 💡 Usage Examples

### Example 1: Basic Image Generation
```bash
curl -X POST "https://youssefabdelmaksod-pottery-img-generator.hf.space/api/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "elegant ceramic vase with intricate blue and white patterns",
    "steps": 25,
    "guidance_scale": 8.0
  }'
```

### Example 2: High-Quality Generation
```bash
curl -X POST "https://youssefabdelmaksod-pottery-img-generator.hf.space/api/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "handcrafted pottery bowl with earthy glazes and rustic texture",
    "negative_prompt": "plastic, artificial, low quality, blurry",
    "steps": 30,
    "guidance_scale": 9.0,
    "seed": 123
  }'
```

### Example 3: Batch Generation
```bash
curl -X POST "https://youssefabdelmaksod-pottery-img-generator.hf.space/api/batch_generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompts": [
      "modern ceramic lamp with clean lines",
      "traditional Japanese tea cup",
      "decorative pottery plate with mandala patterns"
    ],
    "steps": 20
  }'
```

### Example 4: Download Image File
```bash
curl -X POST "https://youssefabdelmaksod-pottery-img-generator.hf.space/api/generate_file" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "artisan ceramic pitcher with handle",
    "steps": 25
  }' \
  --output pottery_image.png
```

---

## 🎨 Prompt Tips

### Good Prompts for Pottery:
- "handcrafted ceramic bowl with earth tones"
- "elegant porcelain vase with delicate floral patterns"  
- "rustic terracotta pot with textured surface"
- "modern minimalist ceramic sculpture"
- "traditional pottery with geometric tribal patterns"

### Style Keywords:
- **Materials**: ceramic, porcelain, terracotta, stoneware, earthenware
- **Styles**: modern, traditional, rustic, elegant, minimalist, decorative
- **Patterns**: geometric, floral, tribal, mandala, abstract, striped
- **Colors**: earth tones, blue and white, monochrome, glazed, matte
- **Textures**: smooth, textured, crackled, glossy, rustic

### Negative Prompt Suggestions:
- "plastic, artificial, low quality, blurry, distorted"
- "broken, cracked, damaged, poor craftsmanship"
- "cartoon, unrealistic, oversaturated"

---

## ⚙️ Technical Details

- **Base Model**: Stable Diffusion v1.5
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Scheduler**: DPM Solver Multistep
- **Device**: CPU optimized for Hugging Face Spaces
- **Image Format**: PNG
- **Default Resolution**: 512x512 pixels

---

## 🔧 Error Handling

### Common Error Responses:

**400 Bad Request:**
```json
{
  "error": "No prompt provided"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Model inference failed"
}
```

### Response Times:
- Health check: < 1 second
- Image generation: 30-120 seconds (depending on steps and complexity)
- Batch generation: Proportional to number of images

---

## 📝 Notes

- The model is specialized for pottery and ceramic objects
- Higher `steps` values (20-30) produce better quality but take longer
- `guidance_scale` between 7-10 usually works best
- Use specific, descriptive prompts for best results
- Seed values enable reproducible generation

---

## 🤝 Support

For issues or questions, please check the Space logs or contact the Space owner.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
