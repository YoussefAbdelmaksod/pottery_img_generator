# 🏺 Pottery Image Generator - Complete Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Authentication](#authentication) 
3. [API Reference](#api-reference)
4. [Code Examples](#code-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### What is this?
The Pottery Image Generator is an AI model that creates beautiful pottery and ceramic images from text descriptions. It's based on Stable Diffusion and fine-tuned specifically for pottery using LoRA (Low-Rank Adaptation).

### Quick Test
```bash
# Check if the API is working
curl https://youssefabdelmaksod-pottery-img-generator.hf.space/api/health
```

---

## Authentication

### For Private Spaces
If the Hugging Face Space is set to private, you need a token:

1. **Get your token**: Go to https://huggingface.co/settings/tokens
2. **Create a new token** with read access
3. **Use it in requests**:
   ```bash
   curl -H "Authorization: Bearer hf_YOUR_TOKEN_HERE" \
     https://youssefabdelmaksod-pottery-img-generator.hf.space/api/health
   ```

### For Public Spaces
No authentication needed - just use the API directly.

---

## API Reference

### Base URL
```
https://youssefabdelmaksod-pottery-img-generator.hf.space
```

### Headers
```
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN  # Only if Space is private
```

---

## Code Examples

### Python Example
```python
import requests
import json
import base64
from PIL import Image
import io

# Configuration
API_BASE = "https://youssefabdelmaksod-pottery-img-generator.hf.space"
TOKEN = "hf_YOUR_TOKEN_HERE"  # Only needed if Space is private

# Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"  # Remove if Space is public
}

def generate_pottery_image(prompt, steps=20, guidance_scale=7.5):
    """Generate a pottery image from a text prompt"""
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, bad quality, distorted, plastic, artificial",
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": 512,
        "height": 512
    }
    
    response = requests.post(
        f"{API_BASE}/api/generate",
        headers=headers,
        json=payload,
        timeout=300  # 5 minutes timeout
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Decode base64 image
        image_data = result["image"].split(",")[1]  # Remove data:image/png;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        return image, result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None, None

# Example usage
image, result = generate_pottery_image(
    "elegant ceramic vase with intricate blue and white patterns",
    steps=25,
    guidance_scale=8.0
)

if image:
    image.save("pottery_vase.png")
    print(f"Image saved! Used seed: {result.get('seed')}")
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');
const fs = require('fs');

const API_BASE = 'https://youssefabdelmaksod-pottery-img-generator.hf.space';
const TOKEN = 'hf_YOUR_TOKEN_HERE'; // Only needed if Space is private

async function generatePotteryImage(prompt, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${TOKEN}` // Remove if Space is public
    };
    
    const payload = {
        prompt: prompt,
        negative_prompt: options.negativePrompt || 'blurry, bad quality, distorted',
        steps: options.steps || 20,
        guidance_scale: options.guidanceScale || 7.5,
        width: options.width || 512,
        height: options.height || 512,
        seed: options.seed
    };
    
    try {
        const response = await axios.post(
            `${API_BASE}/api/generate`,
            payload,
            { headers, timeout: 300000 } // 5 minutes timeout
        );
        
        // Save image to file
        const imageData = response.data.image.split(',')[1]; // Remove data:image/png;base64,
        const buffer = Buffer.from(imageData, 'base64');
        fs.writeFileSync('pottery_image.png', buffer);
        
        console.log('Image generated and saved!');
        return response.data;
        
    } catch (error) {
        console.error('Error generating image:', error.response?.data || error.message);
        return null;
    }
}

// Example usage
generatePotteryImage(
    'rustic terracotta pot with textured surface and earth tones',
    { steps: 25, guidanceScale: 8.0 }
);
```

### cURL Examples

#### Basic Generation
```bash
curl -X POST "https://youssefabdelmaksod-pottery-img-generator.hf.space/api/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "modern ceramic bowl with geometric patterns",
    "steps": 20,
    "guidance_scale": 7.5
  }' | jq '.image' | sed 's/"//g' | sed 's/data:image\/png;base64,//' | base64 -d > pottery.png
```

#### Batch Generation
```bash
curl -X POST "https://youssefabdelmaksod-pottery-img-generator.hf.space/api/batch_generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompts": [
      "elegant porcelain tea set",
      "rustic ceramic pitcher",
      "decorative pottery vase with floral motifs"
    ],
    "steps": 20
  }'
```

#### Download as File
```bash
curl -X POST "https://youssefabdelmaksod-pottery-img-generator.hf.space/api/generate_file" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "handcrafted ceramic mug with handle",
    "steps": 25
  }' \
  --output my_pottery.png
```

---

## Best Practices

### Prompt Engineering

#### ✅ Good Prompts
- Be specific about the type of pottery
- Include material descriptions
- Mention style and era
- Add details about patterns or decorations

**Examples:**
- ✅ "handcrafted ceramic bowl with matte glaze and geometric patterns"
- ✅ "traditional Japanese raku pottery with crackled surface"
- ✅ "modern minimalist porcelain vase with clean lines"

#### ❌ Avoid
- Generic prompts like "pottery" or "vase"
- Contradictory descriptions
- Too many unrelated elements

### Parameter Optimization

#### Steps (inference steps)
- **10-15**: Fast generation, lower quality
- **20-25**: Balanced quality and speed ⭐ **Recommended**
- **30-50**: High quality, slower generation

#### Guidance Scale
- **5-7**: More creative, less adherence to prompt
- **7.5-10**: Balanced ⭐ **Recommended**
- **10-15**: Very literal, might be over-processed

#### Resolution
- **512x512**: Standard, fast ⭐ **Recommended**
- **768x768**: Higher detail, slower
- Custom ratios work but may produce unexpected results

### Performance Tips

1. **Use consistent seeds** for reproducible results
2. **Batch process** multiple images for efficiency
3. **Cache results** to avoid regenerating identical prompts
4. **Use negative prompts** to avoid unwanted elements

---

## Troubleshooting

### Common Issues

#### 1. "404 Not Found"
**Problem**: Space URL is incorrect or Space is not running
**Solution**: 
- Check the Space status at https://huggingface.co/spaces/YoussefAbdelmaksod/pottery_img_generator
- Verify the URL is correct

#### 2. "401 Unauthorized" 
**Problem**: Missing or invalid authentication token
**Solution**:
- Get a valid token from https://huggingface.co/settings/tokens
- Include it in the Authorization header

#### 3. "400 Bad Request - No prompt provided"
**Problem**: Empty or missing prompt
**Solution**: Ensure your request includes a non-empty "prompt" field

#### 4. "500 Internal Server Error"
**Problem**: Model inference failed
**Solution**: 
- Check Space logs for details
- Try with a simpler prompt
- Reduce the number of steps

#### 5. Timeout Errors
**Problem**: Request takes too long
**Solution**:
- Reduce the number of inference steps
- Increase client timeout settings
- Try again later if Space is under heavy load

### Getting Help

1. **Check Space logs**: Look at the runtime logs in the Space interface
2. **Test with simple prompts**: Start with basic prompts to verify functionality
3. **Monitor Space status**: Ensure the Space is running and not sleeping

### Rate Limits

- No official rate limits, but be respectful
- Avoid concurrent requests to prevent overloading
- Consider caching results for repeated prompts

---

## Advanced Usage

### Custom Integration
```python
class PotteryGenerator:
    def __init__(self, api_base, token=None):
        self.api_base = api_base
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def health_check(self):
        response = requests.get(f"{self.api_base}/api/health", headers=self.headers)
        return response.json() if response.status_code == 200 else None
    
    def generate(self, prompt, **kwargs):
        payload = {"prompt": prompt, **kwargs}
        response = requests.post(
            f"{self.api_base}/api/generate",
            headers=self.headers,
            json=payload,
            timeout=300
        )
        return response.json() if response.status_code == 200 else None

# Usage
generator = PotteryGenerator(
    "https://youssefabdelmaksod-pottery-img-generator.hf.space",
    "hf_YOUR_TOKEN"
)

# Check if ready
if generator.health_check():
    result = generator.generate(
        "artisan ceramic bowl with natural glaze",
        steps=25,
        guidance_scale=8.0
    )
```

---

## Conclusion

The Pottery Image Generator provides a powerful API for creating beautiful pottery images. Use the examples and best practices in this guide to get the most out of the service.

Happy pottery generating! 🏺✨
