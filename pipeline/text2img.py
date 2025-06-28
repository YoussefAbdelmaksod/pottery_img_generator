from diffusers import StableDiffusionPipeline
import torch
import random
import logging


def generate_pottery_image(
    prompt,
    output_path,
    model_name="runwayml/stable-diffusion-v1-5",
    style=None,
    material=None,
    perspective=None,
    guidance_scale=7.5,
    seed=None,
    negative_prompt=None
):
    """
    Generate a pottery image from a text prompt using Stable Diffusion.
    Allows control over style, material, and perspective.
    """
    logging.basicConfig(level=logging.INFO)
    # Build enhanced prompt
    pottery_prompt = prompt
    if style:
        pottery_prompt += f", {style} style"
    if material:
        pottery_prompt += f", {material}"
    if perspective:
        pottery_prompt += f", {perspective}"

    # Set seed for reproducibility
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        # Generate image with prompt and options
        image = pipe(pottery_prompt, guidance_scale=guidance_scale, generator=generator, negative_prompt=negative_prompt).images[0]
        image.save(output_path)
        logging.info(f"Image saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate image: {e}")

# Example usage (for testing)
if __name__ == "__main__":
    generate_pottery_image(
        prompt="A vessel",
        output_path="output.png",
        style="Japanese Raku",
        material="glazed ceramic",
        perspective="studio lighting, side view",
        model_name="runwayml/stable-diffusion-v1-5",
        guidance_scale=8.0,
        seed=42
    )
