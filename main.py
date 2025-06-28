import argparse
from pipeline.text2img import generate_pottery_image

def main():
    parser = argparse.ArgumentParser(description="Generate pottery images using Stable Diffusion.")
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for pottery image')
    parser.add_argument('--output', type=str, default='output.png', help='Output image file')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5', help='Model name or path')
    parser.add_argument('--style', type=str, default=None, help='Pottery style (e.g., Japanese Raku, Greek)')
    parser.add_argument('--material', type=str, default=None, help='Material/texture (e.g., glazed ceramic, earthenware)')
    parser.add_argument('--perspective', type=str, default=None, help='Camera perspective (e.g., side view, top-down)')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for diffusion')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Negative prompt for image generation (what to avoid)')
    args = parser.parse_args()

    generate_pottery_image(
        prompt=args.prompt,
        output_path=args.output,
        model_name=args.model,
        style=args.style,
        material=args.material,
        perspective=args.perspective,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt
    )

if __name__ == "__main__":
    main()
