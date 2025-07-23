import gradio as gr
from pipeline.text2img import generate_image
import tempfile

def pottery_demo(prompt, style, material, perspective, guidance_scale, seed, negative_prompt):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Only pass supported arguments to generate_image
        # lora_weights_dir and lora_scale are required by generate_image, others are not
        lora_weights_dir = "lora-output"  # or your actual LoRA weights directory
        steps = int(guidance_scale) if guidance_scale else 30
        guidance = guidance_scale if guidance_scale else 7.5
        generate_image(
            prompt=prompt,
            lora_weights_dir=lora_weights_dir,
            output_path=tmp.name,
            steps=steps,
            guidance=guidance,
            lora_scale=1.0
        )
        return tmp.name

demo = gr.Interface(
    fn=pottery_demo,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Style"),
        gr.Textbox(label="Material"),
        gr.Textbox(label="Perspective"),
        gr.Slider(5, 15, value=7.5, label="Guidance Scale"),
        gr.Number(label="Seed", value=None),
        gr.Textbox(label="Negative Prompt")
    ],
    outputs=gr.Image(type="filepath"),
    title="Pottery Image Generator",
    description="Generate pottery images using Stable Diffusion and custom prompts."
)

if __name__ == "__main__":
    demo.launch()
