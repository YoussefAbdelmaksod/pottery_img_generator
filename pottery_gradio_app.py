import gradio as gr
from pipeline.text2img import generate_pottery_image
import tempfile

def pottery_demo(prompt, style, material, perspective, guidance_scale, seed, negative_prompt):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        generate_pottery_image(
            prompt=prompt,
            output_path=tmp.name,
            style=style,
            material=material,
            perspective=perspective,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt
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
