# LoRA fine-tuning starter script for pottery dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler, StableDiffusionTrainer, StableDiffusionTrainingArguments
from peft import LoraConfig, get_peft_model
import torch
import os

def train_lora(
    data_dir="data/pottery",
    base_model="runwayml/stable-diffusion-v1-5",
    output_dir="models/lora_pottery",
    num_train_epochs=1,
    batch_size=2,
    learning_rate=1e-4,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1
):
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    unet = pipe.unet
    # Prepare LoRA config
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    unet = get_peft_model(unet, lora_config)
    # Prepare dataset (simple image folder dataset)
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Training arguments
    training_args = StableDiffusionTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=50,
        logging_steps=10,
        remove_unused_columns=False,
        fp16=True
    )
    # Trainer
    trainer = StableDiffusionTrainer(
        model=unet,
        args=training_args,
        train_dataset=dataset,
        tokenizer=pipe.tokenizer,
        data_collator=None,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"LoRA fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    train_lora()
