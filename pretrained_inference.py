from huggingface_hub import model_info

# LoRA weights ~3 MB
model_path = "outputs/models/checkpoint-1500"

model_base = "runwayml/stable-diffusion-v1-5"

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("Spiderman in space", num_inference_steps=25).images[0]
image.save("green_pokemon.png")