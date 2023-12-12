from huggingface_hub import model_info
from tqdm.rich import tqdm
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import matplotlib.pyplot as plt
from transformers import pipeline

# LoRA weights ~3 MB
base_model_path = "runwayml/stable-diffusion-v1-5"
finetuned_model_path = "outputs/models_epic/checkpoint-8000"

prompts = [
    "Cheetah",
    "Guitar player at a concert",
    # "Jungle temple",
]

pipeline_base = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)
pipeline_base.to("cuda")

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
pipeline_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipeline_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipeline_controlnet.scheduler.config)
pipeline_controlnet.enable_model_cpu_offload()

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Create a subplot for each pair of images and prompts
fig, axs = plt.subplots(len(prompts), 3, figsize=(10, 2 * len(prompts)+3))

axs[0, 0].set_title("Prompt")
axs[0, 1].set_title("Base Model")
axs[0, 2].set_title("Controlnet Model")

# Iterate through prompts and generate images for each model
for i, prompt in enumerate(prompts):
    row_index = i

    axs[row_index, 0].text(0.5,0.5, prompt, ha='center', va='center', fontsize=16)
    axs[row_index, 0].axis('off')
    # Generate images for model_a and model_b
    image_a = pipeline_base(prompt).images[0]
    image_b = pipeline_controlnet("dramatic lighting of " + prompt, image=image_a).images[0]
    # load image from file
    # image_a = plt.imread('green_pokemon.png')
    # image_b = plt.imread('green_pokemon.png')

    # Display images and prompt for model_a
    axs[row_index, 1].imshow(image_a)
    axs[row_index, 1].axis('off')

    # Display images and prompt for model_b
    axs[row_index, 2].imshow(image_b)
    axs[row_index, 2].axis('off')

    print("------------")
    print(f"Prompt {i}")
    print("Original Prompt: ", prompt)
    print("A: ", image_to_text(image_a))
    print("B: ", image_to_text(image_b))

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig('output_plot_controlnet.png')
plt.show()