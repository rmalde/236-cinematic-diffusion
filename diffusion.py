from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt

pipeline = DiffusionPipeline.from_pretrained("outputs/models/checkpoint-1500", torch_dtype=torch.float16)
pipeline.to("cuda")
image = pipeline("Super epic HD image of a mountain and sunset").images[0]

plt.imshow(image)
plt.show()