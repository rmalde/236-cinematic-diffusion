from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
import gtts
from playsound import playsound

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
images = pipeline("man staring deeply into your soul while munching on a burrito, heavy shadows, detailed, (vibrant, photo realistic, realistic, dramatic, dark, sharp focus, 8k), (intricate:1.1), (highly detailed:1.1), digital painting, octane render, artstation, concept art, smooth, sharp focus, illustration, art by artgerm, (loish:0.23), wlop ilya kuvshinov., (global illumination, studio light, volumetric light)4k. ").images

for i in range(len(images)):
    plt.imshow(images[i])
    plt.savefig("image" + str(i) + ".png")
# generate speech
speech = gtts.gTTS("Ronak can you open the image file please. thanks")

# Save the speech to a file
speech.save("output.mp3")

# Play the speech
playsound("output.mp3")
