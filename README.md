# Cinematic Frame Generation with Diffusion Models

In this project, we evaluated various approaches to finetuning and other strategies to create more "cinematic" images from AI generated images. We will leave the term "cinematic" to be a subjective term, but generally this would translate to more dramatic lighting, grandiose landscapes, and lens flare effects. We evaluated the following strategies:
1. Base stable diffusion model
2. LoRA Finetuned model
3. Textual Inversion finetuned model to encode "cinematic style" token
4. Dreambooth finetuned model to specify cinematic style
5. ControlNet with prompting

We then evaluated two techniques to take an existing **video**, and make it more cinematic:
1. ControlNet style transfer per frame
2. Control-a-video paper, pretrained

An example of the outputs of the different models can be seen below:
![image](https://github.com/rmalde/236-cinematic-diffusion/assets/47470168/ab17037c-c3c7-43ed-a5fc-93dc1771e990)

