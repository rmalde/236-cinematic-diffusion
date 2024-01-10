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

## Dataset
Since there is no specific dataset that has cinematic photos/videos and also corresponding captions, we constructed our own dataset of around 50 cinematic images and associated captions. The dataset contained a combination of images found on the internet and images generated by DALLE-3, and consisted of shots of people, objects, and landscapes, with features such as dramatic backdrops and rays of light, creating a cinematic effect. We also manually created captions for the images. The cinematic dataset is in the `images` directory. Some examples of images can be found below:
![image](https://github.com/rmalde/236-cinematic-diffusion/assets/47470168/b200e23e-3f79-4b45-acb3-b28a5c2079d9)


## Running

### Setup
Setup a conda environment (we used Python 3.9) and run
```
pip install -r requirements.txt
```
And then run the following command to setup accelerate
```
accelerate config
```

### Finetuning
=
All finetuning scripts can be found in the `finetune` directory. Simply run any of the `.sh` scripts in order to finetune, for example:

```
cd finetune
./textual_inversion.sh
```

If you wish to create your own finetune dataset, feel free to copy the format found in the `images` directory, and update the training script to point to your dataset. 

### Evaluation

All evalution scripts can be found in the `inference` directory. Run any of these python scripts to generate a grid of images that compares that model to the base Stable Diffusion. For example:
```
cd inference
python inference_dreambooth.py
```
