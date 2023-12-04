from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

from tqdm import tqdm
import os
import imageio

def count_frames_manual(video):
	# initialize the total number of frames read
	total = 0
	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()
	 
		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break
		# increment the total number of frames read
		total += 1
	# return the total number of frames in the video file
	return total

def count_frames(path, override=False):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0
	# if the override flag is passed in, revert to the manual
	# method of counting frames
	if override:
		total = count_frames_manual(video)
    
	# otherwise, let's try the fast way first
	else:
		try:
			# check if we are using OpenCV 3
			if is_cv3():
				total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
			# otherwise, we are using OpenCV 2.4
			else:
				total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		# uh-oh, we got an error -- revert to counting manually
		except:
			total = count_frames_manual(video)
	# release the video file pointer
	video.release()
	# return the total number of frames in the video
	return total

try:
    os.remove("output.mp4")
except OSError:
    pass

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

vid_path = "input_short.mp4"
cap = cv2.VideoCapture(vid_path)

total_num_frames = count_frames(vid_path, override=False)

pbar = tqdm(total=total_num_frames)

# writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 360))
writer = imageio.get_writer('output.mp4', fps=24)
num_frames = 1000
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret or i > num_frames:
        break
    
    # Display the frame
    image = np.array(frame)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image], axis=2)
    image = image.reshape((image.shape[0], image.shape[1]))

    canny_image = Image.fromarray(image)
    # canny_image = Image.fromarray(image)

    output = pipe(
        "super epic crazy insane cinmatic", image=canny_image
    ).images[0]

    pil_numpy = np.array(output)
    pil_numpy=cv2.resize(pil_numpy,(640,360))
    # pil_numpy = cv2.cvtColor(pil_numpy, cv2.COLOR_RGB2BGR)

    # print(pil_numpy.shape)

    # writer.write(pil_numpy)
    writer.append_data(pil_numpy)

    pbar.update(1)
    i += 1

pbar.close()
cap.release()
# writer.release()
writer.close()