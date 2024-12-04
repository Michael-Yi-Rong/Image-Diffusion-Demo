from pandas import read_json
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

ALLOW_CUDA = True ###
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

model_file = r"/SSD_DISK/users/rongyi/projects/diffusion/ckpt/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE
dense_caption = read_json(r"/SSD_DISK/users/rongyi/projects/diffusion/get_dense_caption/dense_caption.json")
video_key = "n008-2018-08-01-15-16-36-0400__CAM_FRONT__"
prompt = dense_caption[video_key]["description"]

"""
period_index = prompt.find(".")
# 截断到第一个句号的位置，如果找到了句号
if period_index != -1:
    prompt = prompt[:period_index + 1]  # 包含句号
else:
    prompt = prompt
"""

print("\n prompt: \n", prompt, "\n")
# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
# prompt = "The environment suggests an overcast day."
uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
image_path = r"/SSD_DISK/datasets/nuscenes_mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.8

## SAMPLER

sampler = "ddpm"
num_inference_steps = 800

seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cuda",
    # tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
img=Image.fromarray(output_image)
img.save(r"/SSD_DISK/users/rongyi/projects/diffusion/result_images/img.png")