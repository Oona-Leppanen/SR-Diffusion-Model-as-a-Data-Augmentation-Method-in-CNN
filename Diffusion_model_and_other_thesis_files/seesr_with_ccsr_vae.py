'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''

'''
This file is based on 
SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution
by Rongyuan Wu, et al.
https://github.com/cswry/SeeSR/tree/main

Oona Leppänen
Created 25.4.2025
Last modified 26.11.2025
'''

import os
import sys
sys.path.append(os.getcwd())

import cv2
import glob
import argparse
import numpy as np
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from typing import Mapping, Any

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


tensor_transforms = transforms.Compose([transforms.ToTensor(),])
ram_transforms = transforms.Compose([
			transforms.Resize((384, 384)),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])

def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
	from Thesis.models.controlnet import ControlNetModel
	from Thesis.models.unet_2d_condition import UNet2DConditionModel
	
	text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
	tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
	feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
	unet = UNet2DConditionModel.from_pretrained(args.seesr_model_path, subfolder="unet")
	controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
	scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
	vae = AutoencoderKL.from_pretrained(args.CCSR_VAE_model_path)
	
	for model in [text_encoder, unet, controlnet, vae]:
		model.requires_grad_(False)
		
	if enable_xformers_memory_efficient_attention:
		if is_xformers_available():
			unet.enable_xformers_memory_efficient_attention()
			controlnet.enable_xformers_memory_efficient_attention()
		else:
			raise ValueError('xformers is not available. Make sure it is installed correctly.')
	
	validation_pipeline = StableDiffusionControlNetPipeline(
					vae = vae,
					text_encoder = text_encoder,
					tokenizer = tokenizer,
					feature_extractor = feature_extractor,
					unet = unet,
					controlnet = controlnet,
					scheduler = scheduler,
					safety_checker = None,
					requires_safety_checker = False,
					)
			
	validation_pipeline._init_tiled_vae(encoder_tile_size = args.vae_encoder_tiled_size, decoder_tile_size = args.vae_decoder_tiled_size)
	
	weight_dtype = torch.float32
	if accelerator.mixed_precision == "fp16":
		weight_dtype = torch.float16
	elif accelerator.mixed_precision == "bf16":
		weight_dtype = torch.btorch16
		
	for model in [text_encoder, unet, controlnet, vae]:
		model.to(accelerator.device, dtype = weight_dtype)
	
	return validation_pipeline
	
def load_tag_model(args, device = 'cuda'):
	ram_file_path = 'drive/MyDrive/Yliopisto/Diplomityö/Code/SeeSR_Git_code/SeeSR_main/preset/models/ram_swin_large_14m.pth'
	model = ram(
		pretrained = ram_file_path,
		pretrained_condition = args.dape_checkpoints_path,
		image_size = 384,
		vit = 'swin_l'
		)
		
	model.eval()
	model.to(device)
	
	return model
	
def get_validation_prompt(args, valid_image, tag_model, device = 'cuda'):
	validation_prompt = ''
	
	image_tensor = tensor_transforms(valid_image).unsqueeze(0).to(device)
	image_tensor = ram_transforms(image_tensor)
	resulting_prompts = inference(image_tensor, tag_model)
	
	ram_encoder_hidden_states = tag_model.generate_image_embeds(image_tensor)
	validation_prompt = f"{resulting_prompts[0]}, {args.prompt},"
	
	return validation_prompt, ram_encoder_hidden_states
	
	
def main(args, enable_xformers_memory_efficient_attention = True,):
	image_count = 0
	
	accelerator = Accelerator(mixed_precision = args.mixed_precision,)
	
	if args.seed is not None:
		set_seed(args.seed)
		
	if accelerator.is_main_process:
		os.makedirs(args.output_dir, exist_ok = True)
		accelerator.init_trackers('SeeSR')
		
	pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
	model = load_tag_model(args, accelerator.device)
	
	if accelerator.is_main_process:
		generator = torch.Generator(device = accelerator.device)
		if args.seed is not None:
			generator.manual_seed(args.seed)
			
		for folder in ['train', 'validation', 'test']:
				for subfolder in ['human', 'non-human']:
					txt_path = os.path.join(os.path.join(args.output_dir, folder), f'{subfolder}_txt')
					os.makedirs(txt_path, exist_ok = True)
					
					if os.path.isdir(args.image_path):
						image_names = sorted(glob.glob(f'{args.image_path}/{folder}/{subfolder}/*.*'))
					else:
						image_names = [args.image_path]
					
					for image_idx, image_name in enumerate(image_names[:]):
						print(f'================== process {image_idx} imgs... ===================')
						validation_image = Image.open(image_name).convert('RGB')
						
						validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)
						validation_prompt += args.added_prompt # Possible prompts: clean, extremely detailed, best quality, sharp, clean
						negative_prompt = args.negative_prompt # Possible prompts: dirty, messy, low quality, frames, deformed
						
						if args.save_prompts:
							txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
							file = open(txt_save_path, 'w')
							file.write(validation_prompt)
							file.close()
							print('Tags have been saved')
						else:
							print('No tags saved!')
						print(f'{validation_prompt}')
						
						original_width, original_height = validation_image.size
						resize_flag = False
						rscale = args.upscale

						# If original_width or original_height is smaller than 128 pixels, the image is scaled larger so
						# that the smaller dimension is scaled to 128 pixels.
						if original_width < args.process_size//rscale or original_height < args.process_size//rscale:
							scale = (args.process_size//rscale)/min(original_width, original_height)
							validation_image = validation_image.resize((int(scale*original_width), int(scale*original_height)))
							resize_flag = True
							
						validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale)) # Upscaling the image.
						validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
						width, height = validation_image.size
						resize_flag = True
						print(f'intput size: {height}x{width}')

						for sample_idx in range(args.sample_times):
							#os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok = True)
							
							with torch.autocast('cuda'):
								image = pipeline(
										validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator,
										height=height, width=width, guidance_scale=args.guidance_scale, negative_prompt=negative_prompt,
										conditioning_scale=args.conditioning_scale, start_point=args.start_point,
										ram_encoder_hidden_states=ram_encoder_hidden_states, latent_tiled_size=args.latent_tiled_size,
										latent_tiled_overlap=args.latent_tiled_overlap, args=args,
										).images[0]
							
							if args.align_method == 'nofix':
								image = image
							else:
								if args.align_method == 'wavelet':
									image = wavelet_color_fix(image, validation_image)
								elif args.align_method == 'adain':
									image = adain_color_fix(image, validation_image)
									
							if resize_flag:
								image = image.resize((original_width*rscale, original_height*rscale))
								
							name, ext = os.path.splitext(os.path.basename(image_name))
							image.save(f'{args.output_dir}/{folder}/{subfolder}/{name}.png')
							image_count += 1
							
							print('Image count: ', image_count)
							
# Change the paths to correspond your paths
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--seesr_model_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Code/SeeSR_Git_code/SeeSR_main/preset/models/SeeSR_checkpoints/seesr')
	parser.add_argument("--dape_checkpoints_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Code/SeeSR_Git_code/SeeSR_main/preset/models/SeeSR_checkpoints/DAPE.pth')
	parser.add_argument("--pretrained_model_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Code/SeeSR_Git_code/SeeSR_main/preset/models/stable-diffusion-2-base')
	parser.add_argument("--CCSR_VAE_model_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Code/SeeSR_Git_code/SeeSR_main/Thesis/models/CCSR_VAE_checkpoints')
	parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
	parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
	parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
	parser.add_argument("--image_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/classification_datasets/lr_binary_dataset')
	parser.add_argument("--output_dir", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/classification_datasets/sr_binary_dataset')
	parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
	parser.add_argument("--guidance_scale", type=float, default=5.5)
	parser.add_argument("--conditioning_scale", type=float, default=1.0)
	parser.add_argument("--blending_alpha", type=float, default=1.0)
	parser.add_argument("--num_inference_steps", type=int, default=50)
	parser.add_argument("--process_size", type=int, default=512)
	parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
	parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
	parser.add_argument("--latent_tiled_size", type=int, default=96)
	parser.add_argument("--latent_tiled_overlap", type=int, default=32)
	parser.add_argument("--upscale", type=int, default=4)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--sample_times", type=int, default=1)
	parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
	parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
	parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
	parser.add_argument("--save_prompts", action='store_false') # Change to store_true if don't want to save tags.
	args = parser.parse_args()
	main(args)