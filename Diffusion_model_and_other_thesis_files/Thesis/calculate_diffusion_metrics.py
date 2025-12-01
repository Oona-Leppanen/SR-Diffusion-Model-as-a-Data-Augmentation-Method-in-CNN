'''
Oona Leppänen
Created 26.5.2025
'''

import glob
import argparse
import numpy as np
from PIL import Image
from scipy import linalg
import statistics

import torch
import torchvision
from torchvision import transforms
from torcheval.metrics import FrechetInceptionDistance

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# Calculates FID
def get_fid(args, original_image_names, predicted_images_names, device = 'cuda'):	
	fid_score_list = []
	
	transform_to_tensor = transforms.Compose([transforms.ToTensor(),])
	resize_transform = transforms.Resize(299) # Inception v3 takes inputs with size (299, 299, 3).
	
	fid_model = FrechetInceptionDistance()
	fid_model.to(device)
	
	# Divides data into 4 batches and updates the data into FID.
	for i in range(750, 3750, 750):
		og_images = []
		pred_images = []
	
		# Ground truth images
		for image in original_image_names[i-750:i]:
			original_image = Image.open(image).convert('RGB')
			original_image = resize_transform(original_image)
			original_image_tensor = transform_to_tensor(original_image) # Transforms image to tensor.
			original_image_tensor = original_image_tensor.to(device)
			og_images.append(original_image_tensor)
		
		og_images_tensor = torch.stack(og_images, dim = 0)
		og_images_tensor.to(device)
		fid_model.update(og_images_tensor, is_real = True) # Updates the FID
		og_images = [] # Empty the list to save memory.
		og_images_tensor = 0
		
		
		# Distorted images
		for image in predicted_images_names[i-750:i]:
			predicted_image = Image.open(image).convert('RGB')
			predicted_image = resize_transform(predicted_image)
			predicted_image_tensor = transform_to_tensor(predicted_image) # Transforms image to tensor.
			predicted_image_tensor = predicted_image_tensor.to(device)
			pred_images.append(predicted_image_tensor)
		
		pred_images_tensor = torch.stack(pred_images, dim = 0)
		pred_images_tensor.to(device)		
		fid_model.update(pred_images_tensor, is_real = False) # Updates the FID
		pred_images = [] # Empty the list to save memory.
		pred_images_tensor = 0
		
		# Compute the FID score with one batch and add it to a list of batches.
		batch_fid_score = fid_model.compute()
		fid_score_list.append(batch_fid_score.item())
		
	# Take mean of the FID scores of the batches.
	mean_fid = statistics.mean(fid_score_list)
	
	# Compute the FID score with all data
	fid_score = fid_model.compute()
	
	return fid_score.item(), mean_fid
	
	
def main(args):
	ssim_score_list = []
	psnr_score_list = []
	
	og_image_names = glob.glob(f'{args.original_images_path}/*.*')
	pred_image_names = glob.glob(f'{args.diffusion_images_path}/*.*')
	
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	
	# FID
	fid_score, mean_fid = get_fid(args, og_image_names, pred_image_names, device)
	print(f'FID: {fid_score}\nFID mean: {mean_fid}')
	
	
	# SSIM & PSNR
		
	# There are 3000 ground truth images and 3000 predicted images from the diffusion model
	# (SeeSR with CCSR VAE).
	for i in range(3000):
		print('Image count: ', i+1)
		original_image = Image.open(og_image_names[i]).convert('YCbCr')
		predicted_image = Image.open(pred_image_names[i]).convert('YCbCr')
		
		original_image = np.array(original_image)
		predicted_image = np.array(predicted_image)
		
		# SSIM
		ssim_score = ssim(original_image, predicted_image, data_range = 255, multichannel = True)
		ssim_score_list.append(ssim_score)
		print('SSIM:', ssim_score)

		# PSNR
		psnr_score = psnr(original_image, predicted_image, data_range = 255)
		psnr_score_list.append(psnr_score)
		print('PSNR:', psnr_score, '\n')

		# Save the calculated metrics.
		for metric in ['SSIM_list', 'PSNR_list']:
			txt_save_path = f"{args.output_dir}/{metric}.txt"
			file = open(txt_save_path, 'w')
			if metric == 'SSIM_list':
				for score in ssim_score_list:
					file.write(f"{score}\n")
			elif metric == 'PSNR_list':
				for score in psnr_score_list:
					file.write(f"{score}\n")
			file.close()
					
	# Calculates the means of the SSIM and PSNR scores.
	mean_ssim = statistics.mean(ssim_score_list)
	mean_psnr = statistics.mean(psnr_score_list)
	
	
	# Saving results of all metrics.
	results_path = f"{args.output_dir}/metric_results_with_CCSR_VAE.txt"
	file = open(results_path, 'w')
	file.write(f"SSIM: {mean_ssim}\nPSNR: {mean_psnr}\nFID with all data: {fid_score}\nFID mean of batches: {mean_fid}")
	file.close()
								
	print(f'SSIM mean: {mean_ssim}\nPSNR mean: {mean_psnr}\nFID:  {fid_score}\nFID mean: {mean_fid}')
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--original_images_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/metric_calculations/512_cropped_DIV2K_valid_dataset')
	parser.add_argument("--diffusion_images_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/metric_calculations/cropped_diffusion_DIV2K_valid_dataset')
	parser.add_argument("--output_dir", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/metric_calculations/metric_results')
	args = parser.parse_args()
	main(args)