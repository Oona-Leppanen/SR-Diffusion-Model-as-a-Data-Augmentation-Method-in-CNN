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
Created copy 5.5.2025
Last modified 26.11.2025
'''

import os
import glob
from PIL import Image
import argparse

# Function for opening high resolution images, resizing them and saving the low resolution variants.
def resize_and_save_images(args, folder, subfolder):
	if os.path.isdir(args.image_path):
		image_names = sorted(glob.glob(f'{args.image_path}/{folder}/{subfolder}/*.*'))
	else:
		image_names = [args.image_path]
		
	for image_idx, image_name in enumerate(image_names[:]):
		image = Image.open(image_name).convert('RGB')
		original_width, original_height = image.size
		
		# Resizes the image. Number 6 is used because the smallest side of an image in the dataset is 816 
		# and the largest is 2024. Therefore the image sizes are close to the size used in SeeSR paper
		# if the images are divided by number 6. (816/6=136, 2040/6=340)
		image = image.resize((image.size[0]//6, image.size[1]//6))
		width, height = image.size
		print(f'intput size: {width}x{height}')
		
		# Saves the image.
		name, file_type = os.path.splitext(os.path.basename(image_name))
		image.save(f'{args.output_dir}/{folder}/{subfolder}/{name}.png')
		
def main(args):
	for folder in ['train', 'validation', 'test']:
		for subfolder in ['human', 'non-human']:
			resize_and_save_images(args, folder, subfolder)# Opens, resizes and saves the high resolution images.
			print(f'\n{folder}/{subfolder} ready!')
		print(folder, 'ready!')

	print('All low resolution images saved.')
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/classification_datasets/classified_hr_dataset')
	parser.add_argument("--output_dir", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/classification_datasets/classified_lr_dataset')
	args = parser.parse_args()
	main(args)