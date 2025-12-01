'''
Oona Leppänen
Created 23.5.2025
'''

import os
import glob
from PIL import Image
import argparse
from torchvision import transforms

def main(args):
	image_names = glob.glob(args.image_paths)
	
	# Creates 3000 images out of 100 images.
	for i in range(30):
		for image_name in image_names:
			image = Image.open(image_name).convert('RGB')
			name, file_type = os.path.splitext(os.path.basename(image_name))
			
			# Crops the image randomly. Target size is 512 as in the SeeSR research paper.
			crop = transforms.RandomCrop(512)
			cropped_image = crop(image)
			
			cropped_image.save(f'{args.output_dir_512}/{name}_{i+1}.png')# Saves the image.
			
			# Target image size is 128x128 as in the SeeSR research paper so 
			# the images are divided by number 4 to reach the target size.
			cropped_image = cropped_image.resize((cropped_image.size[0]//4, cropped_image.size[1]//4))
			width, height = cropped_image.size
			
			if width != 128 or height != 128:
				print('Image name: ', name)
				print(f'size: {width}x{height}')
			else:
				cropped_image.save(f'{args.output_dir_128}/{name}_{i+1}.png')# Saves the image.
			
		print(f'Round {i} done')
	print('All cropped images saved.')
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_paths", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/classification_datasets/classified_DIV2K_valid_dataset/*/*.*')
	parser.add_argument("--output_dir_512", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/metric_calculations/512_cropped_DIV2K_valid_dataset')
	parser.add_argument("--output_dir_128", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/metric_calculations/128_cropped_DIV2K_valid_dataset')
	args = parser.parse_args()
	main(args)