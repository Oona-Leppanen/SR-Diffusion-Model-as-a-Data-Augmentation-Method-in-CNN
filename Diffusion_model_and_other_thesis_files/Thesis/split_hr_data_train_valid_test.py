'''
Oona Leppänen
Created 12.5.2025
'''

import glob
import shutil
import torch
import argparse
	
# Create new dataset by copying train, valid and test images to different folder called classified_hr_dataset.
def save_images(args, image_files, folder, subfolder):
	for image_file in image_files:
		output_path = f'{args.output_file_path}/{folder}/{subfolder}'
		shutil.copy(image_file, output_path) # Copy file.
	
def main(args):
	# Goes through all images belonging all classes in the data one class at a time.
	for subfolder in ['human', 'non-human']:
		image_files = glob.glob(f'{args.image_file_path}/{subfolder}/*.*')
		
		# Train, validation, test set split
		train_set, valid_set, test_set = torch.utils.data.random_split(image_files, [0.8, 0.10, 0.10])

		print(subfolder)
		print('train set length: ', len(train_set))
		print('validation set length: ', len(valid_set))
		print('test set length: ', len(test_set))
		print('\n')
		
		save_images(args, train_set, 'train', subfolder)
		save_images(args, valid_set, 'validation', subfolder)
		save_images(args, test_set, 'test', subfolder)
		
	print('All images saved.')
		
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_file_path', type = str ,default = 'drive/MyDrive/Yliopisto/Diplomityö/Datasets/classification_datasets/classified_hr_dataset_no_split/train')
	parser.add_argument('--output_file_path', type = str, default = 'drive/MyDrive/Yliopisto/Diplomityö/Datasets/classification_datasets/classified_hr_dataset')
	args = parser.parse_args()
	main(args)