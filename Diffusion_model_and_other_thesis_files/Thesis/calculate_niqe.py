'''
Oona Leppänen
Created 27.10.2025
'''

import glob
import argparse
import numpy as np
from PIL import Image
import statistics
import torch
from metrics.image_quality import niqe

	
def main(args):
	niqe_score_list = []
	pred_image_names = glob.glob(f'{args.diffusion_images_path}/*.*')
	
	if torch.cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
		
	# There are 3000 ground truth images and 3000 predicted images from the diffusion model
	# (SeeSR with CCSR VAE).
	for i in range(3000):
		print('Image count: ', i+1)
		
		predicted_image = Image.open(pred_image_names[i])
		predicted_image = np.array(predicted_image)

		niqe_score = niqe(predicted_image)
		niqe_score_list.append(niqe_score)
		print('NIQE:', niqe_score, '\n')

		# Save the calculated metric.
		txt_save_path = f"{args.output_dir}/niqe.txt"
		file = open(txt_save_path, 'w')
		for score in niqe_score_list:
			file.write(f"{score}\n")
		file.close()
					
	# Calculates the means of the NIQE score.
	mean_niqe = statistics.mean(niqe_score_list)
	
	
	# Saving results of all metrics.
	results_path = f"{args.output_dir}/niqe_metric_results_with_CCSR_VAE.txt"
	file = open(results_path, 'w')
	file.write(f"NIQE: {mean_niqe}")
	file.close()
								
	print(f'NIQE mean: {mean_niqe}')
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--diffusion_images_path", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/metric_calculations/cropped_diffusion_DIV2K_valid_dataset')
	parser.add_argument("--output_dir", type=str, default='drive/MyDrive/Yliopisto/Diplomityö/Datasets/metric_calculations/metric_results')
	args = parser.parse_args()
	main(args)