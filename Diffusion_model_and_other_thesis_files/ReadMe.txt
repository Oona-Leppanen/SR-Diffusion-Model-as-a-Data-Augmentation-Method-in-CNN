Read Me

This project was a Master's thesis done for University of Turku. The project consists of two pipelines that are the training of three CNN classifiers with high-resolution, low-resolution and super-resolution data and a pipeline for evaluating the used diffusion model.

First, the acknowledgement regarding some of the code is presented followed by instructions of how to install and use the provided code.

-----------------------------------------------------------------------------
 
Acknowledgement

This folder includes some files that are modified versions of the respectively named files that can be found from Github page [1] titled as "SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution (CVPR2024)" by Rongyuan Wu et al. 2024.

These modified files can be found from the ram and Thesis folders. The folders contain another Read Me files to specify which files are from [1] and has been changed.

All changes have been made by Oona Leppänen.

The changes are made to be able to run the modified SeeSR (seesr_with_ccsr_vae) version created for her Master's thesis.

[1] https://github.com/cswry/SeeSR/tree/main

-----------------------------------------------------------------------------

Instructions for installing and using the code

The provided code can be utilized in multiple ways:
	1. using the same process done in my thesis
	2. using only data preprocessing
	3. using only the modified diffusion model
	4. using only the CNN classifier (training)
	5. using only the CNN classifier (classification)
	6. using the modified diffusion model and evaluating it

Each of these are discussed in the following sections from A to D.
	- For using case 1, read all sections (A-D) in the
		corresponding order.
	- For using case 2, read section A.
	- For using case 3, read section B.
	- For using case 4, read section C1 in C.
	- For using case 5, read section C2 in C.
	- For using case 6, read section D.

-----------------------------------------------------------------------------

Section A: Usage of the data preprocessing components

First, you have to find or create a high-resolution classification dataset. When that dataset is ready, follow the numbered steps starting from number one:

1. Change the input and output file paths in split_hr_data_train_valid_test file to match your paths. The input path should be the path to your high-resolution dataset and the output path should lead to any folder where you want to save your split high-resolution dataset. (Diffusion_model_and_other_thesis_files -> Thesis)

2.  Change the input and output file paths in rescale_images_to_lr file to match your paths. The input path should be the path to your split high-resolution dataset and the output path should lead to any folder where you want to save your low-resolution dataset. (Diffusion_model_and_other_thesis_files -> Thesis)

3. Run the "High-resolution data" and "Low-resolution data" sections in Preparing_high_low_and_cropped_datasets file, respectively, to achieve the split version and low-resolution version of your high-resolution dataset. (Meta files for running Python files folder)

-----------------------------------------------------------------------------

Section B: Usage of diffusion model

Because the used diffusion model is a modification of SeeSR [1] model by Rongyuan Wu et al., it is not offered in this project page entirely but the missing pieces can be found and downloaded from [1] (SeeSR) and [2] (VAE of the CCSR model). This modified model utilizes the VAE component used in the CCSR model. CCSR and its VAE can be found from Github page [2] called "CCSR Improving the Stability and Efficiency of Diffusion Models for Content Consistent Super-Resolution" by Lingchen Sun et al. 2024. Specifically, CCSR-v2 is used.

1. Download SeeSR as instructed in [1].
	- NOTICE! Use package list given in this repository. It goes by name
		diffusion_model_requirements. Using the list given in [1]
		leads to a missing package problems.
		(Diffusion_model_and_other_thesis_files folder)
	- To avoid problems, use the file_for_running_seesr_versions file for
		correct package and import loads and usage of the diffusion
		model. (Meta files for running Python files folder)

2. Read the Read Me file in ram folder. Replace the utils.py file of the SeeSR model as instructed in the Read Me in models folder in ram folder.

3. Add all files and Thesis folder in SeeSR_main folder (No ram folder!)

4. Download the checkpoints of VAE of the CCSR model from Quick Inference section in [2]. To download the checkpoints straight away, go to [3] which is a Google Drive folder for the checkpoints. Download all the files and place them in the CCSR_VAE_checkpoints folder. (Diffusion_model_and_other_thesis_files -> Thesis -> models)

5. Change input and output paths in the seesr_with_ccsr_vae file. The input path should be the path to your low-resolution dataset and the output path should lead to any folder where you want to save super-resolution dataset generated by the diffusion model. (Diffusion_model_and_other_thesis_files)

6. Run "Package Installations" and "SeeSR with CCSR VAE" sections of the file_for_running_seesr_versions file to generate the super-resolution dataset. (Meta files for running Python files folder)


[1] https://github.com/cswry/SeeSR/tree/main
[2] https://github.com/csslc/CCSR
[3] https://drive.google.com/drive/folders/1yHfMV81Md6db4StHTP5MC-eSeLFeBKm8

-----------------------------------------------------------------------------

Section C: Usage of the CNN classifier

---------

Section C1: Training CNN classifiers for a/all dataset(s)

1. Change input and output paths given in the cnn_classifier to your paths. Make sure all paths are changed! There are lots of paths in the code, so change all paths you need. (CNN_classifier)

2. Run "Imports" and "Set up" sections and all other sections you want. Notice that "McNemar's Test" section needs all three CNN classifiers to be trained.

---------

Section C2: Using pretrained classifiers

1. Change input and output paths given in the cnn_classifier file to your paths. Make sure all paths are changed! There are lots of paths in the code, so change all paths you need. (CNN_classifier)

2. Run "Imports", "Initializing Dataset Loading" in "Set up", the "Loading the Dataset" and "Prediction and Performance Metrics" in the wanted CNN model section such as in "Low-Resolution CNN Model". This way you receive predictions for the dataset(s) you inputted to the CNN model(s).

-----------------------------------------------------------------------------

Section D: Evaluation of the diffusion model

1. Set up the diffusion model as described in parts 1-4 in section B.

2. Change input and output paths in the crop_DIV2K_valid_data file. The input path should be the path to your high-resolution dataset and the output paths should lead to any folder(s) where you want to save two different sized low-resolution datasets. The name of the file includes the name of the test set dataset originally used in project.(Diffusion_model_and_other_thesis_files -> Thesis)

3. Run the "Cropped low-resolution DIV2K_valid dataset" section in Preparing_high_low_and_cropped_datasets file to achieve the two different sized low-resolution datasets based on your high-resolution dataset. (Meta files for running Python files folder)

4. Run diffusion model as described in parts 5-6 in section B. NOTICE:
	- You need to give the low-resolution dataset that has size 128x128
		and that has been created in the previous part (3) as input
		for the diffusion model.
	- You need to replace the paths in the
		seesr_with_ccsr_vae_cropped_data file instead of the
		seesr_with_ccsr_vae file and run "SeeSR with CCSR VAE Using
		Cropped Data" instead of "SeeSR with CCSR VAE" in the
		file_for_running_seesr_versions file.
		(Diffusion_model_and_other_thesis_files -> Thesis),
		(Meta files for running Python files folder)

5. Change input paths in the calculate_diffusion_metrics and calculate_niqe files. The input paths should be the paths to your the low-resolution dataset that has size 128x128 and the dataset generated by the diffusion model in previous part 4. The output paths should lead to any folder(s) where you want to save the evaluation results.(Diffusion_model_and_other_thesis_files -> Thesis)

-----------------------------------------------------------------------------

NOTICE!

1. The seesr_with_ccsr_vae_test_img file is the same as seesr_with_ccsr_vae expect the input and output file paths. The point of this file was to work as an extra piece from which I received some example image for my thesis.

2. The test_seesr_changed_for_thesis file is the same as the original test_seesr except it has been made to work with the needed changes for the SeeSR files and its given input and output paths has been changed. The point of this file was to work as an extra piece from which I received some example image for my thesis.