#!/bin/bash

## Job file for running on the SoC Compute Cluster

#SBATCH --job-name=capstone_kevin
#SBATCH --partition=long
#SBATCH --gres=gpu:a100
#SBATCH --nodes=1

## You can change any of the settings below
## Time limit for the job, but change this as necessary.
#SBATCH --time=3-00:00:00
## Just useful logfile names
#SBATCH --output=kevin_%j.slurmlog
#SBATCH --error=kevin_%j.slurmlog

# Run the common job file and pass all arguments to it
pip3 install argparse torch torchvision einops torchsummary pandas numpy tqdm
# python3 model.py --num_classes 100 --npz_file_path ../../dataset/dataset_100_200.npz
# python3 model.py --num_classes 200 --npz_file_path ../../dataset/dataset_200_200.npz
# python3 model.py --num_classes 300 --npz_file_path ../../dataset/dataset_300_200.npz
# python3 model.py --num_classes 400 --npz_file_path ../../dataset/dataset_400_200.npz
# python3 model.py --num_classes 499 --npz_file_path ../../dataset/dataset_500_200.npz
# python3 model.py --num_classes 499 --npz_file_path ../../dataset/dataset_500_clem.npz
# python3 model.py --num_classes 499 --npz_file_path ../../dataset/dataset_500_mass.npz
# python3 model.py --num_classes 499 --npz_file_path ../../dataset/dataset_500_utah.npz
# python3 model.py --num_classes 499 --npz_file_path ../../dataset/dataset_500_wisc.npz
python3 model.py --num_classes 200 --npz_file_path ../../dataset/dataset_200_clem.npz
python3 model.py --num_classes 200 --npz_file_path ../../dataset/dataset_200_mass.npz
python3 model.py --num_classes 200 --npz_file_path ../../dataset/dataset_200_utah.npz
python3 model.py --num_classes 200 --npz_file_path ../../dataset/dataset_200_wisc.npz


# open world with threshold - doq
python3 model_threshold.py --num_classes 100 --word_size 3 --npz_file_path doq_100_360_47500_4_ow.npz --open_world
