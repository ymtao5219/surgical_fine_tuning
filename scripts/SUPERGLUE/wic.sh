#!/bin/bash
#SBTACH --job-name=superglue_wic # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs


module load conda 
conda activate petl # CHANGE THIS! if you have different environment names

echo "######################################################################"
echo "full model fine-tuning"
python code/fine_tuner.py --benchmark superglue --task_name wic


echo "######################################################################"
# top layers based on fisher information: [5, 1, 4, 3, 2]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 1 2 3 4 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 2 3 4 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 2 3 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 2 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 6 7 8 9 10 11

echo "######################################################################"
echo "finished"