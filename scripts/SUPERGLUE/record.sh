#!/bin/bash
#SBTACH --job-name=superglue_record  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --time 24:00:00  # time


echo "######################################################################"
echo "full model fine-tuning"
python code/fine_tuner.py --benchmark superglue --task_name record


echo "######################################################################"
# top layers based on fisher information: [4,6,5,2,1,3,11,10,0,7,8,9]
echo "layer-wise fine-tuning top 1"
python code/test_code/fine_tuner.py --benchmark superglue --task_name record --freeze_layers 0 1 2 3 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --benchmark superglue --task_name record --freeze_layers 0 1 2 3 5 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --benchmark superglue --task_name record --freeze_layers 0 1 2 3 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --benchmark superglue --task_name record --freeze_layers 0 1 3 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --benchmark superglue --task_name record --freeze_layers 0 3 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning bottom 1"
python code/fine_tuner.py --benchmark superglue --task_name record --freeze_layers 0 1 2 3 4 5 6 7 8 10 11
echo "######################################################################"
echo "finished"