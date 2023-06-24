#!/bin/bash
#SBTACH --job-name=glue_stsb_bottom  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --time=48:00:00


# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --task_name stsb


# echo "######################################################################"
# # top layers based on fisher information: [ 2,1, 6, 4, 3] 2,1, 6, 4, 3, 5, 0, 7, 11, 8, 10, 9
# echo "layer-wise fine-tuning top 1"
# python code/fine_tuner.py --task_name stsb --freeze_layers 0 1 3 4 5 6 7 8 9 10 11 
# echo "######################################################################"
# echo "layer-wise fine-tuning top 2"
# python code/fine_tuner.py --task_name stsb --freeze_layers 0 3 4 5 6 7 8 9 10 11 
# echo "######################################################################"
# echo "layer-wise fine-tuning top 3"
# python code/fine_tuner.py --task_name stsb --freeze_layers 0 3 4 5 7 8 9 10 11 
# echo "######################################################################"
# echo "layer-wise fine-tuning top 4"
# python code/fine_tuner.py --task_name stsb --freeze_layers 0 3 5 7 8 9 10 11
# echo "######################################################################"
# echo "layer-wise fine-tuning top 5"
# python code/fine_tuner.py --task_name stsb --freeze_layers 0 5 7 8 9 10 11

# echo "######################################################################"
# echo "finished"

echo "######################################################################"
echo "layer-wise fine-tuning bottom 1" #2,1, 6, 4, 3, 5, 0, 7, 11, 8, 10, 9
python code/fine_tuner.py --task_name stsb --freeze_layers 0 1 2 3 4 5 6 7 8 10 11

echo "######################################################################"
echo "layer-wise fine-tuning bottom 2"
python code/fine_tuner.py --task_name stsb --freeze_layers 0 1 2 3 4 5 6 7 8 11

echo "######################################################################"
echo "layer-wise fine-tuning bottom 5"
python code/fine_tuner.py --task_name stsb --freeze_layers 0 1 2 3 4 5 6