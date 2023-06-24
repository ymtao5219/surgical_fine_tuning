#!/bin/bash
#SBTACH --job-name=sparsity_glue_multirc  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/Sparsity/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --time 24:00:00  # time
#SBATCH -G 1  # Number of GPUs

# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --benchmark superglue --task_name multirc


echo "######################################################################"
# top layers based on fisher information: [11, 8, 7, 6, 5, 4, 3, 10, 9, 2, 1, 0]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --benchmark superglue --task_name multirc --freeze_layers 0 1 2 3 4 5 6 7 8 9 10
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --benchmark superglue --task_name multirc --freeze_layers 0 1 2 3 4 5 6 7 9 10
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --benchmark superglue --task_name multirc --freeze_layers 0 1 2 3 4 5 6 9 10
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --benchmark superglue --task_name multirc --freeze_layers 0 1 2 3 4 5 9 10
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --benchmark superglue --task_name multirc --freeze_layers 0 1 2 3 4 9 10

echo "######################################################################"
echo "finished"