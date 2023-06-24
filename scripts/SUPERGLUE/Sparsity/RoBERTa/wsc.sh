#!/bin/bash
#SBTACH --job-name=sparsity_superglue_wsc  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/Sparsity/RoBERTa/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --time 3-00:00:00  # time
#SBATCH -G 1  # Number of GPUs

# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --parent_model roberta-base --benchmark superglue --task_name wsc

echo "######################################################################"
# top layers based on sparsity: [7, 10, 8, 9, 2, 3, 6, 4, 5, 1, 0, 11]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --parent_model roberta-base --benchmark superglue --task_name wsc --freeze_layers 0 1 2 3 4 5 6 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --parent_model roberta-base --benchmark superglue --task_name wsc --freeze_layers 0 1 2 3 4 5 6 8 9 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --parent_model roberta-base --benchmark superglue --task_name wsc --freeze_layers 0 1 2 3 4 5 6 9 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --parent_model roberta-base --benchmark superglue --task_name wsc --freeze_layers 0 1 2 3 4 5 6 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --parent_model roberta-base --benchmark superglue --task_name wsc --freeze_layers 0 1 3 4 5 6 11

echo "######################################################################"
echo "finished"