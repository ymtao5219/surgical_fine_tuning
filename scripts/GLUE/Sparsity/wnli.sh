#!/bin/bash
#SBTACH --job-name=sparsity_glue_wnli  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/Sparsity/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --time 10:00:00  # time
#SBATCH -G 1  # Number of GPUs

# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --task_name wnli

echo "######################################################################"
# top layers based on sparsity: [1, 2, 5, 4, 7, 0, 9, 6, 3, 8, 10, 11]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 2 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 3 4 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 3 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 3 6 8 9 10 11

echo "######################################################################"
echo "finished"