#!/bin/bash
#SBTACH --job-name=sparsity_glue_qnli  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/Sparsity/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH --time 7-00:00:00  # time
#SBATCH -G 1  # Number of GPUs

# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --task_name qnli

# echo "######################################################################"
# top layers based on sparsity: [5, 1, 8, 7, 6, 2, 9, 4, 3, 10, 0, 11]
# echo "layer-wise fine-tuning top 1"
# python code/fine_tuner.py --task_name qnli --freeze_layers 0 1 2 3 4 6 7 8 9 10 11
# echo "######################################################################"
# echo "layer-wise fine-tuning top 2"
# python code/fine_tuner.py --task_name qnli --freeze_layers 0 2 3 4 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --task_name qnli --freeze_layers 0 2 3 4 6 7 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --task_name qnli --freeze_layers 0 2 3 4 6 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --task_name qnli --freeze_layers 0 2 3 4 9 10 11

echo "######################################################################"
echo "finished"