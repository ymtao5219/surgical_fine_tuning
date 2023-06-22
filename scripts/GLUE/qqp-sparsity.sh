#!/bin/bash
#SBATCH --time=6-00:00:00
#SBTACH --job-name=glue_qqp_sparsity  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs
​
​
# module load conda 
conda activate finetuning # CHANGE THIS! if you have different environment names
​
# echo "######################################################################"
# full model finetuning
# echo "full model fine-tuning"
# python code/fine_tuner.py --benchmark glue --task_name qqp
​
echo "######################################################################"
# top layers based on sparsity information: [1, 2, 0, 5, 7, 6, 3, 4, 8, 9, 10, 11]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 0 2 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 0 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 3 4 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 3 4 6 8 9 10 11
​
echo "######################################################################"
echo "finished"
