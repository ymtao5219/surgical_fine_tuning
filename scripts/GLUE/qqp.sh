#!/bin/bash
#SBTACH --job-name=glue_qqp  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs


module load conda 
conda activate ds696-project # CHANGE THIS! if you have different environment names

echo "######################################################################"
full model finetuning
echo "full model fine-tuning"
python code/fine_tuner.py --benchmark glue --task_name qqp

echo "######################################################################"
# top layers based on fisher information: [1, 3, 0, 2, 5, 4, 6, 10, 7, 11, 9, 8]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 0 2 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 0 2 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 2 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --benchmark glue --task_name qqp --freeze_layers 4 6 7 8 9 10 11

echo "######################################################################"
echo "finished"