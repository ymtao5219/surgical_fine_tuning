#!/bin/bash
#SBTACH --job-name=glue_wnli_roberta  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs


echo "######################################################################"
echo "full model fine-tuning"
python code/fine_tuner.py --parent_model roberta-base --task_name wnli


echo "######################################################################"
# top layers based on fisher information: [11, 10, 9, 8, 7]

echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --parent_model roberta-base --task_name wnli --freeze_layers 0 1 2 3 4 5 6 7 8 9 10
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --parent_model roberta-base --task_name wnli --freeze_layers 0 1 2 3 4 5 6 7 8 9 
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --parent_model roberta-base --task_name wnli --freeze_layers 0 1 2 3 4 5 6 7 8 
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --parent_model roberta-base --task_name wnli --freeze_layers 0 1 2 3 4 5 6 7
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --parent_model roberta-base --task_name wnli --freeze_layers 0 1 2 3 4 5 6

echo "######################################################################"
echo "finished"