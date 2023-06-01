#!/bin/bash
#SBTACH --job-name=glue_wnli  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs

# echo "######################################################################"
# # top layers based on fisher information: [1, 4, 2, 3, 0]
# echo "layer-wise fine-tuning bottom 1"
# python code/fine_tuner.py --task_name wnli --freeze_layers 0 1 2 3 4 5 6 7 9 10 11

echo "######################################################################"
# top layers based on fisher information: [1, 4, 2, 3, 0]
echo "layer-wise fine-tuning bottom 1"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 1 2 3 4 5 6 7 9 10 11

echo "######################################################################"
echo "full model fine-tuning"
python code/fine_tuner.py --task_name wnli


echo "######################################################################"
# top layers based on fisher information: [1, 4, 2, 3, 0]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 2 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 2 3 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 3 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --task_name wnli --freeze_layers 5 6 7 8 9 10 11

echo "######################################################################"
echo "finished"