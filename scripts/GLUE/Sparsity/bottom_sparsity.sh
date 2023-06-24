#!/bin/bash
#SBTACH --job-name=bert_bottom  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/Sparsity/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --time 24:00:00  # time

# echo "######################################################################"
# # bottom layers based on fisher information: 11
# echo "layer-wise fine-tuning bottom 1 cb"
# python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

# echo "######################################################################"
# # bottom layers based on fisher information: 11
# echo "layer-wise fine-tuning bottom 1 copa"
# python code/fine_tuner.py --benchmark superglue --task_name copa --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

echo "######################################################################"
# bottom layers based on fisher information: 11
echo "layer-wise fine-tuning bottom 1 wsc"
python code/fine_tuner.py --benchmark superglue --task_name wsc --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

echo "######################################################################"
# bottom layers based on fisher information: 11
echo "layer-wise fine-tuning bottom 1 wic"
python code/fine_tuner.py --benchmark superglue --task_name wic --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

echo "######################################################################"
# bottom layers based on fisher information: 11
echo "layer-wise fine-tuning bottom 1 boolq"
python code/fine_tuner.py --benchmark superglue --task_name boolq --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

echo "######################################################################"
# bottom layer based on fisher information: 11
echo "layer-wise fine-tuning bottom 1 - mrpc"
python code/fine_tuner.py --task_name mrpc --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

echo "######################################################################"
# bottom layer based on fisher information: 11
echo "layer-wise fine-tuning bottom 1 - rte"
python code/fine_tuner.py --task_name rte --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

echo "######################################################################"
# bottom layer based on fisher information: 11
echo "layer-wise fine-tuning bottom 1 - cola"
python code/fine_tuner.py --task_name cola --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

echo "######################################################################"
# bottom layer based on fisher information: 11
echo "layer-wise fine-tuning bottom 1 - wnli"
python code/fine_tuner.py --task_name wnli --freeze_layers 0 1 2 3 4 5 6 7 8 9 10



echo "######################################################################"
echo "finished"