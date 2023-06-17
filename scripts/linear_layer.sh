#!/bin/bash
#SBTACH --job-name=linear_layer_finetuning  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --time 48:00:00  # time
#SBATCH -G 1  # Number of GPUs


# echo "######################################################################"
# echo "linear layer fine-tuning cb"
# python code/fine_tuner_linear_layer.py --benchmark superglue --task_name cb

# echo "######################################################################"
# echo "linear layer fine-tuning copa"
# python code/fine_tuner_linear_layer.py --benchmark superglue --task_name copa

# echo "######################################################################"
# echo "linear layer fine-tuning multirc"
# python code/fine_tuner_linear_layer.py --benchmark superglue --task_name multirc

# echo "######################################################################"
# echo "linear layer fine-tuning wic"
# python code/fine_tuner_linear_layer.py --benchmark superglue --task_name wic

# echo "######################################################################"
# echo "linear layer fine-tuning wsc"
# python code/fine_tuner_linear_layer.py --benchmark superglue --task_name wsc

echo "######################################################################"
echo "linear layer fine-tuning boolq"
python code/fine_tuner_linear_layer.py --benchmark superglue --task_name boolq

echo "######################################################################"
echo "linear layer fine-tuning record"
python code/fine_tuner_linear_layer.py --benchmark superglue --task_name record

echo "######################################################################"
echo "linear layer fine-tuning cola"
python code/fine_tuner_linear_layer.py --benchmark glue --task_name cola

echo "######################################################################"
echo "linear layer fine-tuning mrpc"
python code/fine_tuner_linear_layer.py --benchmark glue --task_name mrpc

echo "######################################################################"
echo "linear layer fine-tuning stsb"
python code/fine_tuner_linear_layer.py --benchmark glue --task_name stsb

echo "######################################################################"
echo "linear layer fine-tuning wnli"
python code/fine_tuner_linear_layer.py --benchmark glue --task_name wnli

echo "######################################################################"
echo "linear layer fine-tuning rte"
python code/fine_tuner_linear_layer.py --benchmark glue --task_name rte



echo "######################################################################"
echo "finished"