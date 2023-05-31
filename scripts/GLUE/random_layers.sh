#!/bin/bash
#SBTACH --job-name=glue_random  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-titanx  # Partition
#SBATCH --time=4-11:50:00 # Time before Unity is closed
#SBATCH -G 1  # Number of GPUs

echo "######################################################################"
echo "Random layers finetuning for 3 layers"
echo "######################################################################"

echo "######################################################################"
# random layers: 0 1 8
echo "layer-wise fine-tuning random 3 - rte"
python code/fine_tuner.py --task_name rte --freeze_layers 2 3 4 5 6 7 9 10 11

echo "######################################################################"
# random layers: 1 3 10
echo "layer-wise fine-tuning random 3 - qnli"
python code/fine_tuner.py --task_name qnli --freeze_layers 0 2 4 5 6 7 8 9 11

echo "######################################################################"
# random layers: 1 6 9
echo "layer-wise fine-tuning random 3 - cola"
python code/fine_tuner.py --task_name cola --freeze_layers 0 2 3 4 5 7 8 10 11

echo "######################################################################"
# random layers: 5 10 11
echo "layer-wise fine-tuning random 3 - sst2"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 1 2 3 4 6 7 8 9

echo "######################################################################"
# random layers: 3 5 9
echo "layer-wise fine-tuning random 3 - mrpc"
python code/fine_tuner.py --task_name mrpc --freeze_layers 0 1 2 4 6 7 8 10 11

echo "######################################################################"
# random layers: 1 4 8
echo "layer-wise fine-tuning random 3 - qqp"
python code/fine_tuner.py --task_name qqp --freeze_layers 0 2 3 5 6 7 9 10 11

echo "######################################################################"
# random layers: 5, 7, 3
echo "layer-wise fine-tuning random 3 - mnli_matched"
python code/fine_tuner.py --task_name mnli_matched --freeze_layers 0 1 2 4 6 8 9 10 11



echo "######################################################################"
echo "finished"