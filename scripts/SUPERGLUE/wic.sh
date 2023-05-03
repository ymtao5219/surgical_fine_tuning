#!/bin/bash
#SBTACH --job-name=superglue_wic # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit # what is max limit? 


module load conda 
conda activate petl # CHANGE THIS! if you have different environment names

echo "######################################################################"
echo "full model fine-tuning"
python code/fine_tuner.py --task_name wic


echo "######################################################################"
# top layers based on fisher information: [5, 1, 4, 3, 2]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --task_name wic --freeze_layers 0 2 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --task_name wic --freeze_layers 0 2 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --task_name wic --freeze_layers 2 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --task_name wic --freeze_layers 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --task_name wic --freeze_layers 4 6 7 8 9 10 11


echo "######################################################################"
echo "few-shot learning"
echo "few-shot learning with full model fine-tuning (1 shot)"
python code/fine_tuner.py --task_name wic --few_shot 1
echo "######################################################################"
echo "few-shot learning with top-5 layers model fine-tuning (1 shot)"
python code/fine_tuner.py --task_name wic --freeze_layers 4 6 7 8 9 10 11 --few_shot 1

echo "######################################################################"
echo "few-shot learning with full model fine-tuning (5 shot)"
python code/fine_tuner.py --task_name wic --few_shot 5
echo "######################################################################"
echo "few-shot learning with full model fine-tuning (5 shot)"
python code/fine_tuner.py --task_name wic --freeze_layers 4 6 7 8 9 10 11 --few_shot 5

echo "######################################################################"
echo "few-shot learning with full model fine-tuning (20 shot)"
python code/fine_tuner.py --task_name wic --few_shot 20
echo "######################################################################"
echo "few-shot learning with full model fine-tuning (20 shot)"
python code/fine_tuner.py --task_name wic --freeze_layers 4 6 7 8 9 10 11 --few_shot 20

echo "######################################################################"
echo "finished"