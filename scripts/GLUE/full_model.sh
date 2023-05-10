#!/bin/bash
#SBTACH --job-name=glue_full_model  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs


echo "######################################################################"
echo "sst2 full model fine-tuning"
python code/fine_tuner.py --task_name sst2

echo "######################################################################"
echo "qnli full model fine-tuning"
python code/fine_tuner.py --task_name qnli

echo "######################################################################"
echo "qqp full model fine-tuning"
python code/fine_tuner.py --task_name qqp

echo "######################################################################"
echo "rte full model fine-tuning"
python code/fine_tuner.py --task_name rte

echo "######################################################################"
echo "mnli_matched full model fine-tuning"
python code/fine_tuner.py --task_name mnli_matched

echo "######################################################################"
echo "mnli_mismatched full model fine-tuning"
python code/fine_tuner.py --task_name mnli_mismatched

echo "######################################################################"
echo "cola full model fine-tuning"
python code/fine_tuner.py --task_name cola

echo "######################################################################"
echo "stsb full model fine-tuning"
python code/fine_tuner.py --task_name stsb
