#!/bin/bash
#SBTACH --job-name=glue_sst2  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs


# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --task_name sst2


echo "######################################################################"
# top layers based on fisher information: [1, 3, 0, 4, 2]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 2 3 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 2 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --task_name sst2 --freeze_layers 2 4 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --task_name sst2 --freeze_layers 2 5 6 7 8 9 10 11
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --task_name sst2 --freeze_layers 5 6 7 8 9 10 11

echo "######################################################################"
echo "finished"