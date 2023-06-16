#!/bin/bash
#SBTACH --job-name=sparsity_glue_sst2  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/Sparsity/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH --time 10-00:00:00  # time
#SBATCH -G 1  # Number of GPUs

module load conda 
conda activate ds696-project # CHANGE THIS! if you have different environment names

# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --task_name sst2


echo "######################################################################"
# top layers based on sparsity: [10, 1, 9, 11, 8, 7, 2, 3, 5, 6, 4, 0]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 1 2 3 4 5 6 7 8 9 11
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 2 3 4 5 6 7 8 9 11
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 2 3 4 5 6 7 8 11
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 2 3 4 5 6 7 8
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --task_name sst2 --freeze_layers 0 2 3 4 5 6 7

echo "######################################################################"
echo "finished"