#!/bin/bash
#SBTACH --job-name=glue_mnli_mm  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-m40  # Partition
#SBATCH --time 15-00:00:00  # time
#SBATCH -G 1  # Number of GPUs

module load conda 
conda activate ds696-project # CHANGE THIS! if you have different environment names

# echo "######################################################################"
# echo "full model fine-tuning"
# python code/fine_tuner.py --task_name mnli_mismatched


echo "######################################################################"
# top layers based on sparsity: [1, 9, 2, 10, 3, 0, 5, 8, 4, 7, 6, 11]
echo "layer-wise fine-tuning top 1"
python code/fine_tuner.py --task_name mnli_mismatched --freeze_layers 0 2 3 4 5 6 7 8 9 10 11 
echo "######################################################################"
echo "layer-wise fine-tuning top 2"
python code/fine_tuner.py --task_name mnli_mismatched --freeze_layers 0 2 3 4 5 6 7 8 10 11 
echo "######################################################################"
echo "layer-wise fine-tuning top 3"
python code/fine_tuner.py --task_name mnli_mismatched --freeze_layers 0 3 4 5 6 7 8 10 11 
echo "######################################################################"
echo "layer-wise fine-tuning top 4"
python code/fine_tuner.py --task_name mnli_mismatched --freeze_layers 0 3 4 5 6 7 8 11 
echo "######################################################################"
echo "layer-wise fine-tuning top 5"
python code/fine_tuner.py --task_name mnli_mismatched --freeze_layers 0 4 5 6 7 8 11 

echo "######################################################################"
echo "finished"