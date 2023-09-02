#!/bin/bash
#SBTACH --job-name=predictions_wsc  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/Predictions/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs


echo "######################################################################"
# top layers based on fisher information: [1, 5, 6, 4, 2]
echo "Predictions and activations for WSC best model"
python code/predictions.py --load_model checkpoints/best_model_wsc --benchmark super_glue --task_name wsc

echo "######################################################################"
echo "finished"