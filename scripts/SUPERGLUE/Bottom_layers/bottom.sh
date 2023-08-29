#!/bin/bash
#SBTACH --job-name=superglue_boolq  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/bottom/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH --time 48:00:00  # time



echo "######################################################################"
echo "layer-wise fine-tuning full WSC"
python code/fine_tuner.py --benchmark superglue --task_name wsc

# echo "######################################################################"
# echo "layer-wise fine-tuning freezing bottom 5 WIC"
# python code/fine_tuner.py --benchmark superglue --task_name wic --freeze_layers 0 1 2 3 4 

# echo "######################################################################"
# echo "layer-wise fine-tuning freezing bottom 5 COLA"
# python code/fine_tuner.py --benchmark superglue --task_name cola --freeze_layers 0 1 2 3 4 

# echo "######################################################################"
# echo "layer-wise fine-tuning freezing bottom 5 WNLI"
# python code/fine_tuner.py --benchmark glue --task_name wnli --freeze_layers 0 1 2 3 4 

# echo "######################################################################"
# echo "layer-wise fine-tuning freezing bottom 5 RTE"
# python code/fine_tuner.py --benchmark glue --task_name rte --freeze_layers 0 1 2 3 4 

# echo "######################################################################"
# echo "layer-wise fine-tuning freezing bottom 5 BOOLQ"
# python code/fine_tuner.py --benchmark superglue --task_name boolq --freeze_layers 0 1 2 3 4 

# echo "######################################################################"
# echo "layer-wise fine-tuning freezing bottom 5 CB"
# python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 1 2 3 4 

# echo "######################################################################"
# echo "layer-wise fine-tuning freezing bottom 5 COPA"
# python code/fine_tuner.py --benchmark superglue --task_name copa --freeze_layers 0 1 2 3 4 

echo "######################################################################"
echo "finished"