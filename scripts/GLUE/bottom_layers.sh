#!/bin/bash
#SBTACH --job-name=glue_bottom  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-m40  # Partition
#SBATCH -G 1  # Number of GPUs


# echo "######################################################################"
# # bottom layer based on fisher information: 8
# echo "layer-wise fine-tuning bottom 1 - rte"
# python code/fine_tuner.py --task_name rte --freeze_layers 0 1 2 3 4 5 6 7 9 10 11

# echo "######################################################################"
# # bottom layer based on fisher information: 8
# echo "layer-wise fine-tuning bottom 1 - qnli"
# python code/fine_tuner.py --task_name qnli --freeze_layers 0 1 2 3 4 5 6 7 9 10 11

# echo "######################################################################"
# # bottom layer based on fisher information: 11
# echo "layer-wise fine-tuning bottom 1 - cola"
# python code/fine_tuner.py --task_name cola --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

# echo "######################################################################"
# # bottom layer based on fisher information: 11
# echo "layer-wise fine-tuning bottom 1 - sst2"
# python code/fine_tuner.py --task_name sst2 --freeze_layers 0 1 2 3 4 5 6 7 8 9 10

# echo "######################################################################"
# # bottom layer based on fisher information: 10
# echo "layer-wise fine-tuning bottom 1 - mrpc"
# python code/fine_tuner.py --task_name mrpc --freeze_layers 0 1 2 3 4 5 6 7 8 9 11

# echo "######################################################################"
# # bottom layer based on fisher information: 8
# echo "layer-wise fine-tuning bottom 1 - qqp"
# python code/fine_tuner.py --task_name qqp --freeze_layers 0 1 2 3 4 5 6 7 9 10 11

echo "######################################################################"
# bottom layer based on fisher information: 10
echo "layer-wise fine-tuning bottom 1 - mnli_matched"
python code/fine_tuner.py --task_name mnli_matched --freeze_layers 0 1 2 3 4 5 6 7 8 9 11



echo "######################################################################"
echo "finished"