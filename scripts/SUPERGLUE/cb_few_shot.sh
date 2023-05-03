
#!/bin/bash
#SBTACH --job-name=superglue_cb_few_shot  # CHANGE THIS! for different tasks
#SBATCH --output=results/logs/%x_job_name_output_%j.log
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gypsum-titanx  # Partition
#SBATCH -G 1  # Number of GPUs


module load conda 
conda activate petl # CHANGE THIS! if you have different environment names

echo "######################################################################"
echo "few-shot learning"
echo "few-shot learning with full model fine-tuning (1 shot)"
python code/fine_tuner.py --benchmark superglue --task_name cb --few_shot 1
echo "######################################################################"
echo "few-shot learning with top-5 layers model fine-tuning (1 shot)"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 6 7 8 9 10 11 --few_shot 1

echo "######################################################################"
echo "few-shot learning with full model fine-tuning (5 shot)"
python code/fine_tuner.py --benchmark superglue --task_name cb --few_shot 5
echo "######################################################################"
echo "few-shot learning with full model fine-tuning (5 shot)"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 6 7 8 9 10 11 --few_shot 5

echo "######################################################################"
echo "few-shot learning with full model fine-tuning (20 shot)"
python code/fine_tuner.py --benchmark superglue --task_name cb --few_shot 20
echo "######################################################################"
echo "few-shot learning with full model fine-tuning (20 shot)"
python code/fine_tuner.py --benchmark superglue --task_name cb --freeze_layers 0 6 7 8 9 10 11 --few_shot 20

echo "######################################################################"
echo "finished"