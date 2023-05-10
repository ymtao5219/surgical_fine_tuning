# What-Is-Transfered-In-Fine-Tuning
UMass-Microsoft Spring23 Project

- `checkpoints/` contains fine-tuned models

- `configs/` contains the configurations being used to train the GLUE and SuperGLUE tasks

- `code/` python scripts for analysis 
  - `data_loader.py`: for getting samples or train/val split from GLUE/SuperGLUE Benchmark 
  - `fim_calculator`: for calculating fisher information wrt. layers
  - `fine_tuner.py`: for fine tuning bert models for different tasks
  - `utils`

- `results/` saved results, figures, logs

- `scripts/` bash scripts for extensive experiments: Please change the parameters in the first 6-7 of lines of the script being run according to the GPU being used. Example of command for running: 

      sbatch scripts/GLUE/full_model.sh
---