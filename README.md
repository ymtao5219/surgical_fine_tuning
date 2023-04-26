# What-Is-Transfered-In-Fine-Tuning
UMass-Microsoft Spring23 Project

- `checkpoints/` contain fine-tuned models

- `code/` python scripts for analysis 
  - `data_loader.py`: for getting samples or train/val split from GLUE/SuperGLUE Benchmark 
  - `fim_calculator`: for calculating fisher information wrt. layers
  - `fine_tuner.py`: for fine tuning bert models for different tasks
  - `utils`

- `results/` saved results, figures, logs

- `scripts/` bash scripts for extensive experiments