#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=dro_omg
#SBATCH --partition=critical
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:1    
#SBATCH --time=5-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=./slurm_script/logs/dro_omg

python main.py --seed 42 --exp_name OMG_42_dropout_0.1 --data OMNIGLOT --mode dropout --dropout_p 0.1