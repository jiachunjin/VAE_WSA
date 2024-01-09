#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=critical
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=./slurm_script/logs/jupyter.log

jupyter lab --ip=0.0.0.0 --port=8888