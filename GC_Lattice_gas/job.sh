#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --mem=30G
#SBATCH --gpus=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=co2
#SBATCH --mail-user=sueminl@umd.edu
#SBATCH --mail-type=END,FAIL

source $MAMBA_ROOT_PREFIX/etc/profile.d/micromamba.sh
micromamba activate thermomaps
python trial_co2.py
