#!/bin/bash
#SBATCH --job-name=stylegan_p100_experiment
#SBATCH --output=stylegan_p100_experiment_%j.log
#SBATCH --error=stylegan_p100_experiment_%j.err
#SBATCH --time=0-01:00
#SBATCH --mem=20000
#SBATCH --partition=preempt
#SBATCH --gres=gpu:p100:1

# Activate the environment
source .venv/bin/activate
module load cuda/11.7
module load gcc/11.2.0
export CXX=$(which g++)
export CC=$(which gcc)

cd cs239_deepfake/codecarbon

# Run the Python script
python stylegan_experiment.py 1-100

# CLI command
#sbatch --job-name=stylegan_p100_experiment --output=stylegan_p100_experiment_%j.log --error=stylegan_p100_experiment_%j.err --time=0-02:00 --mem=4G --partition=preempt --gres=gpu:p100:1
