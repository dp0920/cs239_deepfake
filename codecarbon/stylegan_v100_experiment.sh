#!/bin/bash
#SBATCH --job-name=stylegan_v100_experiment
#SBATCH --output=stylegan_v100_experiment_%j.log
#SBATCH --error=stylegan_v100_experiment_%j.err
#SBATCH --time=0-01:00
#SBATCH --mem=20000
#SBATCH --partition=preempt
#SBATCH --gres=gpu:v100:1

# Activate the environment
source .venv/bin/activate
module load cuda/11.7
module load gcc/11.2.0
export CXX=$(which g++)
export CC=$(which gcc)

cd cs239_deepfake/codecarbon

# Run the Python script
python stylegan_experiment.py 1-100
