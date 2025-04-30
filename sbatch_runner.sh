#!/bin/bash

# This script starts sbatch runs for each supported GPU

GPUs=("p100" "v100" "a100")

for GPU in "${GPUs}[@]"; do
sbatch --job-name=stylegan_$GPU_experiment --output=stylegan_$GPU_experiment_%j.log --error=stylegan_$GPU_experiment_%j.err --time=0-02:00 --mem=4G --partition=preempt --gres=gpu:$GPU:1 experiments/experiment.sh $(pwd)/venv310

done

