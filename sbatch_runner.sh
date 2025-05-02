#!/bin/bash

# This script starts sbatch runs for each supported GPU

GPU_PARTITION=("a100")
TIMESTAMP=$(date +"%m%d_%H%M%S")

GPU_COUNT=$1
if [ -z "$GPU_COUNT" ]; then
  usage
  exit 1
fi


#s == stylegan
#b == biggan
MODELS=("s", "b")

for M in "${MODELS[@]}"; do
	for GPU in "${GPU_PARTITION[@]}"; do
		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP".log --error="$M"_"$GPU"_%j_$TIMESTAMP.err --time=0-04:00 --mem=4G --partition=gpu --gres=gpu:"$GPU":$GPU_COUNT experiments/experiment.sh $(pwd)/venv310
	done
done

GPU_PREEMPT=("v100")

for M in "${MODELS[@]}"; do
	for GPU in "${GPU_PREEMPT[@]}"; do
		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP".log --error="$M"_"$GPU"_%j_"$TIMESTAMP".err --time=0-04:00 --mem=4G --partition=preempt --gres=gpu:"$GPU":1 experiments/experiment.sh $(pwd)/venv310
	done
done

