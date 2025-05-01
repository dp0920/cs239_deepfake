#!/bin/bash

# This script starts sbatch runs for each supported GPU

# GPU_PARTITION=("v100")
TIMESTAMP=$(date +"%m%d_%H%M%S")

#s == stylegan
#b == biggan
MODELS=("b")

# for M in "${MODELS[@]}"; do
# 	for GPU in "${GPU_PARTITION[@]}"; do
# 		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP"_log --error="$M"_"$GPU"_%j_$TIMESTAMP_err --time=0-04:00 --mem=4G --partition=gpu --gres=gpu:"$GPU":1 experiments/experiment.sh $(pwd)/venv310
# 	done
# done
#GPUs=("p100" "a100")
GPUs=("p100" "a100")
=======
GPU_PARTITION=("a100")
TIMESTAMP=$(date +"%m%d_%H%M%S")

#s == stylegan
#b == biggan
MODELS=("s")

for M in "${MODELS[@]}"; do
	for GPU in "${GPU_PARTITION[@]}"; do
		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP"_log --error="$M"_"$GPU"_%j_$TIMESTAMP_err --time=0-04:00 --mem=4G --partition=gpu --gres=gpu:"$GPU":1 experiments/experiment.sh $(pwd)/venv310
	done
done

GPU_PREEMPT=("v100")

for M in "${MODELS[@]}"; do
	for GPU in "${GPU_PREEMPT[@]}"; do
		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP"_log --error="$M"_"$GPU"_%j_"$TIMESTAMP"_err --time=0-04:00 --mem=4G --partition=preempt --gres=gpu:"$GPU":1 experiments/experiment.sh $(pwd)/venv310
	done
done

GPU_PREEMPT=("p100")

for M in "${MODELS[@]}"; do
	for GPU in "${GPU_PREEMPT[@]}"; do
		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP"_log --error="$M"_"$GPU"_%j_"$TIMESTAMP"_err --time=0-04:00 --mem=4G --partition=preempt --gres=gpu:"$GPU":1 experiments/experiment.sh $(pwd)/venv310
	done
done
