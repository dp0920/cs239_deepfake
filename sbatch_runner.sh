#!/bin/bash

# This script starts sbatch runs for each supported GPU

GPU_COUNT=$1
if [ -z "$GPU_COUNT" ]; then
	usage
	exit 1
fi

VENV_BY_GPU["a100"]=$(pwd)/venv310
VENV_BY_GPU["p100"]=$(pwd)/venv310
VENV_BY_GPU["v100"]=$(pwd)/venv368

# GPU partition was not starting for these, switching to preempt
#PARTITION_BY_GPU["a100"]="gpu"
#PARTITION_BY_GPU["p100"]="gpu"
PARTITION_BY_GPU["a100"]="preempt"
PARTITION_BY_GPU["p100"]="preempt"
PARTITION_BY_GPU["v100"]="preempt"

GPUs=("a100" "p100" "v100")
TIMESTAMP=$(date +"%m%d_%H%M%S")

#s == stylegan
#b == biggan
MODELS=("s" "b")

# GPU partition
for M in "${MODELS[@]}"; do
	for GPU in "${GPUs[@]}"; do
		VENV=${VENV_BY_GPU["$GPU"]}
		PARTITION=${PARTITION_BY_GPU["$GPU"]}
		sbatch \
			--mem=4G \
			--time=0-04:00 \
			--job-name="$M"_"$GPU" \
			--output="$M"_"$GPU"_%j_"$TIMESTAMP".log \
			--error="$M"_"$GPU"_%j_$TIMESTAMP.err \
			--partition=$PARTITION \
			--gres=gpu:"$GPU":$GPU_COUNT experiments/experiment.sh $VENV $GPU
	done
done
