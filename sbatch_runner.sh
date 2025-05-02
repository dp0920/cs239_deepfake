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
declare -A VENV_BY_GPU
declare -A PARTITION_BY_GPU

VENV_BY_GPU["a100"]=$(pwd)/venv310
VENV_BY_GPU["p100"]=$(pwd)/venv310
VENV_BY_GPU["v100"]=$(pwd)/venv368

#PARTITION_BY_GPU["a100"]="gpu"
#PARTITION_BY_GPU["p100"]="gpu"
PARTITION_BY_GPU["a100"]="preempt"
PARTITION_BY_GPU["p100"]="preempt"
PARTITION_BY_GPU["v100"]="preempt"


GPUs=("a100" "p100" "v100")
>>>>>>> leftover changes from hpc
TIMESTAMP=$(date +"%m%d_%H%M%S")

#s == stylegan
#b == biggan
MODELS=("s")

# GPU partition
for M in "${MODELS[@]}"; do
	for GPU in "${GPUs[@]}"; do
		VENV=${VENV_BY_GPU["$GPU"]}
		PARTITION=${PARTITION_BY_GPU["$GPU"]}
		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP".log --error="$M"_"$GPU"_%j_$TIMESTAMP.err --time=0-04:00 --mem=4G --partition=$PARTITION --gres=gpu:"$GPU":$1 experiments/experiment.sh $VENV $GPU
	done
done

GPU_PREEMPT=("p100")

for M in "${MODELS[@]}"; do
	for GPU in "${GPU_PREEMPT[@]}"; do
		sbatch --job-name="$M"_"$GPU" --output="$M"_"$GPU"_%j_"$TIMESTAMP"_log --error="$M"_"$GPU"_%j_"$TIMESTAMP"_err --time=0-04:00 --mem=4G --partition=preempt --gres=gpu:"$GPU":1 experiments/experiment.sh $(pwd)/venv310
	done
done
