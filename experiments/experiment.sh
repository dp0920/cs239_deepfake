#!/bin/bash

# This file will run the codecarbon script k times. Each run of the codecarbon script
# adds a line to the CSV file that shows metrics for the run.

cleanup() {
  if command -v deactivate &> /dev/null; then
    deactivate
    echo "Virtual environment deactivated."
  fi
}

usage() {
  echo "Usage: $0 /path/to/venv gpu_type"
}

green_echo() {
    echo -e "\033[0;32m$1\033[0m"
}

#Always cleanup on exit
trap cleanup EXIT

VENV_PATH=$1
if [ -z "$VENV_PATH" ]; then
  usage
  exit 1
fi

GPU_TYPE=$2
#TODO: add some validation on type of GPUs we support

#Activate the virtual environment
source "$VENV_PATH/bin/activate"

module purge
module load cuda/11.7
module load gcc/11.2.0
export CXX=$(which g++)
export CC=$(which gcc)

GIT_ROOT=$(git rev-parse --show-toplevel)
CODECARBON_SCRIPT="$GIT_ROOT/codecarbon/main.py"
timestamp=$(date +"%m%d_%H%M%S")
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CODECARBON_CSV_OUTPUT_FILE_PATH="$GIT_ROOT/CSV/$GPU_COUNT_$GPU_TYPE/$timestamp"
mkdir -p "$CODECARBON_CSV_OUTPUT_FILE_PATH"

echo "saving to $CODECARBON_CSV_OUTPUT_FILE_PATH"


# 100 iterations of codecarbon
K=100
for ((i=1; i<=K; i++))
do	
    green_echo "Running experiment $i..."
    python "$CODECARBON_SCRIPT" "$i" $CODECARBON_CSV_OUTPUT_FILE_PATH
    # delete generated images
    #rm -rf "$GIT_ROOT/codecarbon/stylega2-ada-pytorch/out"
done

echo "Outputs stored in $CODECARBON_CSV_OUTPUT_FILE_PATH"

