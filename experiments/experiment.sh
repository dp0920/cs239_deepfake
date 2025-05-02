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
<<<<<<< HEAD
  echo "Usage: $0 /path/to/venv"
=======
>>>>>>> leftover changes from hpc
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
<<<<<<< HEAD
VENV_PATH=../CS-239-Final-Project/env

#Activate the virtual environment
source "$VENV_PATH/bin/activate"
=======
>>>>>>> leftover changes from hpc

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

<<<<<<< HEAD
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

c
K=100
IMAGE_COUNT=50
green_echo "Starting $K iterations of generating $IMAGE_COUNT..."

GIT_ROOT=$(git rev-parse --show-toplevel)
TIMESTAMP=$(date +"%m%d_%H%M%S")
CODECARBON_SCRIPT="$GIT_ROOT/codecarbon/main.py"
CODECARBON_CSV_OUTPUT_FILE_PATH="$GIT_ROOT/CSV/$timestamp"

echo "saving to $CODECARBON_CSV_OUTPUT_FILE_PATH"

for ((i=1; i<=K; i++))
do	
    green_echo "Running experiment $i..."
    python3 "$CODECARBON_SCRIPT" "$IMAGE_COUNT" "$CODECARBON_CSV_OUTPUT_FILE_PATH" "$i"
done

# Copying the file over to avoid overwriting ORIG_CSV="$CSV_DIR/emissions.csv"
ORIG_CSV="$CSV_DIR/emissions.csv"
FINAL_CSV="$CSV_DIR/${TIMESTAMP}_${K}_${IMAGE_COUNT}.csv"

if [[ -f "$ORIG_CSV" ]]; then
  mv "$ORIG_CSV" "$FINAL_CSV"
  echo "✔ Moved $ORIG_CSV → $FINAL_CSV"
else
  echo "⚠️  Didn’t find $ORIG_CSV—nothing to rename!"
fi
=======

# 100 iterations of codecarbon
K=100
>>>>>>> leftover changes from hpc
for ((i=1; i<=K; i++))
do	
    green_echo "Running experiment $i..."
    python "$CODECARBON_SCRIPT" "$i" $CODECARBON_CSV_OUTPUT_FILE_PATH
    # delete generated images
    rm -rf "$GIT_ROOT/codecarbon/stylega2-ada-pytorch/out"
done

<<<<<<< HEAD
# Copying the file over to avoid overwriting
BATCH_SCRIPT_CSV_PATH="$GIT_ROOT/CSV/$timestamp"_"$K"_"$IMAGE_COUNT".csv
mkdir -p "$BATCH_SCRIPT_CSV_PATH"
cp "$CODECARBON_CSV_OUTPUT_FILE_PATH" "$BATCH_SCRIPT_CSV_PATH"

echo "🎉 All done—see your CSV at: $FINAL_CSV"
=======
echo "Outputs stored in $CODECARBON_CSV_OUTPUT_FILE_PATH"
>>>>>>> leftover changes from hpc

