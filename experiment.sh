#!/bin/bash

# This file will run the codecarbon script k times. Each run of the codecarbon script
# adds a line to the CSV file that shows metrics for the run.

usage() {
	echo "./experiment.sh ITERATION_COUNT"
}

green_echo() {
	echo -e "\033[0;32m$1\033[0m"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

K=$1
IMAGE_COUNT=10

GIT_ROOT=$(git rev-parse --show-toplevel)
CODECARBON_SCRIPT="$GIT_ROOT/codecarbon/main.py"
for ((i = 1; i <= K; i++)); do
	green_echo "Running experiment $i..."
	python "$CODECARBON_SCRIPT" "$IMAGE_COUNT"
done

# Copying the file over to avoid overwriting
timestamp=$(date +"%m%d_%H%M%S")
CODECARBON_CSV_OUTPUT_FILE_PATH="$GIT_ROOT/codecarbon/codecarbon_logs/emissions.csv"
BATCH_SCRIPT_CSV_PATH="$GIT_ROOT/CSV/$timestamp"_"$K"_"$IMAGE_COUNT".csv
mkdir -p "$BATCH_SCRIPT_CSV_PATH"
cp "$CODECARBON_CSV_OUTPUT_FILE_PATH" "$BATCH_SCRIPT_CSV_PATH"

echo "Outputs stored in $BATCH_SCRIPT_CSV_PATH"
