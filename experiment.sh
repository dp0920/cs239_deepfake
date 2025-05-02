#!/bin/bash

# This file will run the codecarbon script k times. Each run of the codecarbon script
# adds a line to the CSV file that shows metrics for the run.

usage() {
  echo "./experiment.sh ITERATION_COUNT"
}

if [ $# -ne 1 ]; then
    usage
    exit 1
fi

K=$1
IMAGE_COUNT=200
CODECARBON_SCRIPT="codecarbon/main.py"
CSV_FILE_PATH="codecarbon/codecarbon_logs/emissions.csv"


for ((i=1; i<=K; i++))
do
    python "$CODECARBON_SCRIPT" "$IMAGE_COUNT"
done

# Copying the file over to avoid overwriting
timestamp=$(date +"%m%d_%H%M%S")
cp $CSV_FILE_PATH "$timestamp"_"$K"_"$IMAGE_COUNT".csv
