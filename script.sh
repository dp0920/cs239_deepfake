#!/bin/bash

module load anaconda/2021.05
conda create -n env310 python=3.10

/cluster/tufts/hpc/tools/anaconda/202105/bin/python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py