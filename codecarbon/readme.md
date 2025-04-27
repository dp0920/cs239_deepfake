opens to login node, switch to compute node with gpu
srun -t 0-01:00 --mem 20000 -p preempt --gres=gpu:v100:1 --pty bash

activate virtual environment
source .venv/bin/activate

module load cuda/11.7
module load gcc/11.2.0
export CXX=$(which g++)
export CC=$(which gcc)

rm -rf $HOME/.cache/torch_extensions (optional)

python main.py
or in stylegan2-ada-pytorch directory:
python generate.py --outdir=out --trunc=1 --seeds=2 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

-----------------------------------------------------
Initial Environment setup:
In terminal inside project folder
python3 -m venv venv
source venv/bin/activate
pip install codecarbon
pip install torch
pip install Pillow
pip install ninja
-----------------------------------------------------

