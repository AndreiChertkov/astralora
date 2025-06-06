#!/bin/bash

# Create and activate conda environment
conda create --name astralora python=3.10 -y
conda activate astralora

# Install optional package isolation
conda install -c conda-forge conda-ecosystem-user-package-isolation -y

# Install dependencies
pip install -e .

# In case of errors, install gcc and gxx
conda install gcc_linux-64 -y && conda install gxx_linux-64 -y

# Run the experiment

torchrun --standalone --nproc_per_node=2 nanogpt_fineweb/run.py --gpus 0,1 --mode digital --name digital
# torchrun --standalone --nproc_per_node=2 nanogpt_fineweb/run.py --gpus 2,3 --mode bb_one --name bb_one_rank10 --rank 10
# torchrun --standalone --nproc_per_node=2 nanogpt_fineweb/run.py --gpus 4,5 --mode bb --name bb_rank10 --rank 10 --batch_size 16

# Cleanup (optional - uncomment if you want to remove the environment after running)
# conda deactivate
# conda remove --name astralora --all -y