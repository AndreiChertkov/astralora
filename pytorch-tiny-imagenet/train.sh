#!/bin/bash

# Simple training launcher for Pytorch Tiny ImageNet
set -e

ENV_NAME="pytorch-tiny-imagenet"

echo "Setting up conda environment: $ENV_NAME"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment..."
    conda create -n $ENV_NAME python=3.9 -y
fi

# Activate environment and install dependencies
echo "Activating environment and installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
elif [ -f "pyproject.toml" ]; then
    pip install poetry
    poetry install
fi

# Prepare dataset if needed
if [ ! -d "tiny-224" ]; then
    echo "Preparing dataset..."
    python prepare_dataset.py
fi

# Launch Python training script with all arguments
echo "Launching training..."
python launch_training.py "$@"

echo "Training completed!" 