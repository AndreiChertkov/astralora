#!/bin/bash

# Create and activate conda environment

conda create --name optinet python=3.10.17 -y
source activate optinet

# Install dependencies
pip install -r ../requirements.txt -r ../test_requirements.txt

# Run the experiment
python ../run_experiment.py --run-name nanogpt_ts_ffttiles_sur --config ../experiments/ts/nanogpt_ts_ffttiles_sur.yaml

# Cleanup (optional - uncomment if you want to remove the environment after running)
# conda deactivate
# conda remove --name optinet --all -y