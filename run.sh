#!/bin/bash -ex

# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh
conda activate crsa

# Experiments
python -m crsa.scripts.run_rsa rsa/default

# Finish
conda deactivate
