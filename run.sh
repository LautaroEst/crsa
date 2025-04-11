#!/bin/bash -e

# Source the Conda initialization script
# source ~/anaconda3/etc/profile.d/conda.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate crsa

# Experiments
python -m crsa.scripts.rsa_hyperparams \
    --world no_structural_zeros \
    --alphas 0.1 0.5 1.0 2.0 \
    --max_depths 100 100 100 100

# python -m crsa.scripts.rsa_dataset \
#     --dataset "tuna" \
#     --alpha 1.0 \
#     --tolerance 1e-6

# python -m crsa.scripts.yrsa_hyperparams \
#     --world toy_game \
#     --alphas 0.1 1.0 \
#     --tolerances 1e-3 1e-3

# python -m crsa.scripts.multiyrsa_hyperparams \
#     --world multiround_toy_game \
#     --alphas 1.0 \
#     --max_depths 3

python -m crsa.scripts.crsa_hyperparams \
    --world crsa_toy_game \
    --alphas 1.0 \
    --max_depths 3

# Finish
conda deactivate
