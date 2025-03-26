#!/bin/bash -e

# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh
conda activate crsa

# Experiments
# python -m crsa.scripts.rsa_hyperparams \
#     --world no_structural_zeros \
#     --alphas 0.1 0.5 1.0 2.0 \
#     --max_depths 100 100 100 100

# python -m crsa.scripts.rsa_dataset \
#     --dataset "tuna" \
#     --alpha 1.0 \
#     --tolerance 1e-6

python -m crsa.scripts.yrsa_hyperparams \
    --world toy_game \
    --alphas 0.1 1.0 2 \
    --tolerances 1e-8 1e-8 1e-8 \

# Finish
conda deactivate
