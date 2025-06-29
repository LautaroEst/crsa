#!/bin/bash -e

# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh
conda activate crsa

python -m crsaarr.scripts.naive_reference_game \
    --game_size 6 \
    --models "crsa" "crsa_wm" "rsa" "rsa_wm" "literal" "literal_wm" "prior" \
    --alpha 2.5 \
    --tolerance 1e-3 \
    --seed 7423 \
    --n_seeds 500 \

base_model="meta-llama/Llama-3.2-1B-Instruct"
# base_model="EleutherAI/pythia-70m"
python -m crsaarr.scripts.run_mddial \
    --base_model $base_model \
    --models "crsa" "rsa" \
    --save_every 2 \
    --alpha 2.5 \
    --tolerance 1e-3 \
    --seed 1234 

# Finish
conda deactivate
