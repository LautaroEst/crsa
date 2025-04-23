#!/bin/bash -e

# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh
# source ~/miniforge3/etc/profile.d/conda.sh
conda activate crsa


# python -m crsa.scripts.parameters \
#     --world findA1_simple \
#     --pasts \
#         "1st 2nd" \
#     --alphas 2.0 \
#     --max_depths 4 \
#     --verbose

# python -m crsa.scripts.parameters \
#     --world findA1 \
#     --pasts \
#         "1st" \
#         "1st 2nd" \
#         "2nd 2nd" \
#         "3rd 2nd" \
#     --alphas 1.0 0.5 2.0 \
#     --max_depths 20 20 20

# python -m crsa.scripts.sample_conversations \
#     --world findA1 \
#     --n_conversations 10 \
#     --listener_threshold 0.95 \
#     --max_turns 5 \
#     --alpha 2 \
#     --max_depth 10 \
#     --seed 1234

python -m crsa.scripts.compare_models \
    --world findA1 \
    --models "crsa_sample" "crsa_max" \
    --n_turns 5 \
    --alpha 0.1 \
    --max_depth 10 \
    --seed 1234 \
    --n_seeds 100

# Finish
conda deactivate
