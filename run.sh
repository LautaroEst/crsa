#!/bin/bash -e

# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh
# source ~/miniforge3/etc/profile.d/conda.sh
conda activate crsa


# python -m crsa.scripts.parameters \
#     --world findA1 \
#     --pasts \
#         "1st" \
#         "1st 2nd" \
#         "2nd 2nd" \
#         "3rd 2nd" \
#     --alphas 1.0 0.5 2.0 \
#     --tolerance 1e-3 1e-3 1e-3

# python -m crsa.scripts.sample_conversations \
#     --world findA1 \
#     --n_conversations 10 \
#     --listener_threshold 0.95 \
#     --max_turns 5 \
#     --alpha 2 \
#     --tolerance 1e-3 \
#     --seed 1234

for p in 4 ; do
    for alpha in 1.2 1.5 2.0 ; do
        python -m crsa.scripts.run_findA1 \
            --n_possitions $p \
            --models "crsa" "memoryless_rsa" "memoryless_literal" "prior_model" \
            --n_turns 9 \
            --alpha $alpha \
            --tolerance 1e-3 \
            --metrics "accuracy" "nll" \
            --seed 1234 \
            --n_seeds 200
    done
done

# python -m crsa.scripts.run_infojigsaw \
#     --models "crsa" "memoryless_rsa" "memoryless_literal" "prior_model"\
#     --alpha 2.0 \
#     --tolerance 1e-3 \
#     --seed 1234


# Finish
conda deactivate
