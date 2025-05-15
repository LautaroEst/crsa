#!/bin/bash -e
#SBATCH -J rsa
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1


# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh
# source ~/miniforge3/etc/profile.d/conda.sh
conda activate crsa

# Go to the crsa repo
# cd /mnt/beegfs/home/estienne/conversations_intelligens/crsa/


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

# llm="meta-llama/Llama-3.2-1B-Instruct"
# for p in 4 ; do
#     for alpha in 1.2 2.0 ; do
#         srun python -m crsa.scripts.run_findA1 \
#             --n_possitions $p \
#             --models "crsa" "memoryless_rsa" "memoryless_literal" "prior_model" "llmrsa_$llm" "llm_$llm" \
#             --n_turns 9 \
#             --alpha $alpha \
# 	    --max_depth 1000 \
#             --tolerance 1e-3 \
#             --metrics "accuracy" "nll" \
#             --seed 1234 \
#             --n_seeds 100
#     done
# done
# llm="meta-llama/Llama-3.2-1B-Instruct"
# llm="EleutherAI/pythia-14m"
# for alpha in 2.0 ; do
#     python -m crsa.scripts.run_infojigsaw \
#         --models "crsa" "llmrsa_$llm" \
#         --alpha $alpha \
#         --max_depth 1000 \
#         --tolerance 1e-3 \
#         --metrics "accuracy" "nll" \
#         --seed 1234
# done

# python -m crsa.scripts.run_infojigsaw \
#     --models "crsa" "memoryless_rsa" "memoryless_literal" "prior_model"\
#     --alpha 2.0 \
#     --tolerance 1e-3 \
#     --seed 1234


# python -m crsa.scripts.naive_reference_game \
#     --game_size 6 \
#     --models "crsa" "crsa_wm" "rsa" "rsa_wm" "literal" "literal_wm" "prior" \
#     --alpha 2.5 \
#     --tolerance 1e-3 \
#     --seed 7423 \
#     --n_seeds 500 \

# python -m crsa.scripts.medical_diagnosis \
#     --models "crsa" "prior" \
#     --alpha 2.5 \
#     --tolerance 1e-3 \
#     --seed 7423

python -m crsa.scripts.run_mddial \
    --base_model "EleutherAI/pythia-70m" \
    --save_every 2 \
    --seed 1234 

# Finish
conda deactivate
