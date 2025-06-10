#!/bin/bash -e
#SBATCH -J rsa
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1

# Command to run the script
if command -v srun >/dev/null 2>&1; then
    runner="srun python -m"
else
    runner="python -m"
fi

# Base directory
if [ -d "/mnt/beegfs/home/estienne/conversations_intelligens/crsa" ]; then
    base_dir="/mnt/beegfs/home/estienne/conversations_intelligens/crsa"
else
    base_dir="/home/estienne/Documents/CRSA/crsa/"
fi

# Source the Conda initialization script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate crsa

# Go to the crsa repo
cd "$base_dir"

# Run the scripts
$runner crsa.scripts.hyperparams "find_a1"
# $runner crsa.scripts.find_a1 "size=6_alpha=2.5"
# $runner crsa.scripts.mddial "llama_alpha=2.5"

# Finish
conda deactivate
