#!/bin/bash -e
#SBATCH -J rsa
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1


# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh
# source ~/miniforge3/etc/profile.d/conda.sh
conda activate crsa

# Go to the crsa repo
# cd /mnt/beegfs/home/estienne/conversations_intelligens/crsa/

# python -m crsa.scripts.find_a1 "size=6_alpha=2.5"
python -m crsa.scripts.mddial "pythia_alpha=2.5"

# Finish
conda deactivate
