#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH -p large-gpu
#SBATCH --nodes=1
#SBATCH -J get_tokens
#SBATCH -o /scratch/rahlab/vedant/adapt/output/slurm_outputs/get_tokens.out
#SBATCH -e /scratch/rahlab/vedant/adapt/output/slurm_outputs/get_tokens.err

# Load necessary modules
# module load cuda/12.2
# module load cudnn/8.1.1
. /c1/apps/anaconda/2023.03/etc/profile.d/conda.sh
conda activate /SEAS/home/g49845314/.conda/envs/nlp

# python3 /scratch/rahlab/vedant/adapt/get_tokens.py \
#     --model_name FacebookAI/roberta-base \
#     --domain_path /scratch/rahlab/vedant/adapt/data/med \
#     --output_path /scratch/rahlab/vedant/adapt/data/med

python3 /scratch/rahlab/vedant/adapt/get_tokens.py \
    --model_name FacebookAI/roberta-base \
    --domain_path /scratch/rahlab/vedant/adapt/data/fin \
    --output_path /scratch/rahlab/vedant/adapt/data/fin