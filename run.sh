#!/bin/sh
#SBATCH --time=7:00:00
#SBATCH -p large-gpu
#SBATCH --nodes=1
#SBATCH -J benchmarking
#SBATCH -o /scratch/rahlab/vedant/adapt/output/slurm_outputs/benchmarking.out
#SBATCH -e /scratch/rahlab/vedant/adapt/output/slurm_outputs/benchmarking.err

# Load necessary modules
# module load cuda/12.2
# module load cudnn/8.1.1
. /c1/apps/anaconda/2023.03/etc/profile.d/conda.sh
conda activate /SEAS/home/g49845314/.conda/envs/nlp

# Record start time
start_time=$(date +%s)

# Launch script with arguments


accelerate launch benchmarking.py \
    --model_name FacebookAI/roberta-base \
    --output_path /scratch/rahlab/vedant/adapt/output \
    --run_name med_without_adapt \
    # --tqdm_disable True

accelerate launch benchmarking.py \
    --model_name FacebookAI/roberta-base \
    --adapt /scratch/rahlab/vedant/adapt/data/med \
    --output_path /scratch/rahlab/vedant/adapt/output \
    --run_name med_with_adapt \
    # --tqdm_disable True

# Record end time
end_time=$(date +%s)
# Compute and print elapsed time
elapsed=$(( end_time - start_time ))
echo "Execution Time for Benchmarking: $(($elapsed / 3600))h $((($elapsed / 60) % 60))m $(($elapsed % 60))s"
