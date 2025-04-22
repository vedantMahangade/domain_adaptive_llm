#!/bin/sh
#SBATCH --time=7:00:00
#SBATCH -p large-gpu
#SBATCH --nodes=1
#SBATCH -J adapt_benchmarking
#SBATCH -o adapt_benchmarking.out
#SBATCH -e adapt_benchmarking.err



# Load necessary modules
# module load cuda/12.2
# module load cudnn/8.1.1
. /c1/apps/anaconda/2023.03/etc/profile.d/conda.sh
conda activate /SEAS/home/g49845314/.conda/envs/nlp

# Record start time
start_time=$(date +%s)

python3 get_domain_specific_tokens.py

# Record end time
end_time=$(date +%s)
# Compute and print elapsed time
elapsed=$(( end_time - start_time ))
echo "Execution Time for token finding: $(($elapsed / 3600))h $((($elapsed / 60) % 60))m $(($elapsed % 60))s"

# Record start time
start_time=$(date +%s)

accelerate launch benchmarking.py

# Record end time
end_time=$(date +%s)
# Compute and print elapsed time
elapsed=$(( end_time - start_time ))
echo "Execution Time for Benchmarking: $(($elapsed / 3600))h $((($elapsed / 60) % 60))m $(($elapsed % 60))s"