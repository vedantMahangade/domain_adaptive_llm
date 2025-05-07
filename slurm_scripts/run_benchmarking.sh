#!/bin/sh
#SBATCH --time=7:00:00
#SBATCH -p large-gpu
#SBATCH --nodes=1
#SBATCH -J benchmarking_%j
#SBATCH -o /scratch/rahlab/vedant/adapt/output/slurm_outputs/benchmarking_%j.out
#SBATCH -e /scratch/rahlab/vedant/adapt/output/slurm_outputs/benchmarking_%j.err

# Load necessary modules
# module load cuda/12.2
# module load cudnn/8.1.1
. /c1/apps/anaconda/2023.03/etc/profile.d/conda.sh
conda activate /SEAS/home/g49845314/.conda/envs/nlp

# Arguments passed from initiator.sh
run_name=$1
model_name=$2
data_path=$3
token_file=$4

# Record start time
start_time=$(date +%s)

# Launch script with arguments
cmd="accelerate launch /scratch/rahlab/vedant/adapt/benchmarking.py \
    --model_name $model_name \
    --dataset_path $data_path \
    --output_path /scratch/rahlab/vedant/adapt/output \
    --run_name $run_name"

# Add token file if provided
if [ -n "$token_file" ]; then
    cmd+=" --adapt $token_file"
fi

# Run the command
echo "Running: $cmd"
eval $cmd

end_time=$(date +%s)

elapsed=$(( end_time - start_time ))
echo "Execution time for $run_name benchmarking: $(($elapsed / 3600))h $((($elapsed / 60) % 60))m $(($elapsed % 60))s"
