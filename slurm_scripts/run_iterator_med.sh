#!/bin/sh

# Define the runs, models, datapaths and their corresponding token files
declare -A run_names
declare -A model_names
declare -A data_paths
declare -A token_files


# Assign run names
run_names["med_without_adapt"]="med_without_adapt"
run_names["med_with_adapt"]="med_with_adapt"

model_names["med_without_adapt"]="FacebookAI/roberta-base"
model_names["med_with_adapt"]="FacebookAI/roberta-base"

# Assign datapaths
data_paths["med_without_adapt"]="/scratch/rahlab/vedant/adapt/data/med"
data_paths["med_with_adapt"]="/scratch/rahlab/vedant/adapt/data/med"

# Assign tokens files
token_files["med_without_adapt"]=""  # No token file
token_files["med_with_adapt"]="/scratch/rahlab/vedant/adapt/data/med/new_tokens.txt"

# Submit a SLURM job for each run
for key in "${!run_names[@]}"; do
    sbatch run_benchmarking.sh "$key" "${model_names[$key]}" "${data_paths[$key]}"  "${token_files[$key]}" 
done

# sbatch run_benchmarking.sh med_without_adapt /scratch/rahlab/vedant/adapt/data/med
# sbatch run_benchmarking.sh med_with_adapt /scratch/rahlab/vedant/adapt/data/med  /scratch/rahlab/vedant/adapt/data/med/new_tokens.txt 
