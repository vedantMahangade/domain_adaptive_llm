#!/bin/sh

# Define the runs, models, datapaths and their corresponding token files
declare -A run_names
declare -A model_names
declare -A data_paths
declare -A token_files


# Assign run names
run_names["fin_without_adapt_small"]="fin_without_adapt_small"
run_names["fin_with_adapt_small"]="fin_with_adapt_small"
run_names["fin_without_adapt_large"]="med_witfin_without_adapt_largeh_adapt"

# Assign model names
model_names["fin_without_adapt_small"]="FacebookAI/roberta-base"
model_names["fin_with_adapt_small"]="FacebookAI/roberta-base"
model_names["fin_without_adapt_large"]="FacebookAI/roberta-large" # large model

# Assign datapaths
data_paths["fin_without_adapt_small"]="/scratch/rahlab/vedant/adapt/data/fin"
data_paths["fin_with_adapt_small"]="/scratch/rahlab/vedant/adapt/data/fin"
data_paths["fin_without_adapt_large"]="/scratch/rahlab/vedant/adapt/data/fin"

# Assign tokens files
token_files["fin_without_adapt_small"]=""  # No token file
token_files["fin_with_adapt_small"]="/scratch/rahlab/vedant/adapt/data/fin/new_tokens_05_01.txt"
token_files["fin_without_adapt_large"]=""

# Submit a SLURM job for each run
for key in "${!run_names[@]}"; do
    sbatch run_benchmarking.sh "$key" "${model_names[$key]}" "${data_paths[$key]}"  "${token_files[$key]}" 
done

# sbatch run_benchmarking.sh fin_without_adapt_small FacebookAI/roberta-base /scratch/rahlab/vedant/adapt/data/fin
# sbatch run_benchmarking.sh fin_with_adapt_small FacebookAI/roberta-base /scratch/rahlab/vedant/adapt/data/fin  /scratch/rahlab/vedant/adapt/data/fin/new_tokens_05_01.txt 
# sbatch run_benchmarking.sh fin_without_adapt_large FacebookAI/roberta-base /scratch/rahlab/vedant/adapt/data/fin