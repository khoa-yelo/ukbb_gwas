#!/bin/bash 
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=12gb                     # Job memory request
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/esmc%j.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

pwd; hostname; date

INPUT="/orange/sai.zhang/khoa/data/UKBB/chr21_splits_exomes_id/chr21_variants_id_map.csv"
OUTPUT="/orange/sai.zhang/khoa/data/UKBB/chr21_splits_exomes_id/chr21_esmc_embeddings.pkl"

python3 esmc.py \
    --input $INPUT \
    --output $OUTPUT