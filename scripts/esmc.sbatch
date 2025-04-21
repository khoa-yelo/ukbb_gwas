#!/bin/bash 
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=24gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/esmc%j.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

pwd; hostname; date

INPUT="$PROCESSED_DATA/chrX_splits_exomes_id/chrX_variants_id_map_split1.csv"
OUTPUT="$PROCESSED_DATA/UKBB/chrX_splits_exomes_id/chrX_split1_esmc_embeddings.pkl"

python3 $REPO/ukbb_gwas/bin/esmc.py \
    --input $INPUT \
    --output $OUTPUT