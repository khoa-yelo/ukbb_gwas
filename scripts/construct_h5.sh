#!/bin/bash

# Initialize dependency variable (empty for the first iteration)
JOB_DEP=""

# Loop over chromosomes 1 to 22 and X. Adjust or extend this list as needed.
for chrom in {1..22} X; do
    # Define paths for input, output, and log based on the current chromosome.
    INPUT="$PROCESSED_DATA/processed/embeddings/chr${chr}_esmc_embeddings.pkl"
    OUTPUT="$PROCESSED_DATA/processed/embeddings/chrom_matrix_esmc_embeddings.h5"
    LOG="$REPO/ukbb_gwas/logs/emb_to_hdf5_chrom_matrix${chr}_%j.out"

    # Submit the job. For the first job, no dependency is added.
    JOB_DEP=$(sbatch --parsable ${JOB_DEP:+--dependency=afterany:$JOB_DEP} \
        --job-name=hdf5 \
        --output="${LOG}" \
        --mem=45G \
        --time=5:00:00 \
        --wrap "python3 $REPO/ukbb_gwas/bin/construct_h5.py --input_path '${INPUT}' --output_path '${OUTPUT}' --log_level INFO")
    
    echo "Submitted job for chr_${chr} with job ID: ${JOB_DEP}"
done
