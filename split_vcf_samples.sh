#!/bin/bash 
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=1gb                   # Job memory request
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/split_vcf_sample_%j.log   # Standard output and error log
#SBATCH --array=0-200                  # Array range

pwd; hostname; date

module load bcftools

# Input parameters
INPUT_VCF="/orange/sai.zhang/UKBB/vcf_qc/EUR/ukb23157_cX_v1.QC3.vcf.gz"  # Path to the input VCF file
OUTPUT_DIR="/orange/sai.zhang/khoa/data/UKBB/chrX_splits_200"       # Directory to save split VCFs
NUM_SPLITS=200                         # Number of splits to create

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the list of samples
SAMPLES_FILE="samples.txt"
if [ ! -f "$SAMPLES_FILE" ]; then
    bcftools query -l "$INPUT_VCF" > "$SAMPLES_FILE"
fi

# Calculate total samples and samples per split
total_samples=$(wc -l < "$SAMPLES_FILE")
samples_per_split=$((total_samples / NUM_SPLITS))

# Determine if SLURM_ARRAY_TASK_ID is valid
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SPLITS" ]; then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds the number of splits $NUM_SPLITS. Exiting."
    exit 0
fi

# Calculate the range of samples for this task
start=$((SLURM_ARRAY_TASK_ID * samples_per_split + 1))
end=$((start + samples_per_split - 1))

# If it's the last split, include the remainder
if [ "$SLURM_ARRAY_TASK_ID" -eq $((NUM_SPLITS - 1)) ]; then
    end=$total_samples
fi

# Extract samples for this split
split_samples=$(sed -n "${start},${end}p" "$SAMPLES_FILE")

# Skip processing if no samples for this task
if [ -z "$split_samples" ]; then
    echo "No samples to process for SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID. Exiting."
    exit 0
fi

# Create a temporary file to store sample names
split_samples_file=$(mktemp)
echo "$split_samples" > "$split_samples_file"

# Output file name
output_vcf="${OUTPUT_DIR}/split_${SLURM_ARRAY_TASK_ID}.vcf.gz"

# Extract the subset of samples
bcftools view -S "$split_samples_file" -Oz -o "$output_vcf" "$INPUT_VCF"

# Clean up
rm -f "$split_samples_file"

echo "Split ${SLURM_ARRAY_TASK_ID} completed: $output_vcf"
