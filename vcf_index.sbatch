#!/bin/bash 
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=12gb                     # Job memory request
#SBATCH --time=12:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/index_vcf_%j.log   # Standard output and error log
#SBATCH --array=0-24

pwd; hostname; date

module load bcftools

# Get all VCF files in RAW_DATA folder
# vcf_dir=$RAW_DATA
vcf_dir=$1
vcf_files=($vcf_dir/*.vcf.gz)  # Array of all VCF files
num_files=${#vcf_files[@]}     # Total number of VCF files

# Ensure the SLURM_ARRAY_TASK_ID is within the range of available files
if [ $SLURM_ARRAY_TASK_ID -ge $num_files ]; then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds available files $num_files. Exiting."
    exit 1
fi

# Get the file corresponding to this SLURM task
vcf_file=${vcf_files[$SLURM_ARRAY_TASK_ID]}
echo "Indexing $vcf_file..."
bcftools index --tbi "$vcf_file"
