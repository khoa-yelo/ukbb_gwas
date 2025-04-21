#!/bin/bash 
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=15gb                     # Job memory request
#SBATCH --time=2:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/parse_vcf_%j.log   # Standard output and error log
#SBATCH --array=0-2000

pwd; hostname; date

# Set path to environment we want and pre-pend to PATH variable
env_path=$CONDA_ENVS/gwas/bin/
export PATH=$env_path:$PATH

# read input output folder
INPUT_FOLDER=$1  #$PROCESSED_DATA/chrX_splits_2000
OUTPUT_FOLDER=$2 #$PROCESSED_DATA/chrX_splits_2000_variants

echo "Input folder: $INPUT_FOLDER"  
echo "Output folder: $OUTPUT_FOLDER"

# make output folder
mkdir -p $OUTPUT_FOLDER

# Get all VCF files in RAW_DATA folder
vcf_dir=$INPUT_FOLDER
vcf_files=($vcf_dir/*.vcf.gz)  # Array of all VCF files
num_files=${#vcf_files[@]}     # Total number of VCF files

# Ensure the SLURM_ARRAY_TASK_ID is within the range of available files
if [ $SLURM_ARRAY_TASK_ID -ge $num_files ]; then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds available files $num_files. Exiting."
    exit 1
fi

# Get the file corresponding to this SLURM task
vcf_file=${vcf_files[$SLURM_ARRAY_TASK_ID]}

# Set output file name
output_file="${OUTPUT_FOLDER}/$(basename ${vcf_file%.vcf.gz}).sample_variants"

# skip if output file already exists
if [ -f "${output_file}.pklz" ]; then
    echo "Output file $output_file already exists. Skipping."
    exit 0
fi

# Run the Python script
python $REPO/ukbb_gwas/bin/parse_vcf.py --vcf_file "$vcf_file" --out_file "$output_file"

echo "Processed $vcf_file into $output_file"