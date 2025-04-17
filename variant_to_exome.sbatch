#!/bin/bash 
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --time=9:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/variant_exome_%j.log   # Standard output and error log
#SBATCH --array=0-50

pwd; hostname; date

# Set path to environment we want and pre-pend to PATH variable
env_path=$CONDA_ENVS/gwas/bin/
export PATH=$env_path:$PATH

# read input output folder
INPUT_FOLDER=$1 # /orange/sai.zhang/khoa/data/UKBB/chr22_splits_variants
OUTPUT_FOLDER=$2 # /orange/sai.zhang/khoa/data/UKBB/chr22_splits_exomes
ANNOT_FILE=$3 # /orange/sai.zhang/khoa/data/UKBB/annots/c22_annot.tsv

echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Annotation file: $ANNOT_FILE"

# make output folder
mkdir -p $OUTPUT_FOLDER

# Get all variants pklz files
variant_dir=$INPUT_FOLDER
variant_files=($variant_dir/*.pklz)  
num_files=${#variant_files[@]}   

# Ensure the SLURM_ARRAY_TASK_ID is within the range of available files
if [ $SLURM_ARRAY_TASK_ID -ge $num_files ]; then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds available files $num_files. Exiting."
    exit 1
fi

# Get the file corresponding to this SLURM task
variant_file=${variant_files[$SLURM_ARRAY_TASK_ID]}

# Set output file name
output_file="${OUTPUT_FOLDER}/$(basename ${variant_file%.pklz}).exome.tsv"

# Run the Python script
python variant_to_exome.py --variant_file "$variant_file" --output_file "$output_file" --annot_file "$ANNOT_FILE"

echo "Processed $variant_file into $output_file"