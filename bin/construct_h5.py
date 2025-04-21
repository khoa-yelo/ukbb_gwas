#!/usr/bin/env python3
"""
Module: construct_h5.py

This script reads a pickle file containing embeddings for a single chromosome and saves them into an HDF5 file
with an optimized, matrix-style layout. The pickle file should contain a dictionary mapping embedding IDs to
dictionaries of metric arrays. For example:

    {
       "chr1_embedding_001": {"mean": np.array(...), "max": np.array(...), ...},
       "chr1_embedding_002": {"mean": np.array(...), "max": np.array(...), ...},
       ...
    }

The output HDF5 file will have one chromosome group (e.g., "/chr1"). Inside that group, each metric is stored
as a single dataset (with shape = (num_embeddings, embedding_dim)) and the dataset "embedding_ids" stores the sorted
embedding IDs corresponding to each row.
"""

import argparse
import pickle
import h5py
import numpy as np
import logging

def process_embeddings(input_path, output_path):
    logging.info("Loading embeddings from %s", input_path)
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if not data:
        logging.warning("No embeddings found in the pickle file.")
        return

    # Expect pickle file contains only one chromosome, extract the chromosome name from the first key.
    first_key = next(iter(data))
    chrom = first_key.split("_")[0]
    logging.info("Detected chromosome: %s", chrom)

    # Sort the embedding IDs for consistent ordering.
    sorted_ids = sorted(data.keys(), key=lambda x: int(x.split("_")[-1]))
    logging.info("Processing %d embeddings for %s", len(sorted_ids), chrom)

    # Assume that all embeddings have the same metrics and dimensions.
    first_embedding = data[sorted_ids[0]]
    metrics = list(first_embedding.keys())
    logging.info("Found metrics: %s", metrics)
    
    # Determine embedding dimension (e.g., each metric array has shape (embedding_dim,))
    example_array = first_embedding[metrics[0]]
    emb_dim = example_array.shape[0]

    # Prepare containers for each metric.
    metric_data = {metric: [] for metric in metrics}
    for emb_id in sorted_ids:
        if len(metric_data[metrics[0]]) % 1000 == 0:
            logging.info("Processing embedding (%d): %s", len(metric_data[metrics[0]]), emb_id)
        emb_metrics = data[emb_id]
        for metric in metrics:
            # If a metric is missing, fill with zeros.
            array = emb_metrics.get(metric, np.zeros(emb_dim, dtype=example_array.dtype))
            metric_data[metric].append(array)

    # Open (or create) the output HDF5 file in append mode.
    with h5py.File(output_path, 'a') as f_out:
        # Create or get the group corresponding to the chromosome.
        chrom_group = f_out.require_group(chrom)
        
        # For each metric, stack the arrays into a 2D matrix and create a dataset.
        for metric, arrays in metric_data.items():
            matrix = np.stack(arrays, axis=0)  # shape = (num_embeddings, embedding_dim)
            chrom_group.create_dataset(metric, data=matrix)
            logging.info("Created dataset '%s' for %s with shape %s", metric, chrom, matrix.shape)
        
        # Store the sorted embedding IDs as a dataset.
        emb_ids_array = np.array(sorted_ids, dtype="S")
        chrom_group.create_dataset("embedding_ids", data=emb_ids_array)
        logging.info("Stored embedding IDs for %s", chrom)

    logging.info("Completed writing embeddings to '%s'.", output_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert embeddings for a single chromosome from a pickle file to an HDF5 file with matrix layout."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input pickle file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output HDF5 file.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc.)")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    process_embeddings(args.input_path, args.output_path)

if __name__ == "__main__":
    main()

# sbatch --job-name=hdf5 --output=$REPO/ukbb_gwas/logs/construct_h5_%j.out --mem=25G --time=5:00:00 --wrap "python3 $REPO/ukbb_gwas/bin/construct_h5.py --input_path '$PROCESSED_DATA/processed/embeddings/chr22_esmc_embeddings.pkl' --output_path '$PROCESSED_DATA/processed/embeddings/protein_embeddings.h5' --log_level INFO"
