"""
ESM-C Protein Embedding Script
"""

import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pandas as pd
import argparse
import pickle
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# default result dict with zeros (1152,) np.array
default_result = {
    "max": torch.zeros((1152,), dtype=torch.float32).detach().cpu().numpy(),
    "mean": torch.zeros((1152,), dtype=torch.float32).detach().cpu().numpy(),
    "max_middle_layer_12": torch.zeros((1152,), dtype=torch.float32).detach().cpu().numpy(),
    "mean_middle_layer_12": torch.zeros((1152,), dtype=torch.float32).detach().cpu().numpy()
}

def embed_protein(client, protein_seq, middle_layer=12):
    # Check if the protein sequence is empty or NaN.
    # If so, return the default result.
    if not protein_seq or pd.isna(protein_seq):
        return default_result
    protein = ESMProtein(sequence=protein_seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
    )
    embeddings = logits_output.embeddings.squeeze()
    logits = logits_output.hidden_states[middle_layer].squeeze()

    max_ = torch.max(embeddings, dim=0).values.detach().cpu().numpy()
    mean = embeddings.mean(dim=0).detach().cpu().numpy()
    max_middle = torch.max(logits, dim=0).values.float().detach().cpu().numpy()
    mean_middle = logits.mean(dim=0).float().detach().cpu().numpy()
    
    result = {
        "max": max_,
        "mean": mean,
        f"max_middle_layer_{middle_layer}": max_middle,
        f"mean_middle_layer_{middle_layer}": mean_middle
    }
    return result

def parse_arguments():
    parser = argparse.ArgumentParser(description="ESM-C Protein Embedding")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_path = args.input_file
    output_path = args.output_file

    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Read the input file
    logger.info(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path)
    proteins = df.Protein.str.split("*").str[0].values
    ids = df.ID.values

    # Initialize the ESMC client on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    client = ESMC.from_pretrained("esmc_600m").to(device)
    embeddings = {}
    start = time.time()

    # Process each protein and log progress every 1000 entries
    for i, protein in enumerate(proteins):
        res = embed_protein(client, protein)
        embeddings[ids[i]] = res
        if i % 1000 == 0 and i > 0:
            elapsed = round(time.time() - start)
            logger.info(f"{elapsed} seconds elapsed, {i} proteins embedded")

    end = time.time()
    total_elapsed = round(end - start)
    logger.info(f"Total {total_elapsed} seconds elapsed")

    # Save the embeddings to disk
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    main()
