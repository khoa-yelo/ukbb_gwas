import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pandas as pd
import argparse
import pickle
import os
import time


def embed_protein(client, protein_seq, middle_layer = 12):
    protein = ESMProtein(sequence=protein_seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
      )
    embeddings = logits_output.embeddings.squeeze()
    logits = logits_output.hidden_states[middle_layer].squeeze()

    max_ = torch.max(embeddings, dim=0).values.detach().cpu().numpy()
    mean = embeddings.mean(dim=0).detach().cpu().numpy()
    max_middle = torch.max(logits,dim=0).values.float().detach().cpu().numpy()
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
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Read the input file
    df = pd.read_csv(input_path)
    proteins = df.Protein.str.split("*").str[0].values
    ids = df.ID.values
    # Embed
    client = ESMC.from_pretrained("esmc_600m").to("cuda")
    embeddings = {}
    start = time.time()
    for i, protein in enumerate(proteins):
        res = embed_protein(client, protein)
        embeddings[ids[i]] = res
        if i % 1000 == 0:
            print(round(abs(time.time() - start)),  "seconds elapsed", flush=True)
            print(i, "proteins embedded", flush=True)
    end = time.time()
    print("Total", round(end - start), "seconds elapsed")
    # Save the embeddings
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()