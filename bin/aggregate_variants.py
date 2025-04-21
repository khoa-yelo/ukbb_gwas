#!/usr/bin/env python3

import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Process variant ID mapping and concatenate split files.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--chromosome', type=str, required=True, help='Chromosome label (e.g., chr22)')
    return parser.parse_args()


def process_protein(seq: str) -> str:
    """
    Add 99 'A' if sequence does not end with '*', then trim at first '*'.
    """
    if not seq.endswith('*'):
        seq += 'A' * 99
    return seq.split('*')[0]


def process_variant_mapping(df: pd.DataFrame, chromosome: str) -> pd.DataFrame:
    df['Protein'] = df['Protein'].apply(process_protein)
    df['ID'] = df['ID'].apply(lambda x: f"chr{chromosome}_{x}")
    return df


def concat_split_files(folder: str, chromosome: str) -> pd.DataFrame:
    df_splits = []
    print(f"Looking for split files in {folder}", flush=True)
    for filename in os.listdir(folder):
        if filename.startswith('split') and filename.endswith('.tsv'):
            print(f"Processing split file: {filename}", flush=True)
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath, sep='\t')
            df['ID'] = df['ID'].apply(lambda x: f"chr{chromosome}_{x}")
            df_splits.append(df)
    return pd.concat(df_splits, ignore_index=True) if df_splits else pd.DataFrame()


def main():
    args = parse_args()
    input_file = args.input
    chromosome = args.chromosome

    parent_folder = os.path.dirname(input_file)
    output_map_file = os.path.join(parent_folder, f"chr{chromosome}_variants_id_map.csv")
    output_concat_file = os.path.join(parent_folder, f"chr{chromosome}_variants.csv")

    # Process variant mapping
    df = pd.read_csv(input_file)
    df_processed = process_variant_mapping(df, chromosome)
    df_processed.to_csv(output_map_file, index=False)

    # Concatenate split files
    df_concat = concat_split_files(parent_folder, chromosome)
    if not df_concat.empty:
        df_concat.to_csv(output_concat_file, index=False)


if __name__ == '__main__':
    main()
