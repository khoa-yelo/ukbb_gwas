#!/usr/bin/env python3

"""
Index Exome data to save space
Exome data has form 
Sample, rna_id, seq

Unique seq across all sample is obtained and assigned index
"""

import argparse
from pathlib import Path
import pandas as pd
from Bio.Seq import Seq

def get_unique_column_values(input_dir: Path, column_index: int) -> set:
    """
    Process each TSV file in the input directory using pandas and return 
    unique values from the specified column (0-indexed).
    This function processes all regular files in the directory.
    """
    unique_values = set()
    for file_path in input_dir.iterdir():
        if not file_path.is_file():
            continue
        print(f"Processing file for unique values: {file_path}", flush=True)
        try:
            # Read only the specified column; assumes files have no header.
            df = pd.read_csv(file_path, sep="\t", header=None, usecols=[column_index])
            # Update the set with unique values from the target column.
            unique_values.update(df.iloc[:, 0].dropna().unique())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        finally:
            try:
                del df
            except NameError:
                pass
    return unique_values

def translate(seq: str) -> str:
    """Translate a nucleotide sequence to an amino acid sequence."""
    seq_obj = Seq(seq)
    protein_seq = seq_obj.translate()
    return str(protein_seq)

def process_variant_files(input_dir: Path, output_dir: Path, unique_vals_df: pd.DataFrame):
    """
    For each variant file in the input directory:
      - Map the 'seq' column to the ID from unique_vals_df,
      - Drop the 'seq' column,
      - Save the updated file to the output directory with a modified filename.
    """
    mapping = unique_vals_df.set_index("CDS")["ID"].to_dict()
    
    for file_path in input_dir.iterdir():
        if not file_path.is_file():
            continue
        print(f"Processing variant file: {file_path}", flush=True)
        try:
            df_variant = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        if "seq" not in df_variant.columns:
            print(f"Warning: 'seq' column not found in {file_path}")
            continue
            
        df_variant["ID"] = df_variant["seq"].map(mapping)
        df_variant.drop(columns=["seq"], inplace=True)
        
        # Modify filename: append '_id' before the extension.
        out_file_name = file_path.stem + "_id" + file_path.suffix
        output_file = output_dir / out_file_name
        df_variant.to_csv(output_file, index=False, sep="\t")
        print(f"Saved processed file: {output_file}")
        del df_variant

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique column values from TSV files, translate sequences, "
                    "and map variant files to generated IDs."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing input TSV files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write output files")
    parser.add_argument("--column_index", type=int, default=2,
                        help="0-indexed target column for extracting unique values (default: 2)")
    parser.add_argument("--mapping_output", default="variants_id_map.csv",
                        help="Filename for the unique values mapping CSV (default: variants_id_map.csv)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting unique values using pandas...")
    unique_vals = get_unique_column_values(input_dir, args.column_index)
    print(f"Total unique values found in column {args.column_index + 1}: {len(unique_vals)}")
    
    unique_vals_df = pd.DataFrame(unique_vals, columns=["CDS"])
    unique_vals_df["ID"] = unique_vals_df.index
    # Remove entries where the sequence equals "seq" (ignoring case)
    unique_vals_df = unique_vals_df[unique_vals_df["CDS"].str.lower() != "seq"]
    unique_vals_df["Protein"] = unique_vals_df["CDS"].apply(translate)
    
    mapping_output_path = output_dir / args.mapping_output
    unique_vals_df.to_csv(mapping_output_path, index=False)
    print(f"Unique values mapping saved to {mapping_output_path}")
    
    print("Processing variant files...")
    process_variant_files(input_dir, output_dir, unique_vals_df)
    print("Processing completed.")

if __name__ == "__main__":
    main()
