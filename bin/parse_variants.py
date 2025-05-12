"""
Module to output variants per sample in AAChange format from variant data.
"""

import pandas as pd
import numpy as np
import argparse
import pickle
import zlib
import os


def read_variants(file):
    with open(file, "rb") as f:
        compressed_data = f.read()
    loaded_dict = pickle.loads(zlib.decompress(compressed_data))
    return loaded_dict

GT_MAP = {
    "[1, 0, 0]": 1,
    "[0, 1, 0]": 1,
    "[1, 1, 0]": 2,
    "[-1, -1, 0]": np.nan
}

def parse_args():
    parser = argparse.ArgumentParser(description="Convert variants to exome sequences.")
    parser.add_argument("--variant_file", type=str, required=True, help="Path to the variants file.")
    parser.add_argument("--annot_file", type=str, required=True, help="Path to the annotation file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load the annotation file
    df_annot = pd.read_csv(args.annot_file, sep="\t")
    # Load the variants file
    variants = args.variant_file
    loaded_dict = read_variants(variants)
    samples = list(loaded_dict.keys())
    all_df_groups = []
    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"{i}, {sample}", flush=True)
        df_sample = pd.DataFrame(loaded_dict[sample])
        df_sample.columns = ["ID", "Chr", "Start", "End", "Ref", "Alt", "GT"]
        df_sample["Alt"] = df_sample.Alt.apply(lambda x: x[0])
        df_sample.Start = df_sample.Start + 1
        # merge sample with annotation
        df_merge = pd.merge(df_sample, df_annot, left_on=["ID","Chr", "Start", "End", "Ref", "Alt"], \
                            right_on=["Otherinfo6", "Chr", "Start", "End", "Ref", "Alt"], how = "left")
        df_merge["GT"] = df_merge["GT"].astype(str).map(GT_MAP)
        df_grouped = df_merge[df_merge["cds_seq"].notna()].groupby("rna_id").\
                    agg({"NuChange": list, "AAChange": list, "cds_seq": "first", "GT": list})
        df_grouped["sample"] = sample
        df_grouped.reset_index(drop=False, inplace=True)
        df_grouped.rename(columns={"index": "rna_id"}, inplace=True)
        all_df_groups.append(df_grouped)

    df_results = []
    for group in all_df_groups:
        group = group[["sample", "rna_id", "AAChange"]]
        group.columns = ["sample", "rna_id", "variants"]
        df_results.append(group)
    df_result = pd.concat(df_results)
    # make directory if not exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # save the result
    df_result.to_csv(args.output_file, sep="\t", index=False)
    print(f"Results saved to {args.output_file}", flush=True)

if __name__ == "__main__":
    main()