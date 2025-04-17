import pandas as pd
import numpy as np
import argparse
import pickle
import zlib
import re
from Bio.Seq import MutableSeq
import sqlite3
import os

class NuChangeParser:

    def __init__(self, nuchange):
        
        self.pos = None
        self.ref = None
        self.alt = None
        self.change_type = None
        if "ins" in nuchange:
            self.set_type("ins")
        elif "del" in nuchange:
            self.set_type("del")
        elif "dup" in nuchange:
            self.set_type("dup")
        else:
            self.set_type("sub")
        self.parse(nuchange)

    def parse(self, nuchange):
        if self.change_type == "ins":
            self.pos = self.get_variant_postions(nuchange)
            self.ref = ""
            self.alt = nuchange.split(".")[1].split("ins")[1]
        elif self.change_type == "del":
            self.pos = self.get_variant_postions(nuchange)
            self.ref = ""
            self.alt = ""
        elif self.change_type == "dup":
            self.pos = self.get_variant_postions(nuchange)
            self.ref = nuchange.split(".")[1].split("dup")[1]
            self.alt = nuchange.split(".")[1].split("dup")[1]
        elif self.change_type == "sub":
            self.pos = self.get_variant_postions(nuchange)
            self.ref = nuchange.split(".")[1].split(str(self.pos[0]))[0]
            self.alt = nuchange.split(".")[1].split(str(self.pos[0]))[1]
        return self.pos, self.ref, self.alt
    
    def set_type(self, change_type):
        self.change_type = change_type

    def mutate(self, seq):
        seq = MutableSeq(seq)
         # Insert the new nucleotide(s) at the specified position
         # Note: pos is 1-based index, so we need to adjust it for 0-based index
        if self.change_type == "ins":
            seq[self.pos[0]:self.pos[0]] = self.alt
        elif self.change_type == "del":
            del seq[self.pos[0]-1:self.pos[-1]]
        elif self.change_type == "sub":
            seq[self.pos[0]-1:self.pos[0]-1 + len(self.ref)] = self.alt
        elif self.change_type == "dup":
            seq[self.pos[0]:self.pos[0]] = self.ref
        return str(seq)

    def get_variant_postions(self, value):
        matches = re.findall(r'\d+(?:\D+\d+)*', value)
        matches = "".join(matches).split("_")
        int_matches = [int(i) for i in matches]
        return int_matches


def read_variants(file):
    with open(file, "rb") as f:
        compressed_data = f.read()
    loaded_dict = pickle.loads(zlib.decompress(compressed_data))
    return loaded_dict

def mutate_seq(seq, nuchanges, gt):
    mutations = []
    for nuchange, genotype in zip(nuchanges, gt):
        if genotype == 1 or genotype == 2:
            parser = NuChangeParser(nuchange)
            mutations.append(parser)
    mutations.sort(key=lambda x: x.pos, reverse=True)
    # Apply all mutations from right to left
    for parser in mutations:
        seq = parser.mutate(seq)

    return str(seq)

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

    for i, group in enumerate(all_df_groups):
        group["mutated_seq"] = group.apply(
            lambda row: mutate_seq(row["cds_seq"], row["NuChange"], row["GT"]), axis=1
        )

    df_results = []
    for group in all_df_groups:
        group = group[["sample", "rna_id", "mutated_seq"]]
        group.columns = ["sample", "rna_id", "seq"]
        df_results.append(group)
    df_result = pd.concat(df_results)
    df_result.to_csv(args.output_file, sep="\t", index=False)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()