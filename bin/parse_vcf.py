"""
Parse VCF file and extract sample variants
Only save non-zero genotypes
"""
import os
from os.path import join
from collections import defaultdict
import time
import numpy as np
import pickle
import zlib
import argparse
from cyvcf2 import VCF

DATA_PATH = os.environ.get('RAW_DATA')
OUT_PATH = os.environ.get('PROCESSED_DATA')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--out_path', type=str, default=OUT_PATH)
    parser.add_argument('--vcf_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    return parser.parse_args()

def parse_vcf(vcf_file):
    print("Parsing VCF file")
    vcf = VCF(vcf_file)
    sample_names = np.array(vcf.samples)
    sample_data = defaultdict(list)
    for i, variant in enumerate(vcf):
        if i % 1000 == 0:
            print("Variant# ", i, "ID: ", variant.ID, flush=True)
        #if i == 10:
        #    break
        genotypes = variant.genotype.array()
        alt_sample_indexes = list(np.where(np.any(genotypes != 0, axis=1))[0])
        alt_genotypes = genotypes[alt_sample_indexes]
        alt_sample_names = sample_names[alt_sample_indexes]
        variant_data = (variant.ID, variant.CHROM, variant.start, variant.end, variant.REF, variant.ALT)
        for sample, genotype in zip(alt_sample_names, alt_genotypes):
            sample_data[str(sample)].append((*variant_data, genotype.tolist()))
    return sample_data

def main():

    args = parse_args()
    vcf_file = join(args.data_path, args.vcf_file)
    out_file = join(args.out_path, args.out_file)
    sample_data = parse_vcf(vcf_file)
    print("Writing to file")
    pickle_sample_data = pickle.dumps(sample_data)
    compressed_sample_data = zlib.compress(pickle_sample_data)
    with open(out_file + ".pklz", "wb") as f:
        f.write(compressed_sample_data)

    print("Write to", out_file + ".pklz")

if __name__ == '__main__':
    tic = time.time()
    main()
    toc = time.time()
    print("Run time", abs(tic - toc), "seconds")
    print("Done")
                           
