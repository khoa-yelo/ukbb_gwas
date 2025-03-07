import pandas as pd 
import numpy as np
from os.path import join, basename
import glob
import os


def process_annotation_file(df_annot, gene_sequence_dict, refseq_genes):
    selected_cols = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'Func.refGene', 'Gene.refGene', \
                 'ExonicFunc.refGene', 'AAChange.refGene', 'Otherinfo6', 'Otherinfo11']
    df_annot = df_annot[selected_cols]
    df_annot = df_annot[(df_annot["Func.refGene"]=="exonic") & (df_annot["ExonicFunc.refGene"]!= "synonymous SNV")]
    df_annot["AC"] = df_annot["Otherinfo11"].apply(lambda x: int(str(x).split(";")[-1].split("=")[-1]))
    df_annot["AF"] = df_annot["Otherinfo11"].apply(lambda x: float(str(x).split(";")[0].split("=")[-1]))
   
    # Selecting the first gene if it has a sequence
    df_annot["AAChange.refGene.selected"] = df_annot["AAChange.refGene"].apply(lambda x: x.split(",")[0] if \
                                                                               isinstance(x, str) and len(x.split(",")) > 1 else x)
    df_annot["rna_id"] = df_annot["AAChange.refGene.selected"].apply(lambda x: x.split(":")[1] if isinstance(x, str)\
                                                                      and len(x.split(":")) > 1 else np.nan)
    df_annot["NuChange"] = df_annot["AAChange.refGene.selected"].apply(lambda x: x.split(":")[-2] if isinstance(x, str) \
                                                                    and  len(x.split(":")) >= 2 else np.nan)
    df_annot["AAChange"] = df_annot["AAChange.refGene.selected"].apply(lambda x: x.split(":")[-1] if isinstance(x, str) \
                                                                    and  len(x.split(":")) >= 2 else np.nan)
    df_annot["cds_seq"] = df_annot["rna_id"].map(gene_sequence_dict)

    return df_annot

def main():
    
    print("Starting preprocessing annotations")
    refseq_data_path = "/orange/sai.zhang/khoa/data/RefSeqGene/GRCh38_latest_rna.gbff.tsv"
    df_refseq = pd.read_csv(refseq_data_path, sep = "\t")
    df_refseq["ID"] = df_refseq["Accession"].apply(lambda x: x.split(".")[0])
    refseq_genes = set(df_refseq["ID"].unique())
    gene_sequence_dict = df_refseq.set_index("ID").to_dict()["CDS"]

    DATA_PATH = "/orange/sai.zhang/UKBB/vcf_qc/EUR/"
    ANNOT_PATH = "/orange/sai.zhang/khoa/data/UKBB/annots2"
    os.makedirs(ANNOT_PATH, exist_ok = True)
    df_annot_paths = glob.glob(join(DATA_PATH, "*.hg38_multianno.txt"))
    df_annot_out_paths = []
    
    for path in df_annot_paths:
        new_name = basename(path).split("_")[1] + "_annot.tsv"
        new_path = join(ANNOT_PATH, new_name)
        df_annot_out_paths.append(new_path)
    for i in range(len(df_annot_paths)):
        df_annot = pd.read_csv(df_annot_paths[i], sep = "\t")
        df_annot = process_annotation_file(df_annot, gene_sequence_dict, refseq_genes)
        df_annot.to_csv(df_annot_out_paths[i], sep = "\t", index = False)
        print(f"Finished processing {df_annot_out_paths[i]}")

if __name__ == "__main__":
    main()