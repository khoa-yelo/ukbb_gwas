"""
Parse GenBank Flat File (GBFF) to extract CDS 
Using GRCh38_latest_rna.fna.gz                     2024-08-27 09:57  129M 
https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/
"""
from Bio import SeqIO
import pandas as pd
import os

def parse_gbff(file_path):
    """
    Parses a GBFF (GenBank Flat File) and extracts information about genes, CDS, and protein translations.

    Args:
        file_path (str): Path to the GBFF file.

    Returns:
        list: A list of dictionaries containing gene information.
    """
    parsed_data = []

    for record in SeqIO.parse(file_path, "genbank"):
        accession = record.id
        organism = record.annotations.get("organism", "Unknown")
        # get sequence
        seq = str(record.seq)
        for feature in record.features:
            if feature.type == "CDS":
                gene_name = feature.qualifiers.get("gene", ["Unknown"])[0]
                protein_id = feature.qualifiers.get("protein_id", ["Unknown"])[0]
                translation = feature.qualifiers.get("translation", ["Unknown"])[0]
                locus_tag = feature.qualifiers.get("locus_tag", ["Unknown"])[0]
                start = int(feature.location.start)
                end = int(feature.location.end)
                strand = feature.location.strand

                parsed_data.append({
                    "Sequence": seq,
                    "Accession": accession,
                    "Organism": organism,
                    "Gene": gene_name,
                    "Locus_Tag": locus_tag,
                    "Protein_ID": protein_id,
                    "Start": start,
                    "End": end,
                    "Strand": strand,
                    "Protein_Sequence": translation
                })

    return parsed_data

def main():
    my_folder = os.getenv("KHOA")
    parsed_results = parse_gbff(os.path.join(my_folder, "data/RefSeqGene/GRCh38_latest_rna.gbff"))
    df_ref = pd.DataFrame(parsed_results)
    # chunk from Start to End of Sequence
    df_ref["CDS"] = df_ref.apply(lambda x: x["Sequence"][x["Start"]:x["End"]], axis = 1)
    df_ref.to_csv(os.path.join(my_folder, "data/RefSeqGene/GRCh38_latest_rna.gbff.tsv"), sep = "\t", index = False)

if __name__ == "__main__":
    main()