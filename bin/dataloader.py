"""
Module for loading and processing exome data and embeddings from a SQLite database and HDF5 file.
Generating a matrix of dim (NUM_TRANSCRIPT, EMBEDDING_DIM) for each sample.
"""
import os
from os.path import join
import sys
import time
from tqdm import tqdm
REPO = join(os.getenv("REPO"), "ukbb_gwas/bin")
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py
import json

from construct_sqlite import SQLiteDB

class VariantLoader:
    
    def __init__(self, database_path):
        self.database_path = database_path
        self.database = SQLiteDB(self.database_path)
        self.variant_table_name = "variants"
        self.sample_table_name = "samples"

    def get_sample_variants(self, samples: list) -> dict:
        self.database.connect()
        placeholders = ','.join('?' for _ in samples)
        query = f"SELECT * FROM {self.variant_table_name} WHERE sample IN ({placeholders})"
        df = self.database.execute_query(query, samples)
        df = df.groupby("sample", as_index=False).agg({
            "rna_id": self.merge_json_lists,
            "ID": self.merge_json_lists
        })
        self.database.close()
        result = dict(df.apply(lambda row: (str(row['sample']),\
                                                dict(zip(row['rna_id'],\
                                                row['ID']))), axis=1).tolist())
        return result
    
    def get_all_samples(self):
        self.database.connect()
        query = f"SELECT DISTINCT id FROM {self.sample_table_name}"
        samples = self.database.execute_query(query).values.flatten()
        self.database.close()
        return list(int(i) for i in samples)
    
    @staticmethod
    def merge_json_lists(series, return_json = False):
        merged_data = []
        for json_string in series:
            if pd.notna(json_string):
                merged_data.extend(json.loads(json_string))
        if return_json:
            return json.dumps(merged_data)
        return merged_data

class EmbeddingLoader:
    
    def __init__(self, database_path, metric = "mean"):
        self.database_path = database_path
        self.metric = metric
        self.embeddings = self.load_embeddings()
        
    def load_embeddings(self):
        embeddings = {}
        with h5py.File(self.database_path, 'r') as f:
            for key, data in tqdm(f.items(), desc="Loading embeddings"):
                embeddings[key] = data[self.metric][:]
        return embeddings
    
    def get_embeddings(self, chromosome:str, ids:list):
        return self.embeddings[chromosome][ids,:]
    
    def get_ref_embeddings(self):
        return self.embeddings["ref"][:]

    
class ExomeLoader:
    
    def __init__(self, variant_loader, embedding_loader, reference_db, metric = "mean"):
        self.variant_db = variant_loader
        self.embedding_db = embedding_loader
        self.reference_db = pd.read_csv(reference_db)
        
    def get_sample_matrix(self, samples):
        variants = self.variant_db.get_sample_variants(samples) # {sample:{transcript:ID,...}, sample:...}
        variants_maps = self.match_template(variants)
        sample_matrix = {}
        for sample, variant_map in variants_maps.items():
            sample_matrix[sample] = self.construct_matrix(variant_map)
            print(f"matrix {sample} constructed")
        return sample_matrix
    
    def construct_matrix(self, variant_map):
        matrix = []
        order = []
        for chrom, arr_index in variant_map["arr_index"].items():
            emb = self.embedding_db.embeddings[chrom][arr_index,:]
            matrix.append(emb)
        matrix = np.vstack(matrix)
        for chrom, reorder_index in variant_map["reorder_index"].items():
            order.extend(reorder_index)
        ordered_matrix = np.empty_like(matrix)
        ordered_matrix[order] = matrix
        return ordered_matrix
    
    def match_template(self, variants):
        chrom_maps = {}
        for sample, variant in variants.items():
            full_variants = self.reference_db.transcript.map(variant)
            full_variants = full_variants.fillna(self.reference_db.ID)
            chrom_map = self.get_mappings(full_variants)
            chrom_maps[sample] = chrom_map
        return chrom_maps

    def get_ref_embeddings(self):
        return self.embedding_db.get_ref_embeddings()
    
    @staticmethod
    def get_mappings(variants):
        df_temp = pd.DataFrame(variants.str.split("_").to_list(), columns = ["chrom", "arr_index"])
        df_temp["chrom_index"] = df_temp["chrom"].str.split("chr").str[-1]
        df_temp["chrom_index"] = df_temp["chrom_index"].replace({"ref": "0", "X": "23"}).astype(int)
        df_temp.index.name = "reorder_index"
        df_temp = df_temp.reset_index()
        df_temp.arr_index = df_temp.arr_index.astype(int)
        df_temp = df_temp.sort_values(["chrom_index", "arr_index"])
        df_temp_grouped = df_temp.groupby("chrom").agg({
            "arr_index": list,
            "reorder_index": list,
            "chrom_index": "first" 
        })

        df_temp_grouped = df_temp_grouped.sort_values("chrom_index")
        return df_temp_grouped.to_dict()