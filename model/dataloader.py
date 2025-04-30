import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os.path import join


def pool_by_gene(data, gene_groups):
    pooled_data = np.vstack([
        data[idxs].mean(axis=0)
        for idxs in gene_groups
    ])
    return pooled_data

def pool_mean_std_from_index_groups(means, stds, index_groups):
    pooled_means = []
    pooled_stds = []

    for indices in index_groups:
        mu = means[indices]           # shape (k, d)
        std = stds[indices]           # shape (k, d)
        var = std ** 2                # convert to variance

        mu_group = np.mean(mu, axis=0)
        pooled_var = np.mean(var, axis=0)
        between_var = np.mean((mu - mu_group) ** 2, axis=0)
        total_var = pooled_var + between_var

        pooled_means.append(mu_group)
        pooled_stds.append(np.sqrt(total_var))  # back to std

    return np.array(pooled_means), np.array(pooled_stds)

class ExomeDataset(Dataset):

    def __init__(self, indices, sample_id_map, exome_loader, label_matrix, \
                 normalization, gene_groups, binary=False):
        """
        indices: list/array of integer sample‐IDs
        exome_loader:   your ExomeLoader instance
        sample_id_map: map from index in label matrix to sample id in UKBB
        label_matrix:   numpy array of shape (n_samples, n_labels) (, 119)
        normalization:  normalization npz file
        transform:      optional transform on features
        """
        self.indices = list(indices)
        self.sample_id_map  = sample_id_map
        self.exome_loader   = exome_loader
        self.labels         = label_matrix
        self.binary = binary
        self.gene_groups = gene_groups
        mean = normalization['mean']
        std = normalization['std']
        self.global_mean_pooled, self.global_std_pooled =\
              pool_mean_std_from_index_groups(mean, std, self.gene_groups)
        
    def __len__(self):
        return len(self.indices)

    def transform(self, x):
        epsilon = 1e-8
        x = pool_by_gene(x, self.gene_groups)
        x = x - self.global_mean_pooled
        x = x / (self.global_std_pooled + epsilon)
        return x
        
    def __getitem__(self, i):
        index = self.indices[i]
        sample_id = self.sample_id_map[index]
        exome_embeddings = self.exome_loader.get_sample_matrix([sample_id])[str(sample_id)]
        # exome_embeddings = load_emb(sample_id) - self.exome_loader.get_ref_embeddings()  
        # exome_embeddings = load_emb(sample_id)
        label  = self.labels[index]
        x = torch.from_numpy(exome_embeddings).float()
        y = torch.from_numpy(label).float()   # multi‐label float targets
        
        # sum up if binary classification
        if self.binary:
            y = y.sum().to(torch.long)
        x = self.transform(x)
        return x, y