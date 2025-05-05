import sys
import time
import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
REPO = join(os.getenv("REPO"), "ukbb_gwas/bin")
MODEL = join(os.getenv("REPO"), "ukbb_gwas/model")
sys.path.insert(0, REPO)
sys.path.insert(0, MODEL)

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from exomeloader import VariantLoader, EmbeddingLoader, ExomeLoader
from utils import check_ram, check_gpu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from sgformer import SGFormer
from dataloader import ExomeDataset

os.environ["DEBUG"] = "1"
check_ram()

multilabel_matrix_path = "/orange/sai.zhang/UKBB/finngen/label_matrix.npy"
sample_path = "/orange/sai.zhang/UKBB/finngen/sample_uid.txt"
variant_db = join(os.getenv("PROCESSED_DATA"), "processed/sqlite/ukbb_sample_indexed_merged2.db")
embedding_db = join(os.getenv("PROCESSED_DATA"), "processed/embeddings/protein_embeddings.h5")
splits_np = np.load(join(os.getenv("PROCESSED_DATA"),"splits", "splits.npz"))
refseq_path = "/orange/sai.zhang/khoa/data/RefSeqGene/GRCh38_latest_rna.gbff.tsv"
network_path = "/orange/sai.zhang/khoa/data/biokg.csv"
REF_DB = join(os.getenv("PROCESSED_DATA_EUR"), "processed/ref_exome.csv")
normalization_matrix = np.load(join(os.getenv("PROCESSED_DATA"), "processed", "normalization.npz"))

eur_samples = VariantLoader(variant_db, "merged_variants").get_all_samples()
multilabel_matrix = np.load(multilabel_matrix_path)
sample_ids = pd.read_csv(sample_path, header = None).to_numpy().flatten()
id2index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
eur_indices = [id2index[sample_id] for sample_id in eur_samples if sample_id in id2index]
eur_multilabel_matrix = multilabel_matrix[eur_indices]
eur_id2index = {sample_id: index for index, sample_id in enumerate(eur_samples)}
eur_index2id = {index : sample_id  for index, sample_id in enumerate(eur_samples)}

df_refgenome = pd.read_csv(REF_DB)
df_refseq = pd.read_csv(refseq_path, sep = "\t")
df_network = pd.read_csv(network_path)
df_refseq["transcript"] = df_refseq.Accession.str.split(".").str[0]
df_refgenome["Gene"] = df_refgenome.transcript.map(df_refseq.set_index("transcript")["Gene"])
df_htemp = df_network[["h_type", "h_id"]]
df_ttemp = df_network[["t_type", "t_id"]]
network_genes = set(df_ttemp[df_ttemp["t_type"] == "Gene"]["t_id"].unique()).\
                union(set(df_htemp[df_htemp["h_type"] == "Gene"]["h_id"].unique()))
df_group = df_refgenome.reset_index().groupby("Gene").agg(list).sort_values("index")
df_group["new_index"] = np.arange(len(df_group))
gene_groups = df_group["index"].tolist()
df_network = df_network[(df_network.rel.str.startswith("Gene")) & (df_network.rel.str.endswith("Gene"))]
df_network["h_exome_id"] = df_network["h_id"].map(df_group["new_index"])
df_network["t_exome_id"] = df_network["t_id"].map(df_group["new_index"])
df_network = df_network.dropna()
df_network = df_network.assign(
    h_exome_id = lambda df: df.h_exome_id.astype(int),
    t_exome_id = lambda df: df.t_exome_id.astype(int),
)
edges = torch.tensor(df_network[["h_exome_id", "t_exome_id"]].values).T

variant_loader   = VariantLoader(variant_db, "merged_variants")
embedding_loader = EmbeddingLoader(embedding_db, metric="mean")
exome_loader     = ExomeLoader(variant_loader, embedding_loader, REF_DB)

train_index, val_index, test_index = splits_np["train"][:400], splits_np["val"][:400], splits_np["test"][:400]
train_dataset = ExomeDataset(train_index, eur_index2id, exome_loader, eur_multilabel_matrix, normalization_matrix, gene_groups, binary=False)
val_dataset   = ExomeDataset(val_index, eur_index2id,  exome_loader, eur_multilabel_matrix, normalization_matrix, gene_groups, binary=False)
test_dataset  = ExomeDataset(test_index, eur_index2id, exome_loader, eur_multilabel_matrix, normalization_matrix, gene_groups, binary=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=8)
val_dataloader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=8)
test_dataloader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=8)

def toGraphBatch(
    X: torch.Tensor,           # (B, N, D)
    edge_index: torch.Tensor   # (2, E)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a batch of node features X and a single shared edge_index,
    produces:
      - x_batch:       (B*N, D)
      - edge_index_batch: (2, B*E)
      - batch_vector:  (B*N,) mapping each node to [0..B-1]
    for use in PyG.
    """
    B, N, D = X.shape
    E = edge_index.size(1)

    # 1) flatten features
    x_batch = X.reshape(B * N, D)

    # 2) construct batch vector: [0..B-1] each repeated N times
    batch_vector = torch.arange(B, device=X.device).repeat_interleave(N)

    # 3) repeat & shift edge_index for each graph
    ei_rep = edge_index.repeat(1, B)  # (2, B*E)
    offsets = torch.arange(B, device=X.device).repeat_interleave(E).mul(N)
    offsets = offsets.unsqueeze(0).repeat(2, 1)  # (2, B*E)
    edge_index_batch = ei_rep + offsets

    return x_batch, edge_index_batch, batch_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
check_gpu()

in_channels=1152
hidden_dims=256
out_channels=119
model = SGFormer(
    in_channels=in_channels,
    hidden_channels=hidden_dims,
    out_channels=out_channels,
    trans_num_layers=2,
    trans_num_heads=1,
    trans_dropout=0.5,
    gnn_num_layers=3,
    gnn_dropout=0.5,
    graph_weight=0.5,
    aggregate='add'
).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
check_gpu()

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_targets = []
    all_probs   = []

    for x, y in train_loader:
        x_batch, edge, batch = toGraphBatch(x, edges)
        optimizer.zero_grad()

        logits = model(x_batch.float().to(device), edge.to(device), batch.to(device))
        loss   = criterion(logits, y.float().to(device))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        # collect for metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    all_probs   = np.vstack(all_probs)    # shape: [N, num_labels]
    all_targets = np.vstack(all_targets)  # same shape

    num_labels = all_targets.shape[1]
    pr_auc = []
    roc_auc = []
    acc    = []

    for i in range(num_labels):
        t, p = all_targets[:, i], all_probs[:, i]
        pr_auc.append(average_precision_score(t, p))
        roc_auc.append(roc_auc_score(t, p))
        preds = (p >= 0.5).astype(int)
        acc.append((preds == t).mean())

    return avg_loss, pr_auc, roc_auc, acc


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs   = []

    with torch.no_grad():
        for x, y in val_loader:
            x_batch, edge, batch = toGraphBatch(x, edges)
            logits = model(x_batch.float().to(device), edge.to(device), batch.to(device))
            loss   = criterion(logits, y.float().to(device))
            total_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    all_probs   = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    num_labels = all_targets.shape[1]
    pr_auc = [
        average_precision_score(all_targets[:, i], all_probs[:, i])
        for i in range(num_labels)
    ]
    roc_auc = [
        roc_auc_score(all_targets[:, i], all_probs[:, i])
        for i in range(num_labels)
    ]
    acc = []
    for i in range(num_labels):
        preds = (all_probs[:, i] >= 0.5).astype(int)
        acc.append((preds == all_targets[:, i]).mean())

    return avg_loss, pr_auc, roc_auc, acc


def train(model, train_loader, val_loader, optimizer, criterion,
          epochs, device, label_names=None):
    """
    Tracks per-label: loss, PR-AUC, ROC-AUC, and accuracy.
    Returns a `history` dict suitable for converting to a DataFrame later.
    """
    num_labels = train_loader.dataset[0][1].shape[0]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(num_labels)]

    # initialize history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_pr_auc': [], 'val_pr_auc': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'train_acc': [], 'val_acc': []
    }

    def fmt(metrics):
        return " | ".join(f"{n}: {m:.3f}" for n, m in zip(label_names, metrics))

    for epoch in range(1, epochs+1):
        tic = time.time()
        t_loss, t_pr, t_roc, t_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        v_loss, v_pr, v_roc, v_acc = validate(
            model, val_loader, criterion, device
        )

        # save to history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_pr_auc'].append(t_pr)
        history['val_pr_auc'].append(v_pr)
        history['train_roc_auc'].append(t_roc)
        history['val_roc_auc'].append(v_roc)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        # print summary
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {t_loss:.4f}   PR-AUC -> {fmt(t_pr)}")
        print(f"             ROC-AUC -> {fmt(t_roc)}")
        print(f"             Acc     -> {fmt(t_acc)}")
        print(f"  Val   Loss: {v_loss:.4f}   PR-AUC -> {fmt(v_pr)}")
        print(f"             ROC-AUC -> {fmt(v_roc)}")
        print(f"             Acc     -> {fmt(v_acc)}")
        print("-" * 80)
        toc = time.time()
        print("Runtime per epoch: ", np.round((toc - tic),3), flush=True)
        # save model per epoch
        # torch.save(model.state_dict(), f"sgformer_epoch_{epoch}.pth")
        # print(f"Model saved at epoch {epoch}")

    return history

history = train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=40, device=device)

# save model and history
history_df = pd.DataFrame(history)
history_df.to_csv("sgformer_history_16_workers.csv", index=False)