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
# from torch_geometric.nn import DataParallel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


from sklearn.model_selection import train_test_split

from sgformer import SGFormer
from dataloader import ExomeDataset

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

os.environ["DEBUG"] = "1"
check_ram()

multilabel_matrix_path = "/orange/sai.zhang/UKBB/finngen/label_matrix.npy"
sample_path = "/orange/sai.zhang/UKBB/finngen/sample_uid.txt"
variant_db = join(os.getenv("PROCESSED_DATA"), "processed/sqlite/ukbb_sample_indexed_merged2.db")
embedding_db = join(os.getenv("PROCESSED_DATA"), "processed/embeddings/protein_embeddings.h5")
splits_np = np.load(join(os.getenv("PROCESSED_DATA"),"splits", "splits.npz"))
refseq_path = "/orange/sai.zhang/khoa/data/RefSeqGene/GRCh38_latest_rna.gbff.tsv"
network_path = "/orange/sai.zhang/khoa/data/biokg.csv"
REF_DB = join(os.getenv("PROCESSED_DATA_EUR"), "processed/ref_exome_auto_mane.csv")
normalization_matrix = np.load(join(os.getenv("PROCESSED_DATA"), "processed", "normalization_auto_mane.npz"))

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

# disease_index = 70
# disease_mask = (eur_multilabel_matrix[:,disease_index] == 1) #& ((eur_multilabel_matrix.sum(axis=1) == 1))
# healthy_mask = (eur_multilabel_matrix[:,disease_index] != 1) #(eur_multilabel_matrix.sum(axis=1) == 0)
# np.random.seed(1809)
# disease_indices = np.random.choice(np.where(disease_mask)[0], size=len(np.where(disease_mask)[0]), replace=False)
# healthy_indices = np.random.choice(np.where(healthy_mask)[0], size=len(np.where(disease_mask)[0]), replace=False)

# n_samples = min(np.sum(disease_mask), np.sum(healthy_mask))  # balanced size
# all_indices = np.concatenate([disease_indices, healthy_indices])
# labels = np.array([1] * n_samples + [0] * n_samples)

# train_index, temp_index, y_train, y_temp = train_test_split(
#     all_indices, labels, test_size=0.5, stratify=labels, random_state=1809
# )
# val_index, test_index, y_val, y_test = train_test_split(
#     temp_index, y_temp, test_size=0.5, stratify=y_temp, random_state=1809
# )

# save indexes 
# np.savez(join(os.getenv("PROCESSED_DATA"), "train_test_split.npz"),
#             train=train_index, val=val_index, test=test_index)
# train_index, val_index, test_index = splits_np["train"][:400], splits_np["val"][:400], splits_np["test"][:400]

alz_train_test = np.load(join(os.getenv("PROCESSED_DATA"), "alz_g30.9_train_test_split.npz"))
train_index = alz_train_test["train_index"]
val_index = alz_train_test["val_index"]
test_index = alz_train_test["test_index"]
y_train = alz_train_test["train_label"]
y_val = alz_train_test["val_label"]
y_test = alz_train_test["test_label"]
train_index = [eur_id2index[i] for i in train_index]
val_index = [eur_id2index[i] for i in val_index]
test_index = [eur_id2index[i] for i in test_index]

train_dataset = ExomeDataset(train_index, eur_index2id, exome_loader, eur_multilabel_matrix, \
                             normalization_matrix, gene_groups, binary=True, binary_labels=y_train)
val_dataset   = ExomeDataset(val_index, eur_index2id,  exome_loader, eur_multilabel_matrix,\
                             normalization_matrix, gene_groups, binary=True, binary_labels=y_val)
test_dataset  = ExomeDataset(test_index, eur_index2id, exome_loader, eur_multilabel_matrix,\
                             normalization_matrix, gene_groups, binary=True, binary_labels=y_test)

# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=8)
# val_dataloader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=8)
test_dataloader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=1)

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
hidden_dims=64
out_channels=2
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
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     model = DataParallel(model)
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
check_gpu()

# custom loss function with regularization
def custom_loss_function(logits, labels, model, l2_lambda=0.01):
    """
    Custom loss function that combines CrossEntropyLoss with L2 regularization.
    """
    # CrossEntropyLoss
    ce_loss = criterion(logits, labels)

    # L2 regularization
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)

    # Combine losses
    total_loss = ce_loss + l2_lambda * l2_reg
    return total_loss
# ───────────────────────────────────────────────────────────────────────────────

import time
import logging
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
# ───────────────────────────────────────────────────────────────────────────────

checkpoint_path = "/orange/sai.zhang/khoa/repos/ukbb_gwas/model/ckpts_alz_g30.9_061625_mane/sgformer_alz_g30.9_mane_80batch_l2_1e-4_lr_1e-3_epoch_189.pth"
# Load checkpoint
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state)
model.eval()

from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
all_probs = []
all_preds = []
y_true = []
attributions = []

logger.info("Starting interpretation")
i = 0
for x, y in test_dataloader:
    x_batch, edge, batch = toGraphBatch(x, edges)
    x_batch = x_batch.float().to(device)
    edge = edge.to(device) 
    batch = batch.to(device)
    
    # Get model prediction
    logits = model(x_batch, edge, batch)
    probs = torch.softmax(logits, dim=1)[:, 1]
    # Calculate attributions
    baseline = torch.zeros_like(x_batch)
    attr = ig.attribute(x_batch, 
                       additional_forward_args=(edge, batch),
                       target=y.numpy().item(),
                       n_steps=50,
                       internal_batch_size=1,
                       baselines=baseline)
    
    all_probs.append(probs.detach().cpu().numpy())
    y_true.append(y.numpy())
    attributions.append(attr.detach().cpu().numpy().astype(np.float16))
    if i % 100 == 0:
        logger.info(f"Processed {i} samples")
    i += 1

all_probs = np.concatenate(all_probs)
all_preds = (all_probs >= 0.5).astype(int)
y_true = np.concatenate(y_true)
attributions = np.stack(attributions, axis=0)

# save the attributions
np.save("interpret_062325_attributions_0baseline_matchtarget.npy", attributions)

# Map back to sample IDs
results = []
for idx, prob, pred, true in zip(test_index, all_probs, all_preds,y_true):
    sample_id = eur_index2id[int(idx)]
    results.append((sample_id, prob, pred, true))

# Save to CSV
import pandas as pd
df = pd.DataFrame(results, columns=['sample_id', 'prob_positive', 'pred_label', 'true_label'])
out_file = 'test_predictions_g30.9_epoch_189_062325_mane.csv'
df.to_csv(out_file, index=False)
print(f"Saved predictions to {out_file}")
