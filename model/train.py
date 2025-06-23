import sys
import time
import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import DataParallel
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import wandb
from transformers import get_cosine_schedule_with_warmup

# Add paths to system
REPO = join(os.getenv("REPO"), "ukbb_gwas/bin")
MODEL = join(os.getenv("REPO"), "ukbb_gwas/model")
sys.path.insert(0, REPO)
sys.path.insert(0, MODEL)

# Import custom modules
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from exomeloader import VariantLoader, EmbeddingLoader, ExomeLoader
from utils import check_ram, check_gpu
from sgformer import SGFormer
from dataloader import ExomeDataset

# Set up logging
def setup_logging():
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
    return logger

logger = setup_logging()

# Constants and Configuration
class Config:
    DEBUG = "1"
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    IN_CHANNELS = 1152
    HIDDEN_DIMS = 64
    OUT_CHANNELS = 2
    LEARNING_RATE = 1e-4
    L2_LAMBDA = 0.001
    EPOCHS = 250
    CKPT_DIR = "ckpts_alz_g30.9_061625_mane"
    MODEL_NAME = "sgformer_alz_g30.9_mane_32batch_l2_0.001_lr_1e-4"
    
    # Learning rate scheduler config
    WARMUP_RATIO = 0.05  # 10% of total steps for warmup
    NUM_CYCLES = 0.5    # one half-cycle of cosine
    
    # Wandb config
    WANDB_PROJECT = "ukbb"
    WANDB_ENTITY = None  # Set to None to use default entity
    WANDB_TAGS = ["alzheimer", "gwas", "graph", "sgformer"]

# Data Loading and Preprocessing
def load_data():
    """Load and preprocess all required data"""
    # Define paths
    multilabel_matrix_path = "/orange/sai.zhang/UKBB/finngen/label_matrix.npy"
    sample_path = "/orange/sai.zhang/UKBB/finngen/sample_uid.txt"
    variant_db = join(os.getenv("PROCESSED_DATA"), "processed/sqlite/ukbb_sample_indexed_merged2.db")
    embedding_db = join(os.getenv("PROCESSED_DATA"), "processed/embeddings/protein_embeddings.h5")
    refseq_path = "/orange/sai.zhang/khoa/data/RefSeqGene/GRCh38_latest_rna.gbff.tsv"
    network_path = "/orange/sai.zhang/khoa/data/biokg.csv"
    REF_DB = join(os.getenv("PROCESSED_DATA_EUR"), "processed/ref_exome_auto_mane.csv")
    normalization_matrix = np.load(join(os.getenv("PROCESSED_DATA"), "processed", "normalization_auto_mane.npz"))
    count_path = "/orange/sai.zhang/khoa/repos/ukbb_gwas/notebooks/variant_counts_all.csv"

    # Initialize loaders
    embedding_loader = EmbeddingLoader(embedding_db, metric="mean")
    variant_loader = VariantLoader(variant_db, "merged_variants")
    exome_loader = ExomeLoader(variant_loader, embedding_loader, REF_DB)    

    # Load and process data
    eur_samples = VariantLoader(variant_db, "merged_variants").get_all_samples()
    multilabel_matrix = np.load(multilabel_matrix_path)
    sample_ids = pd.read_csv(sample_path, header=None).to_numpy().flatten()
    
    # Create mappings
    id2index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    eur_indices = [id2index[sample_id] for sample_id in eur_samples if sample_id in id2index]
    eur_multilabel_matrix = multilabel_matrix[eur_indices]
    eur_id2index = {sample_id: index for index, sample_id in enumerate(eur_samples)}
    eur_index2id = {index: sample_id for index, sample_id in enumerate(eur_samples)}

    # Load reference data
    df_refgenome = pd.read_csv(REF_DB)
    df_refseq = pd.read_csv(refseq_path, sep="\t")
    df_network = pd.read_csv(network_path)
    
    # Process reference data
    df_refseq["transcript"] = df_refseq.Accession.str.split(".").str[0]
    df_refgenome["Gene"] = df_refgenome.transcript.map(df_refseq.set_index("transcript")["Gene"])
    
    # Process network data
    df_htemp = df_network[["h_type", "h_id"]]
    df_ttemp = df_network[["t_type", "t_id"]]
    network_genes = set(df_ttemp[df_ttemp["t_type"] == "Gene"]["t_id"].unique()).union(
        set(df_htemp[df_htemp["h_type"] == "Gene"]["h_id"].unique())
    )
    
    # Group and process gene data
    df_group = df_refgenome.reset_index().groupby("Gene").agg(list).sort_values("index")
    df_group["new_index"] = np.arange(len(df_group))
    
    # Load and process count data
    df_count = pd.read_csv(count_path)
    df_refgenome["count"] = df_refgenome.transcript.map(
        df_count[["rna_id", "count"]].set_index("rna_id").to_dict()["count"]
    )
    count_dict = df_refgenome.to_dict()["count"]
    
    # Process gene groups
    df_group['best_index'] = df_group['index'].apply(
        lambda idx_list: [max(idx_list, key=count_dict.get)]
    )
    gene_groups = df_group["index"].tolist()
    
    # Process network edges
    df_network = df_network[(df_network.rel.str.startswith("Gene")) & (df_network.rel.str.endswith("Gene"))]
    df_network["h_exome_id"] = df_network["h_id"].map(df_group["new_index"])
    df_network["t_exome_id"] = df_network["t_id"].map(df_group["new_index"])
    df_network = df_network.dropna()
    df_network = df_network.assign(
        h_exome_id=lambda df: df.h_exome_id.astype(int),
        t_exome_id=lambda df: df.t_exome_id.astype(int),
    )
    edges = torch.tensor(df_network[["h_exome_id", "t_exome_id"]].values).T

    return {
        'eur_samples': eur_samples,
        'eur_multilabel_matrix': eur_multilabel_matrix,
        'eur_id2index': eur_id2index,
        'eur_index2id': eur_index2id,
        'gene_groups': gene_groups,
        'edges': edges,
        'normalization_matrix': normalization_matrix,
        'exome_loader': exome_loader,
        'embedding_loader': embedding_loader
    }

def create_data_loaders(data_dict):
    """Create train, validation, and test data loaders"""
    # Load train/test split
    alz_train_test = np.load(join(os.getenv("PROCESSED_DATA"), "alz_g30.9_train_test_split.npz"))
    train_index = [data_dict['eur_id2index'][i] for i in alz_train_test["train_index"]]
    val_index = [data_dict['eur_id2index'][i] for i in alz_train_test["val_index"]]
    test_index = [data_dict['eur_id2index'][i] for i in alz_train_test["test_index"]]
    
    # Create datasets
    train_dataset = ExomeDataset(
        train_index, data_dict['eur_index2id'], data_dict['exome_loader'],
        data_dict['eur_multilabel_matrix'], data_dict['normalization_matrix'],
        data_dict['gene_groups'], binary=True, normalize=True,
        binary_labels=alz_train_test["train_label"], pool=False
    )
    val_dataset = ExomeDataset(
        val_index, data_dict['eur_index2id'], data_dict['exome_loader'],
        data_dict['eur_multilabel_matrix'], data_dict['normalization_matrix'],
        data_dict['gene_groups'], binary=True, normalize=True,
        binary_labels=alz_train_test["val_label"], pool=False
    )
    test_dataset = ExomeDataset(
        test_index, data_dict['eur_index2id'], data_dict['exome_loader'],
        data_dict['eur_multilabel_matrix'], data_dict['normalization_matrix'],
        data_dict['gene_groups'], binary=True, normalize=True,
        binary_labels=alz_train_test["test_label"], pool=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader

# Model and Training Functions
def create_model(device):
    """Create and initialize the model"""
    model = SGFormer(
        in_channels=Config.IN_CHANNELS,
        hidden_channels=Config.HIDDEN_DIMS,
        out_channels=Config.OUT_CHANNELS,
        trans_num_layers=2,
        trans_num_heads=1,
        trans_dropout=0.5,
        gnn_num_layers=3,
        gnn_dropout=0.5,
        graph_weight=0.5,
        aggregate='add'
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    return model

def custom_loss_function(logits, labels, model, l2_lambda=Config.L2_LAMBDA):
    """Custom loss function combining CrossEntropyLoss with L2 regularization"""
    ce_loss = nn.CrossEntropyLoss()(logits, labels)
    l2_reg = sum(torch.norm(param) for param in model.parameters())
    total_loss = ce_loss + l2_lambda * l2_reg
    return total_loss, ce_loss, l2_reg * l2_lambda

def toGraphBatch(X: torch.Tensor, edge_index: torch.Tensor):
    """Convert batch data to graph format"""
    B, N, D = X.shape
    E = edge_index.size(1)
    
    x_batch = X.reshape(B * N, D)
    batch_vector = torch.arange(B, device=X.device).repeat_interleave(N)
    
    ei_rep = edge_index.repeat(1, B)
    offsets = torch.arange(B, device=X.device).repeat_interleave(E).mul(N)
    offsets = offsets.unsqueeze(0).repeat(2, 1)
    edge_index_batch = ei_rep + offsets
    
    return x_batch, edge_index_batch, batch_vector

def create_model_and_optimizer(device, train_loader):
    """Create model, optimizer, and scheduler"""
    model = create_model(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Calculate total steps and warmup steps
    total_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    # Create scheduler using transformers implementation
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=Config.NUM_CYCLES,
        last_epoch=-1
    )
    
    return model, optimizer, scheduler

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, edges, global_step):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_l2_loss = 0.0
    all_targets = []
    all_probs = []

    for batch_idx, (x, y) in enumerate(train_loader):
        x_batch, edge, batch = toGraphBatch(x, edges)
        
        optimizer.zero_grad()
        logits = model(
            x_batch.float().to(device),
            edge.to(device),
            batch.to(device)
        )
        y_labels = y.view(-1).long().to(device)
        
        loss, ce_loss, l2_loss = custom_loss_function(logits, y_labels, model)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # Log gradients before optimizer step
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({
                    f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()),
                    "global_step": global_step
                })
        
        optimizer.step()
        scheduler.step()  # Step the scheduler
        
        # Log parameters after optimizer step
        for name, param in model.named_parameters():
            wandb.log({
                f"parameters/{name}": wandb.Histogram(param.data.cpu().numpy()),
                "global_step": global_step
            })
        
        # Log learning rate and loss components
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({
            "batch_learning_rate": current_lr,
            "batch/total_loss": loss.item(),
            "batch/ce_loss": ce_loss.item(),
            "batch/l2_loss": l2_loss.item(),
            "global_step": global_step
        })
        
        total_loss += loss.item() * y_labels.size(0)
        total_ce_loss += ce_loss.item() * y_labels.size(0)
        total_l2_loss += l2_loss.item() * y_labels.size(0)
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(y_labels.cpu().numpy())
        
        global_step += 1
    
    # Calculate average losses
    avg_loss = total_loss / len(train_loader.dataset)
    avg_ce_loss = total_ce_loss / len(train_loader.dataset)
    avg_l2_loss = total_l2_loss / len(train_loader.dataset)
    
    # Log epoch-level loss components
    wandb.log({
        "epoch/total_loss": avg_loss,
        "epoch/ce_loss": avg_ce_loss,
        "epoch/l2_loss": avg_l2_loss,
        "global_step": global_step
    })
    
    metrics = calculate_metrics(all_probs, all_targets, total_loss, len(train_loader.dataset))
    return metrics, global_step

def validate(model, val_loader, criterion, device, edges, global_step):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_l2_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x_batch, edge, batch = toGraphBatch(x, edges)
            logits = model(
                x_batch.float().to(device),
                edge.to(device),
                batch.to(device)
            )
            y_labels = y.view(-1).long().to(device)
            
            loss, ce_loss, l2_loss = custom_loss_function(logits, y_labels, model)
            total_loss += loss.item() * y_labels.size(0)
            total_ce_loss += ce_loss.item() * y_labels.size(0)
            total_l2_loss += l2_loss.item() * y_labels.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_labels.cpu().numpy())
    
    # Calculate average losses
    avg_loss = total_loss / len(val_loader.dataset)
    avg_ce_loss = total_ce_loss / len(val_loader.dataset)
    avg_l2_loss = total_l2_loss / len(val_loader.dataset)
    
    # Log validation loss components with global step
    wandb.log({
        "val/total_loss": avg_loss,
        "val/ce_loss": avg_ce_loss,
        "val/l2_loss": avg_l2_loss,
        "global_step": global_step
    })
    
    metrics = calculate_metrics(all_probs, all_targets, total_loss, len(val_loader.dataset))
    return metrics

def calculate_metrics(all_probs, all_targets, total_loss, dataset_size):
    """Calculate evaluation metrics"""
    avg_loss = total_loss / dataset_size
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    pr_auc = average_precision_score(all_targets, all_probs)
    roc_auc = roc_auc_score(all_targets, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    acc = (preds == all_targets).mean()
    
    return avg_loss, pr_auc, roc_auc, acc

def init_wandb(config):
    """Initialize wandb run with configuration"""
    # Get API key from environment variable
    api_key = os.getenv("WANDB_API_KEY")
    if api_key is None:
        raise ValueError("WANDB_API_KEY environment variable not set. Please set it in your .bashrc file.")
    
    wandb_config = {
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "l2_lambda": config.L2_LAMBDA,
        "epochs": config.EPOCHS,
        "in_channels": config.IN_CHANNELS,
        "hidden_dims": config.HIDDEN_DIMS,
        "out_channels": config.OUT_CHANNELS,
        "model_name": config.MODEL_NAME,
        # Add learning rate scheduler config
        "warmup_ratio": config.WARMUP_RATIO,
        "scheduler": "cosine_with_warmup"
    }
    
    # Initialize wandb with or without entity
    init_kwargs = {
        "project": config.WANDB_PROJECT,
        "config": wandb_config,
        "tags": config.WANDB_TAGS,
        "name": config.MODEL_NAME  # Use model name as run name
    }
    
    # Only add entity if it's specified
    if config.WANDB_ENTITY is not None:
        init_kwargs["entity"] = config.WANDB_ENTITY
    
    run = wandb.init(**init_kwargs)
    return run

def log_metrics_to_wandb(metrics, prefix="train", step=None):
    """Log metrics to wandb"""
    wandb_metrics = {
        f"{prefix}/loss": metrics[0],
        f"{prefix}/pr_auc": metrics[1],
        f"{prefix}/roc_auc": metrics[2],
        f"{prefix}/accuracy": metrics[3]
    }
    wandb.log(wandb_metrics, step=step)

def save_checkpoint(model, epoch, history, is_final=False):
    """Save model checkpoint and history"""
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    
    # Save model
    if is_final:
        model_path = f"{Config.CKPT_DIR}/{Config.MODEL_NAME}.pth"
    else:
        model_path = f"{Config.CKPT_DIR}/{Config.MODEL_NAME}_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save history
    if is_final:
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{Config.CKPT_DIR}/{Config.MODEL_NAME}.csv", index=False)
        
        # Log model artifact to wandb
        artifact = wandb.Artifact(
            name=f"{Config.MODEL_NAME}",
            type="model",
            description=f"Final model checkpoint for {Config.MODEL_NAME}"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, edges, label_name="Positive"):
    """Main training loop"""
    history = {
        'train_loss': [], 'val_loss': [],
        'train_pr_auc': [], 'val_pr_auc': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'train_acc': [], 'val_acc': [],
        'learning_rate': []  # Track learning rate
    }
    
    # Log model architecture to wandb with more detailed logging
    wandb.watch(
        model,
        log="all",
        log_freq=10,
        log_graph=True,
        idx=0
    )
    
    # Initialize global step counter
    global_step = 0
    
    for epoch in range(1, Config.EPOCHS + 1):
        start = time.time()
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rate'].append(current_lr)
        
        # Train and validate
        (t_loss, t_pr, t_roc, t_acc), global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, edges, global_step
        )
        v_loss, v_pr, v_roc, v_acc = validate(
            model, val_loader, criterion, device, edges, global_step
        )
        
        # Update history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_pr_auc'].append(t_pr)
        history['val_pr_auc'].append(v_pr)
        history['train_roc_auc'].append(t_roc)
        history['val_roc_auc'].append(v_roc)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        # Log metrics to wandb using the same global step
        metrics = {
            "train/loss": t_loss,
            "train/pr_auc": t_pr,
            "train/roc_auc": t_roc,
            "train/accuracy": t_acc,
            "val/loss": v_loss,
            "val/pr_auc": v_pr,
            "val/roc_auc": v_roc,
            "val/accuracy": v_acc,
            "learning_rate": current_lr,
            "epoch": epoch
        }
        wandb.log(metrics, step=global_step)
        
        # Log progress
        log_epoch_progress(epoch, t_loss, t_pr, t_roc, t_acc, v_loss, v_pr, v_roc, v_acc, start, label_name)
        
        # Save checkpoint
        if epoch % 1 == 0:
            save_checkpoint(model, epoch, history)
            logger.info(f"Model saved at epoch {epoch}")
            logger.info("-" * 80)
    
    # Save final model
    save_checkpoint(model, Config.EPOCHS, history, is_final=True)
    return history

def log_epoch_progress(epoch, t_loss, t_pr, t_roc, t_acc, v_loss, v_pr, v_roc, v_acc, start_time, label_name):
    """Log training progress for an epoch"""
    logger.info(f"Epoch {epoch}/{Config.EPOCHS}")
    logger.info(
        f"  Train Loss: {t_loss:.4f}   "
        f"PR-AUC ({label_name}): {t_pr:.3f}   "
        f"ROC-AUC ({label_name}): {t_roc:.3f}   "
        f"Acc ({label_name}): {t_acc:.3f}"
    )
    logger.info(
        f"  Val   Loss: {v_loss:.4f}   "
        f"PR-AUC ({label_name}): {v_pr:.3f}   "
        f"ROC-AUC ({label_name}): {v_roc:.3f}   "
        f"Acc ({label_name}): {v_acc:.3f}"
    )
    elapsed = time.time() - start_time
    logger.info(f"Runtime per epoch: {elapsed:.2f}s")
    logger.info("-" * 80)

def main():
    """Main execution function"""
    # Setup
    os.environ["DEBUG"] = Config.DEBUG
    check_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    check_gpu()
    
    # Initialize wandb
    run = init_wandb(Config)
    
    try:
        # Load data
        data_dict = load_data()
        train_loader, val_loader, test_loader = create_data_loaders(data_dict)
        
        # Create model, optimizer, and scheduler
        model, optimizer, scheduler = create_model_and_optimizer(device, train_loader)
        
        # Train model
        history = train(
            model, train_loader, val_loader, optimizer, scheduler,
            nn.CrossEntropyLoss(), device, data_dict['edges']
        )
        
    finally:
        # Ensure wandb run is finished
        wandb.finish()

if __name__ == "__main__":
    main()