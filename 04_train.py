"""
04_train.py
Train the PolypharmacyHGT model.

Loss: Binary cross-entropy over (drug_pair, SE_type) predictions.
      For each positive drug pair, we score all num_se types and compute BCE
      against the multi-hot label vector.
      Negative pairs get an all-zero label vector.

Optimiser: Adam with cosine LR decay.
Checkpoints saved to checkpoints/best_model.pt
"""

import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import PolypharmacyHGT

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED    = Path("data/processed")
CHECKPOINTS  = Path("checkpoints")
CHECKPOINTS.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIM  = 64
NUM_HEADS   = 4
NUM_LAYERS  = 2
DROPOUT     = 0.1
LR          = 1e-3
WEIGHT_DECAY= 1e-5
EPOCHS      = 100
BATCH_SIZE  = 512   # drug pairs per batch
PATIENCE    = 10    # early stopping on val loss
SEED        = 42
TOP_N_SE = 30  # set to None to use all 964

def make_pair_dataset(pos_ei, neg_ei, labels):
    """
    Combine positive and negative drug pairs into a single dataset.
    Positives get their multi-hot label; negatives get all-zero labels.
    Returns TensorDataset of (src, dst, label_vector).
    """
    num_se = labels.shape[1]
    neg_labels = torch.zeros(neg_ei.shape[1], num_se)

    all_src = torch.cat([pos_ei[0], neg_ei[0]])
    all_dst = torch.cat([pos_ei[1], neg_ei[1]])
    all_lbl = torch.cat([labels, neg_labels])

    return TensorDataset(all_src, all_dst, all_lbl)


def compute_loss(scores, labels, pos_weight=None):
    """
    BCE loss over all (pair, SE) entries.
    pos_weight: tensor of shape [num_se] for class imbalance.
    """
    if pos_weight is not None:
        loss_fn = nn.BCELoss(reduction="mean")
        # Weight positive entries
        w = pos_weight.unsqueeze(0).expand_as(labels)
        weights = torch.where(labels > 0.5, w, torch.ones_like(w))
        return (weights * nn.functional.binary_cross_entropy(scores, labels, reduction="none")).mean()
    return nn.functional.binary_cross_entropy(scores, labels)


def compute_pos_weight(train_labels):
    """Compute per-SE positive class weight for imbalanced BCE."""
    n_pos = train_labels.sum(0).clamp(min=1)
    n_neg = (train_labels.shape[0] - n_pos).clamp(min=1)
    return (n_neg / n_pos).clamp(max=50)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load data ───────────────────────────────────────────────────────────
    print("Loading graph and splits...")
    data   = torch.load(PROCESSED / "graph.pt", weights_only=False).to(device)
    splits = torch.load(PROCESSED / "splits.pt", weights_only=False)
    combo  = torch.load(PROCESSED / "combo_edges.pt", weights_only=False)
    top_se_ids = combo["top_se_ids"]
    num_se     = len(top_se_ids)

    # ── Subset to first N side effects ─────────────────────────────────────
    if TOP_N_SE is not None:
        top_se_ids = top_se_ids[:TOP_N_SE]
        num_se     = TOP_N_SE
        # Slice edge_labels to only the selected SE columns
        combo["edge_labels"] = combo["edge_labels"][:, :TOP_N_SE]
        splits["train"]["edge_labels"] = splits["train"]["edge_labels"][:, :TOP_N_SE]
        splits["val"]["edge_labels"]   = splits["val"]["edge_labels"][:, :TOP_N_SE]
        splits["test"]["edge_labels"]  = splits["test"]["edge_labels"][:, :TOP_N_SE]

    with open(PROCESSED / "pathway_memberships.pkl", "rb") as f:
        pathway_data = pickle.load(f)
    drug_pathway_map = pathway_data["drug_pathway_map"]
    num_pathways     = len(pathway_data["pathway_id_to_col"])

    with open(PROCESSED / "meta.json") as f:
        meta = json.load(f)

    # ── Build datasets ───────────────────────────────────────────────────────
    train_ds = make_pair_dataset(
        splits["train"]["pos_edge_index"],
        splits["train"]["neg_edge_index"],
        splits["train"]["edge_labels"],
    )
    val_ds = make_pair_dataset(
        splits["val"]["pos_edge_index"],
        splits["val"]["neg_edge_index"],
        splits["val"]["edge_labels"],
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    pos_weight = compute_pos_weight(splits["train"]["edge_labels"]).to(device)
    print(f"Positive weight range: {pos_weight.min():.1f} – {pos_weight.max():.1f}")

    # ── Build model ──────────────────────────────────────────────────────────
    in_dims = {
        "drug":    data["drug"].x.shape[1],
        "protein": data["protein"].x.shape[1],
        "mono_se": HIDDEN_DIM,               # embedding output
        "_mono_se_count": data["mono_se"].num_nodes,
    }

    model = PolypharmacyHGT(
        in_dims       = in_dims,
        hidden_dim    = HIDDEN_DIM,
        num_heads     = NUM_HEADS,
        num_layers    = NUM_LAYERS,
        num_se        = num_se,
        num_pathways  = num_pathways,
        graph_metadata= data.metadata(),
        dropout       = DROPOUT,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for src, dst, lbl in tqdm(train_loader, desc=f"Epoch {epoch:3d} train", leave=False):
            src, dst, lbl = src.to(device), dst.to(device), lbl.to(device)
            pair_index = torch.stack([src, dst])
            optimiser.zero_grad()
            scores = model(data, pair_index, drug_pathway_map, device)
            loss = compute_loss(scores, lbl, pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item() * src.shape[0]

        train_loss = total_loss / len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, dst, lbl in val_loader:
                src, dst, lbl = src.to(device), dst.to(device), lbl.to(device)
                pair_index = torch.stack([src, dst])
                scores = model(data, pair_index, drug_pathway_map, device)
                loss = compute_loss(scores, lbl, pos_weight)
                val_loss += loss.item() * src.shape[0]
        val_loss /= len(val_ds)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        # ── Checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_loss":    val_loss,
                "config": {
                    "hidden_dim": HIDDEN_DIM,
                    "num_heads":  NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "num_se":     num_se,
                    "num_pathways": num_pathways,
                }
            }, CHECKPOINTS / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    torch.save(history, CHECKPOINTS / "history.pt")
    print("Training complete.")


if __name__ == "__main__":
    main()
