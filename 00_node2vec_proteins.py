"""
00_node2vec_proteins.py  (optional, run before 02_build_graph.py)
Pre-train Node2Vec embeddings on the PPI graph to initialise protein node features.
These replace the simple degree-based features in 02_build_graph.py.

Saves: data/protein_node2vec.npy  — [N_proteins, 128] float32 array,
       indexed in the same order as protein_idx from meta.json.

Note: run 02_build_graph.py first to get meta.json, then run this,
then re-run 02_build_graph.py — it will auto-detect and load the embeddings.
"""

import json
import csv
from pathlib import Path
import numpy as np
import torch

DATA_DIR  = Path("data")
RAW       = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
OUT_EMBED = DATA_DIR / "protein_node2vec.npy"

EMBED_DIM   = 128
WALK_LENGTH = 20
CONTEXT     = 10
WALKS_PER   = 10
P, Q        = 1.0, 1.0   # BFS-like (P=Q=1); set Q<1 for inward bias (community)
EPOCHS      = 5
BATCH_SIZE  = 128
LR          = 0.01
SEED        = 42


def main():
    # Requires meta.json from 02_build_graph.py
    if not (PROCESSED / "meta.json").exists():
        print("Run 02_build_graph.py first to generate meta.json.")
        return

    with open(PROCESSED / "meta.json") as f:
        meta = json.load(f)
    protein_idx = meta["protein_idx"]
    n_proteins  = len(protein_idx)

    print(f"Proteins: {n_proteins}")

    # Build edge_index for PPI
    src_list, dst_list = [], []
    with open(RAW / "bio-decagon-ppi.csv") as f:
        for row in csv.DictReader(f):
            g1, g2 = row["Gene 1"], row["Gene 2"]
            if g1 in protein_idx and g2 in protein_idx:
                src_list.append(protein_idx[g1])
                dst_list.append(protein_idx[g2])

    edge_index = torch.tensor([src_list + dst_list, dst_list + src_list], dtype=torch.long)
    print(f"PPI edges (bidirectional): {edge_index.shape[1]}")

    # Node2Vec via PyG
    from torch_geometric.nn import Node2Vec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Node2Vec(
        edge_index,
        embedding_dim = EMBED_DIM,
        walk_length   = WALK_LENGTH,
        context_size  = CONTEXT,
        walks_per_node= WALKS_PER,
        p             = P,
        q             = Q,
        num_nodes     = n_proteins,
        sparse        = True,
    ).to(device)

    loader = model.loader(batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    optimiser = torch.optim.SparseAdam(list(model.parameters()), lr=LR)

    def train_epoch():
        model.train()
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimiser.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    print("Training Node2Vec...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch()
        print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}")

    embeddings = model.embedding.weight.data.cpu().numpy()
    np.save(OUT_EMBED, embeddings)
    print(f"Saved Node2Vec embeddings to {OUT_EMBED} — shape {embeddings.shape}")
    print("Re-run 02_build_graph.py to use these as protein node features.")


if __name__ == "__main__":
    main()
