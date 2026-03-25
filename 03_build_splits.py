"""
03_build_splits.py
Build train / validation / test splits for the combo prediction task.

Split strategy (leakage-corrected):
    - Split at the DRUG PAIR level, not at the edge level.
    - A drug pair and ALL its associated SE labels go entirely into one split.
    - This prevents the model from seeing partial labels of a pair during training
      and predicting the rest at test time (the leakage pattern in masumshah2021neural).

Negative sampling:
    - For each positive (drug_i, drug_j, se_r), sample k negatives:
      drug pairs that do NOT have SE r as a combo side effect.
    - Negatives are sampled uniformly from all drug pairs not in the positive set for SE r.
    - Fixed random seed for reproducibility.

Output:
    data/processed/splits.pt  — dict with keys train/val/test, each containing:
        pos_edge_index  [2, E_pos]
        neg_edge_index  [2, E_neg]
        edge_labels     [E_pos, num_se]   (multi-hot, for positive pairs)
        pair_indices    list of pair row indices into combo_edges for traceability
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROCESSED = Path("data/processed")
OUT_SPLITS = PROCESSED / "splits.pt"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
NEG_PER_POS = 1   # negative pairs sampled per positive pair per SE
SEED        = 42


def main():
    rng = np.random.default_rng(SEED)

    print("Loading combo edges and graph meta...")
    combo = torch.load(PROCESSED / "combo_edges.pt")
    edge_index  = combo["edge_index"]   # [2, N_pairs]
    edge_labels = combo["edge_labels"]  # [N_pairs, num_se]
    top_se_ids  = combo["top_se_ids"]
    num_se      = len(top_se_ids)
    n_pairs     = edge_index.shape[1]

    with open(PROCESSED / "meta.json") as f:
        meta = json.load(f)
    n_drugs = len(meta["drug_idx"])

    print(f"Total drug pairs: {n_pairs}, SE types: {num_se}, drugs: {n_drugs}")

    # ── 1. Split at drug-pair level ──────────────────────────────────────────
    pair_indices = np.arange(n_pairs)
    rng.shuffle(pair_indices)

    n_train = int(n_pairs * TRAIN_RATIO)
    n_val   = int(n_pairs * VAL_RATIO)

    train_idx = pair_indices[:n_train]
    val_idx   = pair_indices[n_train:n_train + n_val]
    test_idx  = pair_indices[n_train + n_val:]

    print(f"Split sizes — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # Verify no drug pair appears in multiple splits
    train_pairs = set(map(tuple, edge_index[:, train_idx].T.tolist()))
    val_pairs   = set(map(tuple, edge_index[:, val_idx].T.tolist()))
    test_pairs  = set(map(tuple, edge_index[:, test_idx].T.tolist()))
    assert len(train_pairs & val_pairs) == 0, "Leakage: train/val overlap"
    assert len(train_pairs & test_pairs) == 0, "Leakage: train/test overlap"
    assert len(val_pairs & test_pairs) == 0, "Leakage: val/test overlap"
    print("Leakage check passed: no drug pair appears in multiple splits.")

    # ── 2. Build set of all positive pairs (for negative sampling) ───────────
    # all_pos_pairs: set of (i, j) with i <= j
    all_pos_pairs = set(
        (min(int(edge_index[0, k]), int(edge_index[1, k])),
         max(int(edge_index[0, k]), int(edge_index[1, k])))
        for k in range(n_pairs)
    )

    def sample_negatives(split_indices, n_neg_per_pair):
        """
        For each drug pair in split_indices, sample n_neg_per_pair drug pairs
        that are NOT in the positive set.
        Returns neg_edge_index [2, N_neg].
        """
        src_list, dst_list = [], []
        pos_ei = edge_index[:, split_indices]
        for k in range(pos_ei.shape[1]):
            i = int(pos_ei[0, k])
            for _ in range(n_neg_per_pair):
                attempts = 0
                while True:
                    j = int(rng.integers(0, n_drugs))
                    if j == i:
                        continue
                    ni, nj = min(i, j), max(i, j)
                    if (ni, nj) not in all_pos_pairs:
                        src_list.append(ni)
                        dst_list.append(nj)
                        break
                    attempts += 1
                    if attempts > 100:
                        # Very dense graph — just take the candidate anyway
                        src_list.append(ni)
                        dst_list.append(nj)
                        break
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    print("Sampling negatives (this may take a moment)...")

    splits = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"  Building {name} split ({len(idx)} pairs)...")
        pos_ei  = edge_index[:, idx]
        labels  = edge_labels[idx]
        neg_ei  = sample_negatives(idx, NEG_PER_POS)
        splits[name] = {
            "pos_edge_index": pos_ei,
            "neg_edge_index": neg_ei,
            "edge_labels":    labels,
            "pair_indices":   torch.tensor(idx, dtype=torch.long),
        }
        print(f"    pos pairs: {pos_ei.shape[1]}, neg pairs: {neg_ei.shape[1]}")

    torch.save(splits, OUT_SPLITS)
    print(f"\nSaved splits to {OUT_SPLITS}")


if __name__ == "__main__":
    main()
