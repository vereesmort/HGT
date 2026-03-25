"""
05_evaluate.py
Evaluate trained model on the test split.

Metrics (following Zitnik et al. 2018 for comparability):
    - AUROC per SE type, then macro-average
    - AUPRC per SE type, then macro-average
    - AUROC/AUPRC at top-10, top-50, top-964 SE types by frequency

Outputs:
    results/metrics_per_se.csv   — per-SE type AUROC and AUPRC
    results/metrics_summary.json — macro averages + frequency-binned averages
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from model import PolypharmacyHGT

PROCESSED   = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS     = Path("results")
RESULTS.mkdir(exist_ok=True)

BATCH_SIZE  = 512
HIDDEN_DIM  = 64
NUM_HEADS   = 4
NUM_LAYERS  = 2


def load_model(checkpoint_path, data, num_se, num_pathways, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    in_dims = {
        "drug":    data["drug"].x.shape[1],
        "protein": data["protein"].x.shape[1],
        "mono_se": cfg["hidden_dim"],
        "_mono_se_count": data["mono_se"].num_nodes,
    }
    model = PolypharmacyHGT(
        in_dims       = in_dims,
        hidden_dim    = cfg["hidden_dim"],
        num_heads     = cfg["num_heads"],
        num_layers    = cfg["num_layers"],
        num_se        = cfg["num_se"],
        num_pathways  = cfg["num_pathways"],
        graph_metadata= data.metadata(),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def collect_scores_and_labels(model, data, splits, drug_pathway_map, device):
    """
    Run model on all test pairs (pos + neg) and collect scores + labels.
    Returns:
        all_scores: [N_pairs, num_se]
        all_labels: [N_pairs, num_se]
    """
    from torch.utils.data import DataLoader, TensorDataset

    test = splits["test"]
    num_se = test["edge_labels"].shape[1]
    neg_labels = torch.zeros(test["neg_edge_index"].shape[1], num_se)

    all_src = torch.cat([test["pos_edge_index"][0], test["neg_edge_index"][0]])
    all_dst = torch.cat([test["pos_edge_index"][1], test["neg_edge_index"][1]])
    all_lbl = torch.cat([test["edge_labels"], neg_labels])

    ds = TensorDataset(all_src, all_dst, all_lbl)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    score_list, label_list = [], []
    with torch.no_grad():
        for src, dst, lbl in tqdm(loader, desc="Scoring test pairs"):
            src, dst = src.to(device), dst.to(device)
            pair_index = torch.stack([src, dst])
            scores = model(data, pair_index, drug_pathway_map, device)
            score_list.append(scores.cpu())
            label_list.append(lbl)

    return torch.cat(score_list).numpy(), torch.cat(label_list).numpy()


def compute_metrics(scores, labels, top_se_ids, se_names):
    """Compute AUROC and AUPRC per SE type."""
    num_se = scores.shape[1]
    results = []

    for r in range(num_se):
        y_true = labels[:, r]
        y_score = scores[:, r]

        # Skip SE types with no positives in test split
        if y_true.sum() == 0:
            continue

        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        results.append({
            "se_id":   top_se_ids[r],
            "se_name": se_names.get(top_se_ids[r], ""),
            "n_pos":   int(y_true.sum()),
            "auroc":   auroc,
            "auprc":   auprc,
        })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data   = torch.load(PROCESSED / "graph.pt", weights_only=False).to(device)
    splits = torch.load(PROCESSED / "splits.pt", weights_only=False)
    combo  = torch.load(PROCESSED / "combo_edges.pt", weights_only=False)

    with open(PROCESSED / "pathway_memberships.pkl", "rb") as f:
        pathway_data = pickle.load(f)
    drug_pathway_map = pathway_data["drug_pathway_map"]
    num_pathways     = len(pathway_data["pathway_id_to_col"])

    top_se_ids = combo["top_se_ids"]
    se_names   = combo["se_names"]
    num_se     = len(top_se_ids)

    model = load_model(
        CHECKPOINTS / "best_model.pt",
        data, num_se, num_pathways, device
    )

    print("Collecting scores on test split...")
    scores, labels = collect_scores_and_labels(
        model, data, splits, drug_pathway_map, device
    )

    print("Computing metrics...")
    per_se = compute_metrics(scores, labels, top_se_ids, se_names)

    # ── Save per-SE results ─────────────────────────────────────────────────
    import csv
    with open(RESULTS / "metrics_per_se.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["se_id", "se_name", "n_pos", "auroc", "auprc"])
        writer.writeheader()
        writer.writerows(per_se)
    print(f"Saved per-SE metrics to {RESULTS / 'metrics_per_se.csv'}")

    # ── Summary statistics ──────────────────────────────────────────────────
    aurocs = [r["auroc"] for r in per_se]
    auprcs = [r["auprc"] for r in per_se]

    # Frequency-binned averages (following Zitnik et al. convention)
    per_se_sorted = sorted(per_se, key=lambda x: -x["n_pos"])
    def bin_avg(lst, metric, n):
        top = lst[:n]
        return np.mean([r[metric] for r in top]) if top else 0.0

    summary = {
        "n_se_evaluated":     len(per_se),
        "macro_auroc":        float(np.mean(aurocs)),
        "macro_auprc":        float(np.mean(auprcs)),
        "median_auroc":       float(np.median(aurocs)),
        "median_auprc":       float(np.median(auprcs)),
        "top10_auroc":        bin_avg(per_se_sorted, "auroc", 10),
        "top10_auprc":        bin_avg(per_se_sorted, "auprc", 10),
        "top50_auroc":        bin_avg(per_se_sorted, "auroc", 50),
        "top50_auprc":        bin_avg(per_se_sorted, "auprc", 50),
    }

    with open(RESULTS / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Evaluation Summary ──────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")


if __name__ == "__main__":
    main()
