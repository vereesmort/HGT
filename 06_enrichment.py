"""
06_enrichment.py
Validate biological coherence of learned drug embeddings.

Steps:
    1. Load trained drug embeddings z_drug [N_drugs, hidden_dim]
    2. Cluster drugs in embedding space (k-means, k=10..20)
    3. For each cluster: collect target gene sets of member drugs
    4. Run GO term enrichment and KEGG pathway enrichment via g:Profiler
    5. Report: are clusters enriched for specific biological functions?

Outputs:
    results/drug_clusters.csv          — drug_id, cluster_id, drug_name
    results/enrichment_per_cluster/    — one CSV per cluster with enriched terms
    results/enrichment_summary.json    — # significant terms per cluster, top terms
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from model import PolypharmacyHGT

PROCESSED = Path("data/processed")
CHECKPOINTS = Path("checkpoints")
RESULTS = Path("results")
ENRICHMENT_DIR = RESULTS / "enrichment_per_cluster"
ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 15
SEED       = 42


def load_model_and_embeddings(data, drug_pathway_map, num_se, num_pathways, device):
    ckpt = torch.load(CHECKPOINTS / "best_model.pt", map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    in_dims = {
        "drug":    data["drug"].x.shape[1],
        "protein": data["protein"].x.shape[1],
        "mono_se": cfg["hidden_dim"],
        "_mono_se_count": data["mono_se"].num_nodes,
    }
    from model import PolypharmacyHGT
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

    print("Computing drug embeddings...")
    z_drug = model.get_drug_embeddings(data, drug_pathway_map, device)
    return z_drug.cpu().numpy()


def cluster_drugs(z_drug, n_clusters=N_CLUSTERS):
    print(f"K-means clustering into {n_clusters} clusters...")
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    labels = km.fit_predict(z_drug)
    return labels, km


def get_cluster_gene_sets(cluster_labels, drug_idx, drug_to_proteins, n_clusters):
    """
    For each cluster, collect the union of target gene sets of member drugs.
    Returns: {cluster_id: set of gene_ids (Entrez)}
    """
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    cluster_genes = {c: set() for c in range(n_clusters)}

    for node_idx, cluster_id in enumerate(cluster_labels):
        drug_id = idx_to_drug.get(node_idx)
        if drug_id and drug_id in drug_to_proteins:
            cluster_genes[cluster_id].update(drug_to_proteins[drug_id])

    return cluster_genes


def run_gprofiler_enrichment(gene_set, organism="hsapiens"):
    """
    Run GO and KEGG pathway enrichment on a set of Entrez gene IDs via g:Profiler.
    Returns a list of enriched terms (dicts).
    """
    try:
        from gprofiler import GProfiler
    except ImportError:
        print("  gprofiler-official not installed. Skipping enrichment.")
        return []

    if len(gene_set) < 3:
        return []

    gp = GProfiler(return_dataframe=False)
    try:
        results = gp.profile(
            organism=organism,
            query=list(gene_set),
            sources=["GO:BP", "GO:MF", "KEGG", "REAC"],
            significance_threshold_method="fdr",
            user_threshold=0.05,
            no_evidences=True,
        )
        return results
    except Exception as e:
        print(f"  g:Profiler error: {e}")
        return []


def main():
    device = torch.device("cpu")  # embeddings extraction on CPU is fine

    print("Loading data...")
    data   = torch.load(PROCESSED / "graph.pt", weights_only=False)
    combo  = torch.load(PROCESSED / "combo_edges.pt", weights_only=False)
    num_se = len(combo["top_se_ids"])

    with open(PROCESSED / "pathway_memberships.pkl", "rb") as f:
        pathway_data = pickle.load(f)
    drug_pathway_map = pathway_data["drug_pathway_map"]
    num_pathways     = len(pathway_data["pathway_id_to_col"])

    with open(PROCESSED / "meta.json") as f:
        meta = json.load(f)
    drug_idx = meta["drug_idx"]

    # Load targets for gene set construction
    import csv
    drug_to_proteins = {}
    with open("data/raw/bio-decagon-targets.csv") as f:
        for row in csv.DictReader(f):
            d = row["STITCH"]
            if d not in drug_to_proteins:
                drug_to_proteins[d] = set()
            drug_to_proteins[d].add(row["Gene"])

    # ── Get embeddings ───────────────────────────────────────────────────────
    z_drug = load_model_and_embeddings(
        data, drug_pathway_map, num_se, num_pathways, device
    )
    print(f"Drug embeddings shape: {z_drug.shape}")

    # Save embeddings for further analysis
    np.save(RESULTS / "drug_embeddings.npy", z_drug)
    print(f"Saved embeddings to {RESULTS / 'drug_embeddings.npy'}")

    # ── Cluster ──────────────────────────────────────────────────────────────
    cluster_labels, km = cluster_drugs(z_drug, N_CLUSTERS)

    # Save cluster assignments
    idx_to_drug = {v: k for k, v in drug_idx.items()}
    import csv as csv_mod
    with open(RESULTS / "drug_clusters.csv", "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["drug_id", "node_idx", "cluster_id"])
        for node_idx, cluster_id in enumerate(cluster_labels):
            drug_id = idx_to_drug.get(node_idx, "unknown")
            writer.writerow([drug_id, node_idx, int(cluster_id)])
    print(f"Saved cluster assignments to {RESULTS / 'drug_clusters.csv'}")

    # ── Enrichment per cluster ───────────────────────────────────────────────
    cluster_genes = get_cluster_gene_sets(
        cluster_labels, drug_idx, drug_to_proteins, N_CLUSTERS
    )

    summary = {}
    for cluster_id in range(N_CLUSTERS):
        genes = cluster_genes[cluster_id]
        n_drugs = int((cluster_labels == cluster_id).sum())
        print(f"\nCluster {cluster_id}: {n_drugs} drugs, {len(genes)} unique target genes")

        if len(genes) < 3:
            print("  Too few genes for enrichment, skipping.")
            summary[cluster_id] = {"n_drugs": n_drugs, "n_genes": len(genes), "n_significant": 0}
            continue

        enrichment = run_gprofiler_enrichment(genes)

        if enrichment:
            # Save per-cluster results
            cluster_file = ENRICHMENT_DIR / f"cluster_{cluster_id:02d}.csv"
            with open(cluster_file, "w", newline="") as f:
                if enrichment:
                    fields = list(enrichment[0].keys())
                    writer = csv_mod.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(enrichment)

            # Top 5 terms for summary
            top_terms = [
                {"source": r.get("source"), "name": r.get("name"), "p_value": r.get("p_value")}
                for r in sorted(enrichment, key=lambda x: x.get("p_value", 1))[:5]
            ]
            summary[cluster_id] = {
                "n_drugs": n_drugs,
                "n_genes": len(genes),
                "n_significant": len(enrichment),
                "top_terms": top_terms,
            }
            print(f"  Significant terms: {len(enrichment)}")
            for t in top_terms[:3]:
                print(f"    [{t['source']}] {t['name']}  p={t['p_value']:.2e}")
        else:
            summary[cluster_id] = {
                "n_drugs": n_drugs, "n_genes": len(genes), "n_significant": 0
            }

    with open(RESULTS / "enrichment_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved enrichment summary to {RESULTS / 'enrichment_summary.json'}")

    # ── PCA visualisation data (for plotting in thesis) ──────────────────────
    pca = PCA(n_components=2, random_state=SEED)
    z_2d = pca.fit_transform(z_drug)
    np.save(RESULTS / "drug_embeddings_pca2d.npy", z_2d)
    np.save(RESULTS / "cluster_labels.npy", cluster_labels)
    print(f"Saved PCA 2D projections and cluster labels for plotting.")
    print(f"Variance explained by 2 PCs: {pca.explained_variance_ratio_.sum():.3f}")


if __name__ == "__main__":
    main()
