"""
02_build_graph.py
Build a PyG HeteroData object from DECAGON files + KEGG pathways.

Node types:
    drug        — 645 nodes, features: Morgan fingerprint (2048-dim)
    protein     — 19081 nodes, features: one-hot degree (or identity, upgraded by Node2Vec)
    mono_se     — 10184 nodes, features: learned embedding (no input features, random init)

Edge types (all stored bidirectionally for message passing):
    (drug,    targets,     protein)   — drug-protein binding
    (protein, targeted_by, drug)      — reverse
    (protein, interacts,   protein)   — PPI
    (drug,    has_se,      mono_se)   — individual side effect
    (mono_se, se_of,       drug)      — reverse

Combo edges (drug-drug) are NOT added to the graph — they are the prediction target.
They are returned separately as edge_index + edge_label tensors in splits.pt.

Output:
    data/processed/graph.pt         — HeteroData object
    data/processed/meta.json        — index mappings (drug_id -> node_idx, etc.)
"""

import csv
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW        = Path("data/raw")
PROCESSED  = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

MONO_FILE    = RAW / "bio-decagon-mono.csv"
PPI_FILE     = RAW / "bio-decagon-ppi.csv"
TARGET_FILE  = RAW / "bio-decagon-targets.csv"
COMBO_FILE   = RAW / "bio-decagon-combo.csv"
KEGG_FILE    = Path("data/kegg_pathways.json")

OUT_GRAPH    = PROCESSED / "graph.pt"
OUT_META     = PROCESSED / "meta.json"
OUT_COMBO    = PROCESSED / "combo_edges.pt"   # saved separately for split builder

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_COMBO_SE_COUNT = 500   # only keep SE types appearing in >= 500 drug pairs
MORGAN_RADIUS      = 2
MORGAN_NBITS       = 2048


# ── 1. Load raw data ───────────────────────────────────────────────────────────

def load_mono():
    """Returns drug_to_mono_ses: {stitch_id: set of se_ids}"""
    drug_to_ses = defaultdict(set)
    with open(MONO_FILE) as f:
        for row in csv.DictReader(f):
            drug_to_ses[row["STITCH"]].add(row["Individual Side Effect"])
    return drug_to_ses


def load_ppi():
    """Returns list of (gene1, gene2) string pairs."""
    edges = []
    with open(PPI_FILE) as f:
        for row in csv.DictReader(f):
            edges.append((row["Gene 1"], row["Gene 2"]))
    return edges


def load_targets():
    """Returns {stitch_id: set of gene_ids}"""
    drug_to_proteins = defaultdict(set)
    with open(TARGET_FILE) as f:
        for row in csv.DictReader(f):
            drug_to_proteins[row["STITCH"]].add(row["Gene"])
    return drug_to_proteins


def load_combo(min_count=MIN_COMBO_SE_COUNT):
    """
    Returns:
        combo_edges: list of (stitch1, stitch2, se_id)  — filtered to top SE types
        se_names:    {se_id: name}
        top_se_ids:  list of se_ids with count >= min_count (sorted by frequency desc)
    """
    from collections import Counter
    se_counts = Counter()
    rows = []
    with open(COMBO_FILE) as f:
        for row in csv.DictReader(f):
            se_counts[row["Polypharmacy Side Effect"]] += 1
            rows.append((row["STITCH 1"], row["STITCH 2"],
                         row["Polypharmacy Side Effect"], row["Side Effect Name"]))

    top_se = {se for se, cnt in se_counts.items() if cnt >= min_count}
    print(f"  SE types with >= {min_count} occurrences: {len(top_se)}")

    combo_edges = []
    se_names = {}
    for s1, s2, se, name in rows:
        if se in top_se:
            combo_edges.append((s1, s2, se))
            se_names[se] = name

    return combo_edges, se_names, sorted(top_se, key=lambda s: -se_counts[s])


# ── 2. Build node index mappings ───────────────────────────────────────────────

def build_indices(drug_to_mono, ppi_edges, drug_to_proteins, combo_edges):
    """Build integer index for each node type."""
    all_drugs = set()
    for s1, s2, _ in combo_edges:
        all_drugs.add(s1)
        all_drugs.add(s2)
    all_drugs.update(drug_to_mono.keys())
    all_drugs.update(drug_to_proteins.keys())

    all_proteins = set()
    for g1, g2 in ppi_edges:
        all_proteins.add(g1)
        all_proteins.add(g2)
    for genes in drug_to_proteins.values():
        all_proteins.update(genes)

    all_mono_se = set()
    for ses in drug_to_mono.values():
        all_mono_se.update(ses)

    drug_idx    = {d: i for i, d in enumerate(sorted(all_drugs))}
    protein_idx = {p: i for i, p in enumerate(sorted(all_proteins))}
    mono_se_idx = {s: i for i, s in enumerate(sorted(all_mono_se))}

    print(f"  Drugs: {len(drug_idx)}")
    print(f"  Proteins: {len(protein_idx)}")
    print(f"  Mono SE nodes: {len(mono_se_idx)}")

    return drug_idx, protein_idx, mono_se_idx


# ── 3. Drug features (Morgan fingerprints) ────────────────────────────────────

def stitch_to_cid(stitch_id):
    """CID004485548 -> 4485548 (int)"""
    return int(stitch_id.replace("CID", "").lstrip("0") or "0")


def compute_morgan_fingerprints(drug_idx, nbits=MORGAN_NBITS, radius=MORGAN_RADIUS):
    """
    Compute Morgan fingerprints using RDKit via PubChem CID.
    Falls back to zero vector if CID not resolvable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdMolDescriptors
        rdkit_available = True
    except ImportError:
        print("  Warning: RDKit not available, using random drug features.")
        rdkit_available = False

    n = len(drug_idx)
    feats = np.zeros((n, nbits), dtype=np.float32)

    if not rdkit_available:
        np.random.seed(42)
        feats = np.random.randn(n, nbits).astype(np.float32) * 0.1
        return torch.tensor(feats)

    # Try to get SMILES from PubChem for each CID
    # In practice: pre-download a CID->SMILES table from PubChem for your drug set
    # Here we attempt via rdkit's built-in CID lookup if available
    # Fallback: identity features (node index one-hot is too large; use random proj)
    smiles_cache_path = Path("data/drug_smiles.json")
    if smiles_cache_path.exists():
        with open(smiles_cache_path) as f:
            smiles_cache = json.load(f)
    else:
        print("  No drug SMILES cache found at data/drug_smiles.json")
        print("  Run fetch_smiles.py first, or features will be random init.")
        smiles_cache = {}

    hits = 0
    for drug_id, idx in drug_idx.items():
        smiles = smiles_cache.get(drug_id)
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
                    feats[idx] = np.array(fp, dtype=np.float32)
                    hits += 1
            except Exception:
                pass

    print(f"  Morgan fingerprints computed for {hits}/{n} drugs")
    return torch.tensor(feats)


# ── 4. Protein features (degree in PPI) ───────────────────────────────────────

def compute_protein_features(protein_idx, ppi_edges):
    """
    Simple degree-based initial features.
    Will be refined by Node2Vec pre-training (optional, separate script).
    """
    from collections import Counter
    degree = Counter()
    for g1, g2 in ppi_edges:
        if g1 in protein_idx:
            degree[g1] += 1
        if g2 in protein_idx:
            degree[g2] += 1

    n = len(protein_idx)
    # Log-degree as a scalar, expanded to a small feature vector via random proj
    # A proper init would use Node2Vec; this is a placeholder.
    feats = np.zeros((n, 64), dtype=np.float32)
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((1, 64)).astype(np.float32)
    for prot, idx in protein_idx.items():
        log_deg = np.log1p(degree.get(prot, 0))
        feats[idx] = log_deg * proj[0]

    print(f"  Protein features: degree-based 64-dim (replace with Node2Vec embeddings)")
    return torch.tensor(feats)


# ── 5. Build edge tensors ──────────────────────────────────────────────────────

def build_ppi_edges(ppi_edges, protein_idx):
    src, dst = [], []
    for g1, g2 in ppi_edges:
        if g1 in protein_idx and g2 in protein_idx:
            src.append(protein_idx[g1])
            dst.append(protein_idx[g2])
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    print(f"  PPI edges (bidirectional): {edge_index.shape[1]}")
    return edge_index


def build_drug_protein_edges(drug_to_proteins, drug_idx, protein_idx):
    src, dst = [], []
    for drug, proteins in drug_to_proteins.items():
        if drug not in drug_idx:
            continue
        for prot in proteins:
            if prot in protein_idx:
                src.append(drug_idx[drug])
                dst.append(protein_idx[prot])
    fwd = torch.tensor([src, dst], dtype=torch.long)
    rev = torch.tensor([dst, src], dtype=torch.long)
    print(f"  Drug-protein edges: {fwd.shape[1]} forward + {rev.shape[1]} reverse")
    return fwd, rev


def build_drug_mono_se_edges(drug_to_mono, drug_idx, mono_se_idx):
    src, dst = [], []
    for drug, ses in drug_to_mono.items():
        if drug not in drug_idx:
            continue
        for se in ses:
            if se in mono_se_idx:
                src.append(drug_idx[drug])
                dst.append(mono_se_idx[se])
    fwd = torch.tensor([src, dst], dtype=torch.long)
    rev = torch.tensor([dst, src], dtype=torch.long)
    print(f"  Drug-mono SE edges: {fwd.shape[1]} forward + {rev.shape[1]} reverse")
    return fwd, rev


# ── 6. Build combo edge tensors (prediction targets) ──────────────────────────

def build_combo_tensors(combo_edges, drug_idx, top_se_ids):
    """
    Returns:
        edge_index: [2, E] — drug pair indices
        edge_label: [E, num_se] — multi-hot binary labels
        se_to_col:  {se_id: column_index}
    """
    se_to_col = {se: i for i, se in enumerate(top_se_ids)}
    num_se = len(top_se_ids)

    # Group by drug pair
    pair_labels = defaultdict(lambda: np.zeros(num_se, dtype=np.float32))
    for s1, s2, se in combo_edges:
        if s1 not in drug_idx or s2 not in drug_idx:
            continue
        if se not in se_to_col:
            continue
        i, j = drug_idx[s1], drug_idx[s2]
        key = (min(i, j), max(i, j))
        pair_labels[key][se_to_col[se]] = 1.0

    src_list, dst_list, label_list = [], [], []
    for (i, j), labels in pair_labels.items():
        src_list.append(i)
        dst_list.append(j)
        label_list.append(labels)

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_labels = torch.tensor(np.stack(label_list), dtype=torch.float32)
    print(f"  Combo drug pairs: {edge_index.shape[1]}, SE types: {num_se}")
    return edge_index, edge_labels, se_to_col


# ── 7. KEGG pathway membership per drug ───────────────────────────────────────

def build_pathway_data(drug_to_proteins, drug_idx, protein_idx):
    """
    Build a mapping drug_node_idx -> list of (pathway_id, [protein_node_indices])
    Used by the PathwayAttentionPooling module at runtime.
    Saved separately as data/processed/pathway_memberships.pkl
    """
    if not KEGG_FILE.exists():
        print("  KEGG file not found — skipping pathway data. Run 01_fetch_kegg.py first.")
        return None

    with open(KEGG_FILE) as f:
        kegg = json.load(f)

    # Build protein_node_idx -> set of pathway_ids
    gene_to_pathways = defaultdict(list)
    for pw_id, info in kegg.items():
        for gene in info["genes"]:
            gene_to_pathways[gene].append(pw_id)

    # For each drug: collect its target genes, group by pathway
    drug_pathway_map = {}   # drug_node_idx -> {pw_id: [protein_node_indices]}
    covered_drugs = 0
    for drug, proteins in drug_to_proteins.items():
        if drug not in drug_idx:
            continue
        d_idx = drug_idx[drug]
        pw_to_prots = defaultdict(list)
        for prot in proteins:
            if prot not in protein_idx:
                continue
            p_idx = protein_idx[prot]
            for pw in gene_to_pathways.get(prot, []):
                pw_to_prots[pw].append(p_idx)
        if pw_to_prots:
            drug_pathway_map[d_idx] = dict(pw_to_prots)
            covered_drugs += 1

    all_pathways = sorted({pw for pws in drug_pathway_map.values() for pw in pws})
    pathway_id_to_col = {pw: i for i, pw in enumerate(all_pathways)}

    print(f"  Drugs with pathway coverage: {covered_drugs}/{len(drug_idx)}")
    print(f"  Unique pathways: {len(all_pathways)}")

    return {
        "drug_pathway_map": drug_pathway_map,
        "pathway_id_to_col": pathway_id_to_col,
        "pathway_names": {pw: kegg[pw]["name"] for pw in all_pathways if pw in kegg},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Loading raw data...")
    drug_to_mono      = load_mono()
    ppi_edges         = load_ppi()
    drug_to_proteins  = load_targets()

    print("\nLoading combo edges...")
    combo_edges, se_names, top_se_ids = load_combo()

    print("\nBuilding node indices...")
    drug_idx, protein_idx, mono_se_idx = build_indices(
        drug_to_mono, ppi_edges, drug_to_proteins, combo_edges
    )

    print("\nComputing node features...")
    drug_feats    = compute_morgan_fingerprints(drug_idx)
    protein_feats = compute_protein_features(protein_idx, ppi_edges)
    # mono_se nodes: no input features — learned from scratch via embedding table
    # initialised as zero; model will add nn.Embedding for these
    mono_se_feats = torch.zeros(len(mono_se_idx), 64)

    print("\nBuilding edge tensors...")
    ppi_ei                    = build_ppi_edges(ppi_edges, protein_idx)
    dp_fwd, dp_rev            = build_drug_protein_edges(drug_to_proteins, drug_idx, protein_idx)
    dm_fwd, dm_rev            = build_drug_mono_se_edges(drug_to_mono, drug_idx, mono_se_idx)

    print("\nBuilding combo prediction tensors...")
    combo_ei, combo_labels, se_to_col = build_combo_tensors(
        combo_edges, drug_idx, top_se_ids
    )

    print("\nBuilding pathway membership data...")
    pathway_data = build_pathway_data(drug_to_proteins, drug_idx, protein_idx)

    # ── Assemble HeteroData ──────────────────────────────────────────────────
    print("\nAssembling HeteroData graph...")
    data = HeteroData()

    # Node features
    data["drug"].x         = drug_feats          # [N_drug, 2048]
    data["protein"].x      = protein_feats        # [N_prot, 64]
    data["mono_se"].x      = mono_se_feats        # [N_se, 64]  (placeholder, replaced by embedding)

    data["drug"].num_nodes      = len(drug_idx)
    data["protein"].num_nodes   = len(protein_idx)
    data["mono_se"].num_nodes   = len(mono_se_idx)

    # Edges
    data["drug",    "targets",     "protein"].edge_index = dp_fwd
    data["protein", "targeted_by", "drug"   ].edge_index = dp_rev
    data["protein", "interacts",   "protein"].edge_index = ppi_ei
    data["drug",    "has_se",      "mono_se"].edge_index = dm_fwd
    data["mono_se", "se_of",       "drug"   ].edge_index = dm_rev

    print(f"\nHeteroData summary:")
    print(data)

    # ── Save ────────────────────────────────────────────────────────────────
    torch.save(data, OUT_GRAPH)
    print(f"\nSaved graph to {OUT_GRAPH}")

    torch.save({
        "edge_index":  combo_ei,
        "edge_labels": combo_labels,
        "se_to_col":   se_to_col,
        "top_se_ids":  top_se_ids,
        "se_names":    se_names,
    }, OUT_COMBO)
    print(f"Saved combo edges to {OUT_COMBO}")

    # Save index mappings as JSON for inspection
    meta = {
        "drug_idx":    drug_idx,
        "protein_idx": protein_idx,
        "mono_se_idx": mono_se_idx,
        "se_to_col":   se_to_col,
        "se_names":    se_names,
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f)
    print(f"Saved meta to {OUT_META}")

    if pathway_data is not None:
        import pickle
        with open(PROCESSED / "pathway_memberships.pkl", "wb") as f:
            pickle.dump(pathway_data, f)
        print(f"Saved pathway data to {PROCESSED}/pathway_memberships.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()
