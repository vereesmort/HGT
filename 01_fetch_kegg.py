"""
01_fetch_kegg.py
Fetch KEGG human pathway → Entrez gene mappings and save to data/kegg_pathways.json.
Requires network access. Run once; output is cached.

Output:
    data/kegg_pathways.json  — dict mapping pathway_id -> {name, genes: [entrez_id, ...]}
    data/gene_to_pathways.json — dict mapping entrez_id -> [pathway_id, ...]
"""

import json
import os
import time
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUT_PATHWAYS = DATA_DIR / "kegg_pathways.json"
OUT_GENE2PW   = DATA_DIR / "gene_to_pathways.json"


def fetch_kegg_pathways():
    from bioservices import KEGG
    k = KEGG()
    k.organism = "hsa"  # human

    print("Fetching list of human KEGG pathways...")
    pathway_list = k.pathwayIds  # e.g. ['path:hsa00010', ...]
    print(f"  Found {len(pathway_list)} pathways")

    pathways = {}
    for i, pw_id in enumerate(pathway_list):
        short_id = pw_id.replace("path:", "")
        try:
            result = k.get(short_id)
            parsed = k.parse(result)
            genes_raw = parsed.get("GENE", {})
            # genes_raw is dict: {entrez_id: "SYMBOL; description", ...}
            gene_ids = list(genes_raw.keys())
            name = parsed.get("NAME", short_id)
            if isinstance(name, list):
                name = name[0]
            pathways[short_id] = {"name": name.strip(), "genes": gene_ids}
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(pathway_list)} pathways...")
            time.sleep(0.2)  # be polite to KEGG API
        except Exception as e:
            print(f"  Warning: failed to fetch {short_id}: {e}")
            continue

    print(f"Successfully fetched {len(pathways)} pathways")
    return pathways


def build_gene_to_pathways(pathways):
    gene_to_pw = {}
    for pw_id, info in pathways.items():
        for gene in info["genes"]:
            if gene not in gene_to_pw:
                gene_to_pw[gene] = []
            gene_to_pw[gene].append(pw_id)
    return gene_to_pw


def main():
    if OUT_PATHWAYS.exists():
        print(f"Cache found at {OUT_PATHWAYS}, skipping fetch.")
        print("Delete the file to re-fetch.")
        return

    pathways = fetch_kegg_pathways()

    with open(OUT_PATHWAYS, "w") as f:
        json.dump(pathways, f, indent=2)
    print(f"Saved pathway data to {OUT_PATHWAYS}")

    gene_to_pw = build_gene_to_pathways(pathways)
    with open(OUT_GENE2PW, "w") as f:
        json.dump(gene_to_pw, f, indent=2)
    print(f"Saved gene->pathway map to {OUT_GENE2PW}")

    # Stats
    covered = sum(1 for pw in pathways.values() if len(pw["genes"]) > 0)
    all_genes = set(g for pw in pathways.values() for g in pw["genes"])
    print(f"\nStats:")
    print(f"  Pathways with at least 1 gene: {covered}")
    print(f"  Unique genes across all pathways: {len(all_genes)}")
    print(f"  Genes with pathway annotation: {len(gene_to_pw)}")


if __name__ == "__main__":
    main()
