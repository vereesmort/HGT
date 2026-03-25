"""
00b_fetch_smiles.py  (optional, run before 02_build_graph.py)
Fetch SMILES strings for all drugs in the dataset from PubChem using their CID.
Saves data/drug_smiles.json — {stitch_id: smiles_string}

Requires network access. Run once; output is cached.
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

RAW       = Path("data/raw")
OUT_FILE  = Path("data/drug_smiles.json")
COMBO_FILE = RAW / "bio-decagon-combo.csv"
MONO_FILE  = RAW / "bio-decagon-mono.csv"
TARGET_FILE = RAW / "bio-decagon-targets.csv"

SLEEP_BETWEEN = 0.1  # seconds between PubChem requests


def stitch_to_cid(stitch_id):
    """CID004485548 -> 4485548"""
    return int(stitch_id.replace("CID", "").lstrip("0") or "0")


def fetch_smiles_from_pubchem(cid):
    """Fetch canonical SMILES from PubChem REST API."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        return None


def main():
    if OUT_FILE.exists():
        print(f"Cache found at {OUT_FILE}, skipping fetch.")
        return

    # Collect all drug IDs
    all_drugs = set()
    for fpath in [COMBO_FILE, MONO_FILE, TARGET_FILE]:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in reader.fieldnames:
                    if "STITCH" in col.upper():
                        all_drugs.add(row[col])

    print(f"Fetching SMILES for {len(all_drugs)} drugs from PubChem...")

    smiles = {}
    failed = []
    for i, drug_id in enumerate(sorted(all_drugs)):
        cid = stitch_to_cid(drug_id)
        s = fetch_smiles_from_pubchem(cid)
        if s:
            smiles[drug_id] = s
        else:
            failed.append(drug_id)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_drugs)}  ({len(smiles)} fetched, {len(failed)} failed)")
        time.sleep(SLEEP_BETWEEN)

    with open(OUT_FILE, "w") as f:
        json.dump(smiles, f)

    print(f"\nFetched SMILES for {len(smiles)}/{len(all_drugs)} drugs")
    print(f"Failed: {len(failed)}")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
