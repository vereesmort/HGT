# Polypharmacy Side Effect Prediction — HGT Pipeline

## Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install bioservices gprofiler-official scikit-learn tqdm pandas numpy rdkit
```

## Data

Place the DECAGON files in `data/raw/`:
- `bio-decagon-mono.csv`
- `bio-decagon-ppi.csv`
- `bio-decagon-targets.csv`
- `bio-decagon-combo.csv`  (full file, not chunks)

## Pipeline

Run in order:

```bash
python 01_fetch_kegg.py          # fetch KEGG pathway→gene mappings, saves data/kegg_pathways.json
python 02_build_graph.py         # build HeteroData graph, saves data/processed/graph.pt
python 03_build_splits.py        # stratified drug-pair splits, saves data/processed/splits.pt
python 04_train.py               # train HGT model, saves checkpoints/
python 05_evaluate.py            # AUROC/AUPRC per SE type
python 06_enrichment.py          # GO/pathway enrichment on learned embeddings
```
