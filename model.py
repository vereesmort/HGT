"""
model.py
Architecture:
    1. Input projection  — project each node type to hidden_dim
    2. HGT encoder       — L layers of heterogeneous graph transformer
    3. Pathway attention pooling — inject biological priors via KEGG pathway structure
    4. Fusion MLP        — combine HGT drug embedding + pathway fingerprint
    5. DEDICOM decoder   — typed bilinear scoring per SE type
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input projection
# ─────────────────────────────────────────────────────────────────────────────

class InputProjection(nn.Module):
    """Project each node type's raw features to a shared hidden_dim."""

    def __init__(self, in_dims: Dict[str, int], hidden_dim: int):
        super().__init__()
        self.projs = nn.ModuleDict({
            node_type: nn.Linear(in_dim, hidden_dim, bias=True)
            for node_type, in_dim in in_dims.items()
        })

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {ntype: F.relu(self.projs[ntype](x))
                for ntype, x in x_dict.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 2. HGT encoder
# ─────────────────────────────────────────────────────────────────────────────

class HGTEncoder(nn.Module):
    """
    Stack of HGTConv layers operating on the heterogeneous drug-protein-SE graph.
    Each layer uses type-specific W_Q, W_K, W_V per relation type.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        metadata,           # graph.metadata() — node types + edge types
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim)
                for nt in metadata[0]   # metadata[0] = node_types list
            })
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Dict[str, Tensor]:
        for layer, norm_dict in zip(self.layers, self.norms):
            x_dict_new = layer(x_dict, edge_index_dict)
            # Residual + LayerNorm + Dropout
            x_dict = {
                ntype: norm_dict[ntype](
                    self.dropout(x_dict_new[ntype]) + x_dict[ntype]
                )
                for ntype in x_dict
                if ntype in x_dict_new
            }
        return x_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pathway attention pooling
# ─────────────────────────────────────────────────────────────────────────────

class PathwayAttentionPooling(nn.Module):
    """
    For each drug node, aggregate its target proteins' embeddings grouped by
    KEGG pathway, using a learned attention weight per protein-pathway pair.
    Output: a pathway fingerprint vector for each drug.

    drug_pathway_map: {drug_node_idx: {pathway_id: [protein_node_indices]}}
    pathway_id_to_col: {pathway_id: column_index}  — fixed ordering
    """

    def __init__(self, hidden_dim: int, num_pathways: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_pathways = num_pathways

        # Attention scoring: score(protein_h) -> scalar weight within pathway
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

        # Project concatenated pathway pool to output dim
        # We use mean pooling across pathways to handle variable coverage
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        drug_node_indices: Tensor,          # [N_drugs_in_batch]
        protein_h: Tensor,                  # [N_proteins, hidden_dim]
        drug_pathway_map: dict,             # preloaded from pathway_memberships.pkl
        device: torch.device,
    ) -> Tensor:
        """
        Returns pathway_fingerprint: [N_drugs_in_batch, hidden_dim]
        Drugs with no target proteins get a zero vector.
        """
        out = torch.zeros(
            len(drug_node_indices), self.hidden_dim, device=device
        )

        for batch_pos, d_idx in enumerate(drug_node_indices.tolist()):
            pw_dict = drug_pathway_map.get(d_idx)
            if not pw_dict:
                continue  # no target proteins — zero vector (handled by mono SE signal)

            pathway_pools = []
            for pw_id, prot_indices in pw_dict.items():
                if not prot_indices:
                    continue
                prot_idx_t = torch.tensor(prot_indices, dtype=torch.long, device=device)
                # Clamp to valid range
                prot_idx_t = prot_idx_t.clamp(0, protein_h.shape[0] - 1)
                prot_h = protein_h[prot_idx_t]         # [K_pw, hidden_dim]

                # Attention weights over proteins within this pathway
                scores = self.attn_score(prot_h)        # [K_pw, 1]
                weights = torch.softmax(scores, dim=0)  # [K_pw, 1]
                pool = (weights * prot_h).sum(dim=0)    # [hidden_dim]
                pathway_pools.append(pool)

            if pathway_pools:
                # Mean pool across pathways the drug's targets participate in
                stacked = torch.stack(pathway_pools, dim=0)  # [P, hidden_dim]
                fingerprint = stacked.mean(dim=0)             # [hidden_dim]
                out[batch_pos] = fingerprint

        out = F.relu(self.out_proj(self.dropout(out)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fusion MLP
# ─────────────────────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """
    Fuse HGT drug embedding + pathway fingerprint -> final drug embedding z_drug.
    Input: [h_drug || pathway_fp]  (2 * hidden_dim)
    Output: z_drug  (hidden_dim)
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, h_drug: Tensor, pathway_fp: Tensor) -> Tensor:
        return self.net(torch.cat([h_drug, pathway_fp], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEDICOM-style typed decoder
# ─────────────────────────────────────────────────────────────────────────────

class DEDICOMDecoder(nn.Module):
    """
    score(i, j, r) = sigmoid(z_i · D_r · R · D_r · z_j^T)

    D_r : diagonal matrix per SE type (hidden_dim,) — which dimensions matter for SE r
    R   : shared global relation matrix (hidden_dim x hidden_dim)

    This is the same decoder as Zitnik et al. 2018, but now operating on richer
    z_drug embeddings produced by the HGT + pathway pooling encoder.

    D_r diagonals are interpretable: large |D_r[k]| means embedding dimension k
    is important for predicting side effect r — enabling post-hoc analysis of
    which biological signals (learned from which graph neighbourhood) drive each SE.
    """

    def __init__(self, hidden_dim: int, num_se: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_se     = num_se

        # R: shared global relation matrix, initialised near identity
        self.R = nn.Parameter(torch.eye(hidden_dim) + 0.01 * torch.randn(hidden_dim, hidden_dim))

        # D_r: diagonal per SE type, stored as [num_se, hidden_dim]
        self.D = nn.Parameter(torch.ones(num_se, hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z_i: Tensor,   # [B, hidden_dim]  drug i embeddings
        z_j: Tensor,   # [B, hidden_dim]  drug j embeddings
        se_indices: Optional[Tensor] = None,  # [B] — score only these SE types (for efficiency)
    ) -> Tensor:
        """
        Returns scores: [B, num_se] if se_indices is None
                        [B]         if se_indices provided (one score per pair)
        """
        z_i = self.dropout(z_i)
        z_j = self.dropout(z_j)

        if se_indices is not None:
            # Efficient: score only one SE type per pair
            d_r = self.D[se_indices]                    # [B, hidden_dim]
            Rz_j = z_j @ self.R.T                       # [B, hidden_dim]
            scores = (z_i * d_r * (Rz_j * d_r)).sum(-1) # [B]
            return torch.sigmoid(scores)
        else:
            # Score all SE types: returns [B, num_se]
            # z_i · D_r · R · D_r · z_j for each r
            Rz_j = z_j @ self.R.T                        # [B, hidden_dim]
            # Expand: z_i [B,1,d], D [1,S,d], Rz_j [B,1,d]
            z_i_e  = z_i.unsqueeze(1)                    # [B, 1, d]
            Rz_j_e = Rz_j.unsqueeze(1)                   # [B, 1, d]
            D_e    = self.D.unsqueeze(0)                  # [1, S, d]
            scores = (z_i_e * D_e * (Rz_j_e * D_e)).sum(-1)  # [B, S]
            return torch.sigmoid(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full model
# ─────────────────────────────────────────────────────────────────────────────

class PolypharmacyHGT(nn.Module):
    """
    End-to-end model for polypharmacy side effect prediction.

    Forward pass returns scores [B, num_se] for a batch of drug pairs.
    """

    def __init__(
        self,
        in_dims: Dict[str, int],   # {node_type: input_feature_dim}
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_se: int,
        num_pathways: int,
        graph_metadata,
        dropout: float = 0.1,
    ):
        super().__init__()

        # mono_se nodes have no meaningful input features — use learnable embedding
        self.mono_se_embed = nn.Embedding(in_dims["_mono_se_count"], hidden_dim)
        in_dims_proj = {k: v for k, v in in_dims.items() if not k.startswith("_")}
        # mono_se projection is handled via embedding, so remove from proj
        in_dims_proj.pop("mono_se", None)
        in_dims_proj["mono_se"] = hidden_dim   # embedding output goes directly in

        self.input_proj     = InputProjection(in_dims_proj, hidden_dim)
        self.hgt_encoder    = HGTEncoder(hidden_dim, num_heads, num_layers, graph_metadata, dropout)
        self.pathway_pool   = PathwayAttentionPooling(hidden_dim, num_pathways, dropout)
        self.fusion_mlp     = FusionMLP(hidden_dim, dropout)
        self.decoder        = DEDICOMDecoder(hidden_dim, num_se, dropout)

        self.hidden_dim = hidden_dim

    def encode(
        self,
        data: HeteroData,
        drug_pathway_map: dict,
        device: torch.device,
    ) -> Tensor:
        """
        Run encoder and return z_drug: [N_drugs, hidden_dim]
        """
        # Replace mono_se placeholder features with learned embeddings
        x_dict = {
            "drug":    data["drug"].x,
            "protein": data["protein"].x,
            "mono_se": self.mono_se_embed(
                torch.arange(data["mono_se"].num_nodes, device=device)
            ),
        }

        # Project all node types to hidden_dim
        h_dict = self.input_proj(x_dict)

        # HGT message passing
        h_dict = self.hgt_encoder(h_dict, data.edge_index_dict)

        h_drug    = h_dict["drug"]      # [N_drugs, hidden_dim]
        h_protein = h_dict["protein"]   # [N_proteins, hidden_dim]

        # Pathway attention pooling for all drug nodes
        all_drug_indices = torch.arange(h_drug.shape[0], device=device)
        pathway_fp = self.pathway_pool(all_drug_indices, h_protein, drug_pathway_map, device)

        # Fuse HGT drug embedding with pathway fingerprint
        z_drug = self.fusion_mlp(h_drug, pathway_fp)   # [N_drugs, hidden_dim]

        return z_drug

    def forward(
        self,
        data: HeteroData,
        drug_pair_index: Tensor,    # [2, B]
        drug_pathway_map: dict,
        device: torch.device,
        se_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns scores [B, num_se] or [B] if se_indices given.
        """
        z_drug = self.encode(data, drug_pathway_map, device)
        z_i = z_drug[drug_pair_index[0]]   # [B, hidden_dim]
        z_j = z_drug[drug_pair_index[1]]   # [B, hidden_dim]
        return self.decoder(z_i, z_j, se_indices)

    @torch.no_grad()
    def get_drug_embeddings(
        self,
        data: HeteroData,
        drug_pathway_map: dict,
        device: torch.device,
    ) -> Tensor:
        """Return final drug embeddings for analysis/validation."""
        self.eval()
        return self.encode(data, drug_pathway_map, device)