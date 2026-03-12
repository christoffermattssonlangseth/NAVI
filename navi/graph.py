"""Spatial graph construction utilities."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import torch
from anndata import AnnData


def _knn_edges_numpy(coords: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_cells = coords.shape[0]
    if n_cells <= 1 or n_neighbors <= 0:
        return np.empty((2, 0), dtype=np.int64)

    # Use dense distances as a final fallback when kNN backends are unavailable.
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    idx = np.argpartition(dist2, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), n_neighbors)
    cols = idx.reshape(-1).astype(np.int64)
    return np.vstack([rows, cols])


def _knn_edges_fallback(coords: np.ndarray, n_neighbors: int) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
        nbrs.fit(coords)
        idx = nbrs.kneighbors(return_distance=False)[:, 1:]
        rows = np.repeat(np.arange(coords.shape[0], dtype=np.int64), n_neighbors)
        cols = idx.reshape(-1).astype(np.int64)
        return np.vstack([rows, cols])
    except Exception:
        pass

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=n_neighbors + 1)
        idx = np.asarray(idx)[:, 1:]
        rows = np.repeat(np.arange(coords.shape[0], dtype=np.int64), n_neighbors)
        cols = idx.reshape(-1).astype(np.int64)
        return np.vstack([rows, cols])
    except Exception:
        return _knn_edges_numpy(coords, n_neighbors)


def _knn_edges_squidpy(coords: np.ndarray, n_neighbors: int) -> np.ndarray | None:
    try:
        import scipy.sparse as sp
        import squidpy as sq

        tmp = AnnData(np.zeros((coords.shape[0], 1), dtype=np.float32))
        tmp.obsm["spatial"] = coords
        sq.gr.spatial_neighbors(tmp, n_neighs=n_neighbors, coord_type="generic")
        connectivities = tmp.obsp.get("spatial_connectivities")
        if connectivities is None:
            return None
        connectivities = sp.coo_matrix(connectivities)
        mask = connectivities.row != connectivities.col
        rows = connectivities.row[mask].astype(np.int64)
        cols = connectivities.col[mask].astype(np.int64)
        return np.vstack([rows, cols])
    except Exception:
        return None


def build_sample_index(sample_ids: np.ndarray) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Encode sample IDs as integer indices."""
    seen: dict[str, int] = {}
    ordered: list[str] = []
    codes = np.empty(sample_ids.shape[0], dtype=np.int64)
    for i, sample_id in enumerate(sample_ids.astype(str)):
        if sample_id not in seen:
            seen[sample_id] = len(seen)
            ordered.append(sample_id)
        codes[i] = seen[sample_id]
    return codes, seen, ordered


def build_spatial_graph(
    adata: AnnData,
    sample_key: str = "sample_id",
    spatial_key: str = "spatial",
    k_neighbors: int = 15,
    use_squidpy: bool = True,
) -> tuple[torch.LongTensor, np.ndarray, dict[str, int], list[str], dict[str, Any]]:
    """Build a per-sample kNN graph without cross-sample edges."""
    if sample_key not in adata.obs:
        raise KeyError(f"Missing required obs key '{sample_key}'.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing required obsm key '{spatial_key}'.")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1.")

    sample_ids = adata.obs[sample_key].to_numpy().astype(str)
    sample_index, sample_to_idx, ordered_samples = build_sample_index(sample_ids)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"obsm['{spatial_key}'] must be 2D with at least 2 columns.")

    edge_blocks: list[np.ndarray] = []
    adjusted_k: dict[str, int] = {}
    skipped_samples: list[str] = []

    for sample_name, sample_code in sample_to_idx.items():
        sample_mask = sample_index == sample_code
        sample_nodes = np.flatnonzero(sample_mask)
        n_cells = int(sample_nodes.size)
        if n_cells <= 1:
            skipped_samples.append(sample_name)
            continue

        k_eff = min(k_neighbors, n_cells - 1)
        adjusted_k[sample_name] = k_eff
        if k_eff < k_neighbors:
            warnings.warn(
                f"Sample '{sample_name}' has {n_cells} cells; reducing k_neighbors from {k_neighbors} to {k_eff}.",
                RuntimeWarning,
                stacklevel=2,
            )
        local_coords = coords[sample_mask]

        local_edges = _knn_edges_squidpy(local_coords, k_eff) if use_squidpy else None
        if local_edges is None:
            local_edges = _knn_edges_fallback(local_coords, k_eff)
        if local_edges.size == 0:
            continue

        global_src = sample_nodes[local_edges[0]]
        global_dst = sample_nodes[local_edges[1]]
        edge_blocks.append(np.vstack([global_src, global_dst]).astype(np.int64))

    if not edge_blocks:
        raise ValueError("No graph edges were created. Check spatial coordinates and sample sizes.")

    if skipped_samples:
        warnings.warn(
            "Skipped samples with <=1 cell during graph construction: "
            + ", ".join(skipped_samples),
            RuntimeWarning,
            stacklevel=2,
        )

    edge_index = np.concatenate(edge_blocks, axis=1)
    reverse_edges = edge_index[[1, 0], :]
    edge_index = np.concatenate([edge_index, reverse_edges], axis=1)
    edge_index = np.unique(edge_index.T, axis=0).T

    metadata = {
        "adjusted_k_per_sample": adjusted_k,
        "skipped_samples": skipped_samples,
    }
    return (
        torch.as_tensor(edge_index, dtype=torch.long),
        sample_index,
        sample_to_idx,
        ordered_samples,
        metadata,
    )
