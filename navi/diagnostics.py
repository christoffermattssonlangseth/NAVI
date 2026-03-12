"""Disentanglement diagnostics for NAVI latent spaces.

Provides LISI and a kBET-style acceptance-rate metric to quantify whether
z_cell is sample-agnostic after adversarial training.

Usage
-----
    from navi.diagnostics import compute_disentanglement_metrics

    metrics = compute_disentanglement_metrics(
        embeddings=adata.obsm["X_z_cell"],
        sample_labels=adata.obs["sample_id"].to_numpy(),
        k=30,
    )
    print(metrics)
    # {'lisi_mean': 1.85, 'lisi_median': 1.92, 'kbet_acceptance_rate': 0.71}

Interpretation
--------------
LISI (Local Inverse Simpson's Index):
  * Range: 1.0 (all neighbours from one sample) → n_samples (perfect mixing).
  * Higher is better for z_cell disentanglement.

kBET acceptance rate:
  * Fraction of cells whose neighbourhood sample distribution is statistically
    indistinguishable from the global sample frequency (chi-square test, α=0.05).
  * Range: 0 (no mixing) → 1 (perfect mixing).
  * Higher is better for z_cell disentanglement.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _build_knn_index(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return neighbour indices of shape (n_cells, k), excluding self."""
    try:
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nbrs.fit(embeddings)
        indices = nbrs.kneighbors(return_distance=False)[:, 1:]  # drop self
        return indices
    except ImportError:
        pass

    # Pure-numpy fallback (O(n²), suitable for small n only).
    n = embeddings.shape[0]
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    k_actual = min(k, n - 1)
    idx = np.argpartition(dist2, kth=k_actual - 1, axis=1)[:, :k_actual]
    if k_actual < k:
        # Pad with the first neighbour to keep shape consistent.
        pad = np.repeat(idx[:, :1], k - k_actual, axis=1)
        idx = np.concatenate([idx, pad], axis=1)
    return idx


def compute_lisi(
    embeddings: np.ndarray,
    sample_labels: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    """Compute per-cell LISI scores.

    Parameters
    ----------
    embeddings:
        Array of shape (n_cells, n_dims).
    sample_labels:
        1-D array of sample identifiers, length n_cells.
    k:
        Number of neighbours for the local neighbourhood.

    Returns
    -------
    lisi_scores:
        Array of shape (n_cells,). Values in [1, n_samples].
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    sample_labels = np.asarray(sample_labels)
    n_cells = embeddings.shape[0]
    k = min(k, n_cells - 1)

    indices = _build_knn_index(embeddings, k)

    lisi_scores = np.empty(n_cells, dtype=np.float64)
    for i in range(n_cells):
        neighbours = sample_labels[indices[i]]
        _, counts = np.unique(neighbours, return_counts=True)
        # Simpson's index: sum of squared relative frequencies.
        probs = counts / k
        simpson = float(np.sum(probs**2))
        # LISI = effective number of distinct categories.
        lisi_scores[i] = 1.0 / simpson if simpson > 0 else 1.0

    return lisi_scores


def compute_kbet_acceptance_rate(
    embeddings: np.ndarray,
    sample_labels: np.ndarray,
    k: int = 30,
    alpha: float = 0.05,
) -> float:
    """Compute a kBET-style acceptance rate.

    For each cell, tests whether the sample composition of its k-nearest
    neighbours matches the global sample frequency using a chi-square test.
    The acceptance rate is the fraction of cells that pass (are not
    significantly different from the global distribution).

    Parameters
    ----------
    embeddings:
        Array of shape (n_cells, n_dims).
    sample_labels:
        1-D array of sample identifiers, length n_cells.
    k:
        Neighbourhood size.
    alpha:
        Significance level for the chi-square test (default 0.05).

    Returns
    -------
    acceptance_rate:
        Scalar in [0, 1].  Higher → better mixing.
    """
    try:
        from scipy.stats import chi2 as chi2_dist
    except ImportError as exc:
        raise ImportError(
            "scipy is required for kBET computation. Install it with: pip install scipy"
        ) from exc

    embeddings = np.asarray(embeddings, dtype=np.float64)
    sample_labels = np.asarray(sample_labels)
    n_cells = embeddings.shape[0]
    k = min(k, n_cells - 1)

    unique_samples = np.unique(sample_labels)
    n_unique = len(unique_samples)
    sample_to_idx = {s: i for i, s in enumerate(unique_samples)}
    labels_int = np.array([sample_to_idx[s] for s in sample_labels], dtype=np.int64)

    global_freq = np.bincount(labels_int, minlength=n_unique) / n_cells

    # Chi-square critical value at significance level alpha.
    df = n_unique - 1
    critical_val = chi2_dist.ppf(1.0 - alpha, df) if df > 0 else np.inf

    indices = _build_knn_index(embeddings, k)

    n_accepted = 0
    for i in range(n_cells):
        neighbour_labels = labels_int[indices[i]]
        observed = np.bincount(neighbour_labels, minlength=n_unique).astype(np.float64)
        expected = global_freq * k
        # Guard against zero expected counts (prevents division by zero).
        chi2_stat = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-8)))
        if chi2_stat < critical_val:
            n_accepted += 1

    return n_accepted / n_cells


def compute_disentanglement_metrics(
    embeddings: np.ndarray,
    sample_labels: np.ndarray,
    k: int = 30,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute LISI and kBET metrics for a latent embedding.

    Call this on ``adata.obsm["X_z_cell"]`` after training to verify that
    the adversarial loss is making z_cell sample-agnostic.

    Parameters
    ----------
    embeddings:
        Latent representation, shape (n_cells, n_dims).
    sample_labels:
        Sample identifier per cell, length n_cells.
    k:
        Neighbourhood size for both metrics.
    alpha:
        Significance level for kBET chi-square test.

    Returns
    -------
    dict with keys:
        ``lisi_mean``     – mean LISI across cells (higher = better mixing).
        ``lisi_median``   – median LISI across cells.
        ``lisi_per_cell`` – per-cell LISI array for downstream plotting.
        ``kbet_acceptance_rate`` – fraction of cells that pass kBET (higher = better).
        ``n_samples``     – number of unique samples detected.
    """
    sample_labels = np.asarray(sample_labels)
    n_samples = int(np.unique(sample_labels).size)

    lisi = compute_lisi(embeddings, sample_labels, k=k)
    kbet_rate = compute_kbet_acceptance_rate(embeddings, sample_labels, k=k, alpha=alpha)

    return {
        "lisi_mean": float(np.mean(lisi)),
        "lisi_median": float(np.median(lisi)),
        "lisi_per_cell": lisi,
        "kbet_acceptance_rate": kbet_rate,
        "n_samples": n_samples,
    }
