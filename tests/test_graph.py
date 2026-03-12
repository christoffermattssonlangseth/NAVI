from __future__ import annotations

import numpy as np
from anndata import AnnData

from navi.graph import build_spatial_graph


def _make_graph_adata() -> AnnData:
    counts = np.random.poisson(1.0, size=(6, 4)).astype(np.float32)
    adata = AnnData(counts)
    adata.obs["sample_id"] = ["s1", "s1", "s1", "s2", "s2", "s2"]
    adata.obsm["spatial"] = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=np.float32,
    )
    return adata


def test_graph_has_no_cross_sample_edges() -> None:
    adata = _make_graph_adata()
    edge_index, sample_index, _, _, _ = build_spatial_graph(
        adata,
        sample_key="sample_id",
        spatial_key="spatial",
        k_neighbors=2,
        use_squidpy=False,
    )
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    assert np.all(sample_index[src] == sample_index[dst])


def test_graph_auto_adjusts_k_for_small_sample() -> None:
    counts = np.random.poisson(1.0, size=(5, 3)).astype(np.float32)
    adata = AnnData(counts)
    adata.obs["sample_id"] = ["tiny", "normal", "normal", "normal", "normal"]
    adata.obsm["spatial"] = np.array(
        [[0, 0], [2, 2], [3, 2], [2, 3], [3, 3]],
        dtype=np.float32,
    )
    _, _, _, _, metadata = build_spatial_graph(
        adata,
        sample_key="sample_id",
        spatial_key="spatial",
        k_neighbors=15,
        use_squidpy=False,
    )
    assert "tiny" in metadata["skipped_samples"]
    assert metadata["adjusted_k_per_sample"]["normal"] == 3
