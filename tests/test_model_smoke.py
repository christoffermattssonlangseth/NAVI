from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("torch_geometric")
if importlib.util.find_spec("lightning") is None and importlib.util.find_spec("pytorch_lightning") is None:
    pytest.skip("lightning or pytorch_lightning is required", allow_module_level=True)

from navi import NAVI


def test_model_fit_and_get_latent_smoke() -> None:
    rng = np.random.default_rng(123)
    counts = rng.poisson(1.5, size=(20, 12)).astype(np.float32)
    adata = AnnData(counts)
    adata.layers["counts"] = counts.copy()
    adata.obs["sample_id"] = np.array(["s1"] * 10 + ["s2"] * 10, dtype=object)
    adata.obs["condition"] = np.array(["MOG"] * 20, dtype=object)
    adata.obs["timepoint"] = np.array(["d7"] * 20, dtype=object)
    adata.obsm["spatial"] = np.vstack(
        [
            rng.normal(loc=(0.0, 0.0), scale=0.25, size=(10, 2)),
            rng.normal(loc=(5.0, 5.0), scale=0.25, size=(10, 2)),
        ]
    ).astype(np.float32)

    model = NAVI(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        use_squidpy=False,
        k_neighbors=5,
        n_latent_cell=8,
        n_latent_spatial=8,
    )
    model.fit(adata)
    latents = model.get_latent(adata)

    assert "X_z_cell" in adata.obsm
    assert "X_z_spatial" in adata.obsm
    assert "X_z_joint" in adata.obsm
    assert latents["z_cell"].shape == (adata.n_obs, 8)
    assert latents["z_spatial"].shape == (adata.n_obs, 8)
    assert latents["z_joint"].shape == (adata.n_obs, 16)
