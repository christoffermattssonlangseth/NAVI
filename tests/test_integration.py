"""End-to-end integration test for NAVI on synthetic AnnData.

Validates that:
  1. .fit(adata) completes without error on a small two-sample dataset.
  2. .get_latent(adata) returns arrays of correct shape and dtype.
  3. Latent embeddings are stored in adata.obsm with expected keys.
  4. Embeddings are deterministic: two calls to get_latent() return identical arrays.
  5. z_cell and z_spatial have expected dimensionality.
  6. z_joint == concat(z_cell, z_spatial).
  7. Disentanglement diagnostics run without error and return sensible values.
  8. KL annealing parameters are accepted and logged.
  9. Staged training mode completes end-to-end.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("torch_geometric")
if importlib.util.find_spec("lightning") is None and importlib.util.find_spec("pytorch_lightning") is None:
    pytest.skip("lightning or pytorch_lightning is required", allow_module_level=True)

from navi import NAVI, compute_disentanglement_metrics


# ---------------------------------------------------------------------------
# Shared synthetic dataset fixture
# ---------------------------------------------------------------------------


def _make_two_sample_adata(
    n_per_sample: int = 30,
    n_genes: int = 100,
    seed: int = 42,
) -> AnnData:
    """Minimal two-sample spatial transcriptomics AnnData."""
    rng = np.random.default_rng(seed)

    counts_s1 = rng.poisson(2.0, size=(n_per_sample, n_genes)).astype(np.float32)
    counts_s2 = rng.poisson(3.0, size=(n_per_sample, n_genes)).astype(np.float32)
    counts = np.vstack([counts_s1, counts_s2])

    # Spatially separated samples.
    coords_s1 = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n_per_sample, 2)).astype(np.float32)
    coords_s2 = rng.normal(loc=[20.0, 20.0], scale=1.0, size=(n_per_sample, 2)).astype(np.float32)
    coords = np.vstack([coords_s1, coords_s2])

    adata = AnnData(counts)
    adata.layers["counts"] = counts.copy()
    adata.obsm["spatial"] = coords
    adata.obs["sample_id"] = np.array(
        ["sample_A"] * n_per_sample + ["sample_B"] * n_per_sample,
        dtype=object,
    )
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_model(**kwargs) -> NAVI:
    defaults = dict(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        use_squidpy=False,
        k_neighbors=5,
        n_latent_cell=8,
        n_latent_spatial=8,
        cell_hidden_dims=(32,),
        spatial_hidden_dim=16,
        gat_heads=2,
        decoder_hidden_dims=(32,),
        sample_embedding_dim=8,
    )
    defaults.update(kwargs)
    return NAVI(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fit_and_get_latent_basic() -> None:
    """fit() and get_latent() must succeed and return correctly shaped arrays."""
    adata = _make_two_sample_adata()
    n_cells = adata.n_obs

    model = _make_small_model()
    model.fit(adata)
    latents = model.get_latent(adata)

    # Keys present in obsm
    assert "X_z_cell" in adata.obsm
    assert "X_z_spatial" in adata.obsm
    assert "X_z_joint" in adata.obsm

    # Shape checks
    assert latents["z_cell"].shape == (n_cells, 8), f"z_cell shape: {latents['z_cell'].shape}"
    assert latents["z_spatial"].shape == (n_cells, 8), f"z_spatial shape: {latents['z_spatial'].shape}"
    assert latents["z_joint"].shape == (n_cells, 16), f"z_joint shape: {latents['z_joint'].shape}"

    # Dtype check
    assert latents["z_cell"].dtype == np.float32


def test_get_latent_is_deterministic() -> None:
    """get_latent() must return identical arrays on repeated calls (uses means)."""
    adata = _make_two_sample_adata()
    model = _make_small_model()
    model.fit(adata)

    l1 = model.get_latent(adata)
    l2 = model.get_latent(adata)

    np.testing.assert_array_equal(l1["z_cell"], l2["z_cell"],
                                   err_msg="z_cell is not deterministic across get_latent() calls")
    np.testing.assert_array_equal(l1["z_spatial"], l2["z_spatial"],
                                   err_msg="z_spatial is not deterministic")
    np.testing.assert_array_equal(l1["z_joint"], l2["z_joint"],
                                   err_msg="z_joint is not deterministic")


def test_z_joint_equals_concat_of_cell_and_spatial() -> None:
    """z_joint must be the column-concatenation of z_cell and z_spatial."""
    adata = _make_two_sample_adata()
    model = _make_small_model()
    model.fit(adata)
    latents = model.get_latent(adata)

    expected = np.concatenate([latents["z_cell"], latents["z_spatial"]], axis=1)
    np.testing.assert_allclose(latents["z_joint"], expected, atol=1e-6,
                                err_msg="z_joint is not cat(z_cell, z_spatial)")


def test_obsm_matches_return_value() -> None:
    """adata.obsm values must match the dict returned by get_latent()."""
    adata = _make_two_sample_adata()
    model = _make_small_model()
    model.fit(adata)
    latents = model.get_latent(adata)

    np.testing.assert_array_equal(adata.obsm["X_z_cell"], latents["z_cell"])
    np.testing.assert_array_equal(adata.obsm["X_z_spatial"], latents["z_spatial"])
    np.testing.assert_array_equal(adata.obsm["X_z_joint"], latents["z_joint"])


def test_kl_annealing_parameters_accepted() -> None:
    """KL annealing parameters must be accepted by NAVI without error."""
    adata = _make_two_sample_adata()
    model = _make_small_model(
        kl_warmup_epochs=1,
        kl_ramp_epochs=5,
        beta1=4.0,
        beta2=4.0,
        max_epochs=3,
    )
    model.fit(adata)
    latents = model.get_latent(adata)
    assert latents["z_cell"].shape[0] == adata.n_obs


def test_staged_training_mode() -> None:
    """Staged training (cell_pretrain → joint) must complete end-to-end."""
    adata = _make_two_sample_adata()
    model = _make_small_model(
        training_mode="staged",
        cell_pretrain_epochs=1,
        max_epochs=3,
    )
    model.fit(adata)
    latents = model.get_latent(adata)
    assert latents["z_cell"].shape == (adata.n_obs, 8)


def test_no_counts_layer_fallback_to_X() -> None:
    """When counts_layer is not in layers, X must be used as fallback."""
    adata = _make_two_sample_adata()
    del adata.layers["counts"]  # remove the layer so X is used

    model = _make_small_model(counts_layer="counts")  # layer absent → falls back to X
    model.fit(adata)
    assert "X_z_cell" in adata.obsm


def test_graph_metadata_stored() -> None:
    """graph_metadata_ must be populated after fit()."""
    adata = _make_two_sample_adata()
    model = _make_small_model()
    model.fit(adata)
    assert model.graph_metadata_ is not None
    assert "adjusted_k_per_sample" in model.graph_metadata_


def test_disentanglement_metrics_run() -> None:
    """compute_disentanglement_metrics must return expected keys and sensible values."""
    adata = _make_two_sample_adata(n_per_sample=40)
    model = _make_small_model(max_epochs=2)
    model.fit(adata)

    metrics = compute_disentanglement_metrics(
        embeddings=adata.obsm["X_z_cell"],
        sample_labels=adata.obs["sample_id"].to_numpy(),
        k=10,
    )

    assert "lisi_mean" in metrics
    assert "lisi_median" in metrics
    assert "lisi_per_cell" in metrics
    assert "kbet_acceptance_rate" in metrics
    assert "n_samples" in metrics

    # After 2 epochs of training, we can't expect perfect disentanglement,
    # but the values must be in valid ranges.
    assert 1.0 <= metrics["lisi_mean"] <= metrics["n_samples"] + 0.01
    assert 0.0 <= metrics["kbet_acceptance_rate"] <= 1.0
    assert metrics["lisi_per_cell"].shape == (adata.n_obs,)
    assert metrics["n_samples"] == 2


def test_validate_adata_missing_sample_key() -> None:
    """fit() must raise KeyError when sample_key is missing from obs."""
    adata = _make_two_sample_adata()
    del adata.obs["sample_id"]
    model = _make_small_model()
    with pytest.raises(KeyError, match="sample_id"):
        model.fit(adata)


def test_validate_adata_missing_spatial_key() -> None:
    """fit() must raise KeyError when spatial_key is missing from obsm."""
    adata = _make_two_sample_adata()
    del adata.obsm["spatial"]
    model = _make_small_model()
    with pytest.raises(KeyError, match="spatial"):
        model.fit(adata)


def test_get_latent_before_fit_raises() -> None:
    """get_latent() must raise RuntimeError if called before fit()."""
    adata = _make_two_sample_adata()
    model = _make_small_model()
    with pytest.raises(RuntimeError, match="fit()"):
        model.get_latent(adata)
