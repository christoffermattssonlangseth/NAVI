"""Unit tests for critical NAVI components.

Tests are written before implementation fixes so failures are explicit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from navi.encoders import (
    CellEncoder,
    GradientReversal,
    SampleDiscriminator,
    SampleEmbedding,
    SpatialEncoder,
)
from navi.losses import (
    adversarial_weight,
    beta_kl_schedule,
    kl_standard_normal,
    negative_binomial_nll,
)

pytest.importorskip("torch_geometric")


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------


def test_gradient_reversal_flips_sign() -> None:
    """GRL must negate gradients in backward, passing values unchanged in forward."""
    grl = GradientReversal(scale=1.0)
    x = torch.ones(4, 8, requires_grad=True)
    y = grl(x)

    # Forward: output must be identical to input.
    assert torch.equal(y, x), "GRL forward must be an identity."

    # Backward: grad of sum(y) w.r.t. x is all-ones (forward identity),
    # but GRL flips the sign, so x.grad must be all-(-1).
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, -torch.ones_like(x)), (
        f"Expected grad=-1 everywhere, got {x.grad}"
    )


def test_gradient_reversal_scale() -> None:
    """GRL with scale=2.0 must multiply the reversed gradient by 2."""
    grl = GradientReversal(scale=2.0)
    x = torch.ones(3, 5, requires_grad=True)
    grl(x).sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x))


def test_gradient_reversal_chain_rule() -> None:
    """GRL must compose correctly with downstream modules via chain rule."""
    grl = GradientReversal(scale=1.0)
    linear = torch.nn.Linear(4, 1, bias=False)
    torch.nn.init.ones_(linear.weight)  # weight = [1,1,1,1]

    x = torch.ones(1, 4, requires_grad=True)
    out = linear(grl(x))  # out = sum(grl(x)) = 4
    out.backward()

    # d(out)/d(x) through linear = [1,1,1,1], then GRL flips → [-1,-1,-1,-1]
    assert x.grad is not None
    assert torch.allclose(x.grad, -torch.ones(1, 4))


# ---------------------------------------------------------------------------
# Negative Binomial NLL
# ---------------------------------------------------------------------------


def _nb_log_prob_numpy(k: float, r: float, mu: float) -> float:
    """Reference NB log-probability using math (no eps shortcuts)."""
    # P(k; r, mu): p = mu/(mu+r), (1-p) = r/(mu+r)
    return (
        math.lgamma(k + r)
        - math.lgamma(r)
        - math.lgamma(k + 1.0)
        + r * math.log(r / (r + mu))
        + k * math.log(mu / (mu + r))
    )


def test_nb_nll_correct_on_known_inputs() -> None:
    """NB NLL must match reference values computed with math.lgamma."""
    test_cases = [
        (2.0, 5.0, 2.0),   # (k, theta, mu) -- moderate counts
        (0.0, 1.0, 0.5),   # zero counts
        (10.0, 2.0, 8.0),  # high counts, low dispersion
    ]
    for k, theta, mu in test_cases:
        expected_nll = -_nb_log_prob_numpy(k, theta, mu)
        counts = torch.tensor([[k]])
        mu_t = torch.tensor([[mu]])
        theta_t = torch.tensor([[theta]])
        nll = negative_binomial_nll(counts, mu_t, theta_t)
        assert nll.shape == (1,), f"Expected shape (1,) for case {(k, theta, mu)}"
        assert abs(nll.item() - expected_nll) < 1e-4, (
            f"NLL mismatch for k={k}, theta={theta}, mu={mu}: "
            f"got {nll.item():.6f}, expected {expected_nll:.6f}"
        )


def test_nb_nll_is_finite_for_edge_cases() -> None:
    """NB NLL must remain finite for near-zero counts and parameters."""
    counts = torch.tensor([[0.0, 1.0, 100.0]])
    mu = torch.tensor([[1e-6, 1.0, 50.0]])
    theta = torch.tensor([[1e-6, 1.0, 10.0]])
    nll = negative_binomial_nll(counts, mu, theta)
    assert torch.isfinite(nll).all(), "NB NLL must be finite for edge-case inputs."


def test_nb_nll_non_negative() -> None:
    """NB NLL must be non-negative (it is a negative log-probability)."""
    rng = torch.Generator()
    rng.manual_seed(42)
    counts = torch.randint(0, 20, (32, 50), generator=rng).float()
    mu = torch.rand(32, 50, generator=rng) * 10 + 0.1
    theta = torch.rand(32, 50, generator=rng) * 5 + 0.1
    nll = negative_binomial_nll(counts, mu, theta)
    assert (nll >= 0).all(), "NB NLL must be non-negative."


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def test_kl_standard_normal_zero_for_standard_gaussian() -> None:
    mean = torch.zeros(8, 16)
    logvar = torch.zeros(8, 16)
    kl = kl_standard_normal(mean, logvar)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_standard_normal_positive_for_shifted() -> None:
    """KL must be strictly positive when distribution is not N(0,I)."""
    mean = torch.ones(4, 8)
    logvar = torch.zeros(4, 8)
    kl = kl_standard_normal(mean, logvar)
    assert (kl > 0).all()


def test_kl_standard_normal_shape() -> None:
    """KL output must have shape (batch,) — one value per sample, not per dimension."""
    mean = torch.zeros(5, 12)
    logvar = torch.zeros(5, 12)
    kl = kl_standard_normal(mean, logvar)
    assert kl.shape == (5,)


# ---------------------------------------------------------------------------
# Beta-KL annealing schedule
# ---------------------------------------------------------------------------


def test_beta_kl_schedule_warmup() -> None:
    """During warmup, effective beta must be 0."""
    for epoch in range(5):
        assert beta_kl_schedule(epoch, beta_max=1.0, warmup_epochs=5, ramp_epochs=10) == 0.0


def test_beta_kl_schedule_ramp() -> None:
    """After warmup, beta must increase monotonically and reach beta_max."""
    prev = 0.0
    for epoch in range(5, 20):
        w = beta_kl_schedule(epoch, beta_max=2.0, warmup_epochs=5, ramp_epochs=10)
        assert w >= prev, "Beta schedule must be non-decreasing."
        prev = w
    assert prev == pytest.approx(2.0), "Beta must reach beta_max after full ramp."


def test_beta_kl_schedule_no_warmup_no_ramp() -> None:
    """With warmup=0, ramp=0, beta must be beta_max from epoch 0."""
    assert beta_kl_schedule(0, beta_max=3.5, warmup_epochs=0, ramp_epochs=0) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Encoder shapes
# ---------------------------------------------------------------------------


def test_cell_encoder_output_shapes() -> None:
    enc = CellEncoder(n_genes=100, n_samples=3, n_latent=16, hidden_dims=(64, 32), dropout=0.0)
    counts = torch.rand(10, 100)
    sample_idx = torch.randint(0, 3, (10,))
    z, mean, logvar = enc(counts, sample_idx)
    assert z.shape == (10, 16), f"z_cell shape mismatch: {z.shape}"
    assert mean.shape == (10, 16), f"z_cell_mean shape mismatch: {mean.shape}"
    assert logvar.shape == (10, 16), f"z_cell_logvar shape mismatch: {logvar.shape}"


def test_spatial_encoder_output_shapes() -> None:
    enc = SpatialEncoder(in_dim=16, n_latent=8, hidden_dim=16, heads=2, dropout=0.0)
    z_cell = torch.randn(6, 16)
    # chain graph: 0→1, 1→2, ..., 4→5 (and reverses)
    src = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4, 5])
    dst = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2, 3, 4])
    edge_index = torch.stack([src, dst])
    z, mean, logvar = enc(z_cell, edge_index)
    assert z.shape == (6, 8), f"z_spatial shape mismatch: {z.shape}"
    assert mean.shape == (6, 8)
    assert logvar.shape == (6, 8)


def test_sample_embedding_film_shape() -> None:
    emb = SampleEmbedding(n_samples=4, n_latent_spatial=8, embedding_dim=12)
    sample_idx = torch.randint(0, 4, (10,))
    z_spatial = torch.randn(10, 8)
    z_mod, gamma, beta = emb(sample_idx, z_spatial)
    assert z_mod.shape == (10, 8)
    assert gamma.shape == (10, 8)
    assert beta.shape == (10, 8)


def test_sample_discriminator_shape() -> None:
    disc = SampleDiscriminator(n_latent=16, n_samples=3)
    z = torch.randn(8, 16)
    logits = disc(z)
    assert logits.shape == (8, 3)


# ---------------------------------------------------------------------------
# FiLM conditioning actually modulates
# ---------------------------------------------------------------------------


def test_film_conditioning_is_not_identity() -> None:
    """FiLM output must differ from the raw z_spatial (non-trivial modulation)."""
    torch.manual_seed(7)
    emb = SampleEmbedding(n_samples=2, n_latent_spatial=8, embedding_dim=8)
    sample_idx = torch.zeros(5, dtype=torch.long)  # all same sample
    z_spatial = torch.randn(5, 8)
    z_mod, gamma, beta = emb(sample_idx, z_spatial)
    # gamma ≠ 1 (random init) or beta ≠ 0 means modulation is happening
    is_gamma_identity = torch.allclose(gamma, torch.ones_like(gamma), atol=1e-3)
    is_beta_zero = torch.allclose(beta, torch.zeros_like(beta), atol=1e-3)
    assert not (is_gamma_identity and is_beta_zero), (
        "FiLM gamma and beta should not both be trivial at random init."
    )
    assert not torch.allclose(z_mod, z_spatial), (
        "FiLM output must differ from raw z_spatial."
    )


# ---------------------------------------------------------------------------
# Graph: no cross-sample edges (supplement existing test)
# ---------------------------------------------------------------------------


def test_graph_no_cross_sample_edges_with_rng() -> None:
    """Verify no cross-sample edges with random coordinates."""
    from anndata import AnnData

    from navi.graph import build_spatial_graph

    rng = np.random.default_rng(0)
    n = 40
    coords = rng.normal(size=(n, 2)).astype(np.float32)
    counts = rng.poisson(1.0, size=(n, 5)).astype(np.float32)
    adata = AnnData(counts)
    adata.obsm["spatial"] = coords
    adata.obs["sample_id"] = np.array(["A"] * 20 + ["B"] * 20, dtype=object)

    edge_index, sample_index, _, _, _ = build_spatial_graph(
        adata, k_neighbors=5, use_squidpy=False
    )
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    assert np.all(sample_index[src] == sample_index[dst]), (
        "Cross-sample edges detected in spatial graph!"
    )
