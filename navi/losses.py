"""Loss functions for NAVI."""

from __future__ import annotations

import torch
from torch import Tensor


def kl_standard_normal(z_mean: Tensor, z_logvar: Tensor) -> Tensor:
    """Compute KL divergence to standard normal for each sample."""
    return 0.5 * torch.sum(
        torch.exp(z_logvar) + z_mean.pow(2) - 1.0 - z_logvar,
        dim=-1,
    )


def negative_binomial_nll(
    counts: Tensor,
    mu: Tensor,
    theta: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Return per-cell NB negative log-likelihood."""
    counts = counts.clamp(min=0.0)
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    log_prob = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + counts * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )
    return -torch.sum(log_prob, dim=-1)


def film(z_spatial: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    """Apply FiLM modulation."""
    return gamma * z_spatial + beta


def adversarial_weight(
    epoch: int,
    lambda_adv: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """Warm up then linearly ramp adversarial strength."""
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(lambda_adv)
    progress = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return float(lambda_adv) * progress


def beta_kl_schedule(
    epoch: int,
    beta_max: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """Linearly anneal beta-KL from 0 to beta_max.

    During the first ``warmup_epochs`` epochs the KL term is fully suppressed
    (effective beta = 0) to let the reconstruction loss dominate early training.
    Over the following ``ramp_epochs`` the weight increases linearly to
    ``beta_max``.  Set warmup_epochs=0 and ramp_epochs=0 to disable annealing.
    """
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(beta_max)
    progress = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    return float(beta_max) * progress
