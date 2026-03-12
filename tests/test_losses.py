from __future__ import annotations

import torch

from navi.losses import adversarial_weight, kl_standard_normal, negative_binomial_nll


def test_kl_standard_normal_is_zero_for_standard_gaussian() -> None:
    mean = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)
    kl = kl_standard_normal(mean, logvar)
    assert torch.allclose(kl, torch.zeros_like(kl))


def test_nb_nll_is_finite() -> None:
    counts = torch.tensor([[1.0, 2.0, 3.0]])
    mu = torch.tensor([[1.2, 1.8, 2.9]])
    theta = torch.tensor([[2.0, 2.0, 2.0]])
    nll = negative_binomial_nll(counts, mu, theta)
    assert torch.isfinite(nll).all()
    assert nll.shape == (1,)


def test_adversarial_weight_schedule() -> None:
    assert adversarial_weight(epoch=0, lambda_adv=1.0, warmup_epochs=5, ramp_epochs=10) == 0.0
    w = adversarial_weight(epoch=7, lambda_adv=1.0, warmup_epochs=5, ramp_epochs=10)
    assert 0.0 < w < 1.0
    assert adversarial_weight(epoch=20, lambda_adv=1.0, warmup_epochs=5, ramp_epochs=10) == 1.0
