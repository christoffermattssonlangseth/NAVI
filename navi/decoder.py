"""Decoder modules for negative-binomial reconstruction."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class NBDecoder(nn.Module):
    """Negative-binomial decoder for count reconstruction."""

    def __init__(
        self,
        n_latent_joint: int,
        n_genes: int,
        n_samples: int,
        hidden_dims: Sequence[int] = (128,),
        dropout: float = 0.1,
        batch_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.batch_embedding = nn.Embedding(n_samples, batch_embedding_dim)

        layers: list[nn.Module] = []
        in_dim = n_latent_joint + batch_embedding_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.scale_head = nn.Linear(in_dim, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        z_joint: Tensor,
        sample_index: Tensor,
        library_size: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode latent features into NB mean and dispersion."""
        sample_emb = self.batch_embedding(sample_index)
        x = torch.cat([z_joint, sample_emb], dim=-1)
        h = self.backbone(x) if len(self.backbone) else x
        scale_logits = self.scale_head(h)
        px_scale = torch.softmax(scale_logits, dim=-1)
        lib = library_size.unsqueeze(-1).clamp(min=1.0)
        mu = px_scale * lib
        theta = torch.exp(self.log_theta).unsqueeze(0).expand_as(mu)
        return mu, theta
