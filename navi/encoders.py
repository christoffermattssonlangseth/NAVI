"""Encoder modules for NAVI."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch_geometric.nn import GATv2Conv


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))
        in_features = hidden_dim
    return nn.Sequential(*layers)


class CellEncoder(nn.Module):
    """Variational encoder for cell-intrinsic expression features."""

    def __init__(
        self,
        n_genes: int,
        n_samples: int,
        n_latent: int = 32,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        batch_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.batch_embedding = nn.Embedding(n_samples, batch_embedding_dim)
        self.backbone = _build_mlp(n_genes + batch_embedding_dim, hidden_dims, dropout)
        final_dim = hidden_dims[-1] if hidden_dims else (n_genes + batch_embedding_dim)
        self.z_mean = nn.Linear(final_dim, n_latent)
        self.z_logvar = nn.Linear(final_dim, n_latent)

    def forward(self, counts: Tensor, sample_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode raw counts into latent cell state parameters."""
        x = torch.log1p(counts)
        sample_emb = self.batch_embedding(sample_index)
        h = self.backbone(torch.cat([x, sample_emb], dim=-1))
        z_mean = self.z_mean(h)
        z_logvar = self.z_logvar(h)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
        """Sample from a Gaussian with reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class SpatialEncoder(nn.Module):
    """Graph attention variational encoder for local spatial context."""

    def __init__(
        self,
        in_dim: int,
        n_latent: int = 32,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gat_1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.gat_2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.z_mean = nn.Linear(hidden_dim, n_latent)
        self.z_logvar = nn.Linear(hidden_dim, n_latent)

    def forward(self, z_cell: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode graph-aggregated context from cell latents."""
        h = self.gat_1(z_cell, edge_index)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.gat_2(h, edge_index)
        h = self.activation(h)
        z_mean = self.z_mean(h)
        z_logvar = self.z_logvar(h)
        z = CellEncoder.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar


class SampleEmbedding(nn.Module):
    """Learned sample embeddings with FiLM modulation heads."""

    def __init__(
        self,
        n_samples: int,
        n_latent_spatial: int,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_samples, embedding_dim)
        self.gamma_head = nn.Linear(embedding_dim, n_latent_spatial)
        self.beta_head = nn.Linear(embedding_dim, n_latent_spatial)

    def forward(self, sample_index: Tensor, z_spatial: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply sample-specific FiLM to spatial latents."""
        u_sample = self.embedding(sample_index)
        gamma = 1.0 + self.gamma_head(u_sample)
        beta = self.beta_head(u_sample)
        z_spatial_modulated = gamma * z_spatial + beta
        return z_spatial_modulated, gamma, beta


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx: Function, x: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.scale * grad_output, None


class GradientReversal(nn.Module):
    """Layer that reverses gradients for adversarial training."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: Tensor) -> Tensor:
        """Return identity in forward and reversed gradients in backward."""
        return _GradientReversalFn.apply(x, self.scale)


class SampleDiscriminator(nn.Module):
    """Predict sample labels from latent embeddings."""

    def __init__(
        self,
        n_latent: int,
        n_samples: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_latent
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_samples))
        self.network = nn.Sequential(*layers)

    def forward(self, z_cell: Tensor) -> Tensor:
        """Return logits over samples."""
        return self.network(z_cell)
