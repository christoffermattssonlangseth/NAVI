"""PyTorch Lightning training components for NAVI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover - fallback for older installations
    import pytorch_lightning as pl

from .decoder import NBDecoder
from .encoders import (
    CellEncoder,
    GradientReversal,
    SampleDiscriminator,
    SampleEmbedding,
    SpatialEncoder,
)
from .losses import adversarial_weight, beta_kl_schedule, kl_standard_normal, negative_binomial_nll


@dataclass
class TrainConfig:
    """Configuration for training and model architecture."""

    n_genes: int
    n_samples: int
    n_film_groups: int
    n_latent_cell: int = 32
    n_latent_spatial: int = 32
    cell_hidden_dims: tuple[int, ...] = (256, 128)
    spatial_hidden_dim: int = 64
    gat_heads: int = 4
    sample_embedding_dim: int = 16
    decoder_hidden_dims: tuple[int, ...] = (128,)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    normalize_recon_by_genes: bool = True
    beta1: float = 1.0
    beta2: float = 1.0
    lambda_adv: float = 1.0
    lambda_adv_joint: float = 0.0
    lambda_film_reg: float = 0.0
    lambda_sample_align: float = 0.0
    film_scale: float = 1.0
    adv_warmup_epochs: int = 5
    adv_ramp_epochs: int = 10
    # KL annealing: effective beta ramps from 0 → beta_max over kl_ramp_epochs
    # after an initial kl_warmup_epochs freeze.  Set both to 0 to disable.
    kl_warmup_epochs: int = 0
    kl_ramp_epochs: int = 0
    training_mode: str = "joint"


class _SingleGraphDataset(Dataset):
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return self.batch


class GraphDataModule(pl.LightningDataModule):
    """Lightning datamodule for full-graph training."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.batch = batch

    def train_dataloader(self) -> DataLoader:
        dataset = _SingleGraphDataset(self.batch)
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])


class NAVILightningModule(pl.LightningModule):
    """Lightning module implementing the hierarchical dual-encoder VAE."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.training_phase = "joint"

        self.cell_encoder = CellEncoder(
            n_genes=config.n_genes,
            n_samples=config.n_samples,
            n_latent=config.n_latent_cell,
            hidden_dims=config.cell_hidden_dims,
            dropout=config.dropout,
        )
        self.spatial_encoder = SpatialEncoder(
            in_dim=config.n_latent_cell,
            n_latent=config.n_latent_spatial,
            hidden_dim=config.spatial_hidden_dim,
            heads=config.gat_heads,
            dropout=config.dropout,
        )
        self.sample_embedding = SampleEmbedding(
            n_samples=config.n_film_groups,
            n_latent_spatial=config.n_latent_spatial,
            embedding_dim=config.sample_embedding_dim,
        )
        self.decoder = NBDecoder(
            n_latent_joint=config.n_latent_cell + config.n_latent_spatial,
            n_genes=config.n_genes,
            n_samples=config.n_samples,
            hidden_dims=config.decoder_hidden_dims,
            dropout=config.dropout,
        )
        self.gradient_reversal = GradientReversal(scale=1.0)
        self.sample_discriminator = SampleDiscriminator(
            n_latent=config.n_latent_cell,
            n_samples=config.n_samples,
        )
        self.joint_discriminator = SampleDiscriminator(
            n_latent=config.n_latent_cell + config.n_latent_spatial,
            n_samples=config.n_samples,
        )

    @staticmethod
    def _group_means(
        values: torch.Tensor,
        group_index: torch.Tensor,
        n_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-group means for dense tensors."""
        group_one_hot = F.one_hot(group_index, num_classes=n_groups).to(values.dtype)
        group_counts = group_one_hot.sum(dim=0)
        group_sums = group_one_hot.transpose(0, 1) @ values
        means = group_sums / group_counts.clamp(min=1.0).unsqueeze(-1)
        return means, group_counts

    def _sample_to_film_alignment_loss(
        self,
        z_joint: torch.Tensor,
        sample_index: torch.Tensor,
        film_index: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize sample-specific centroid drift away from FiLM-group centroids."""
        sample_means, sample_counts = self._group_means(
            z_joint,
            sample_index,
            self.config.n_samples,
        )
        film_means, _ = self._group_means(
            z_joint,
            film_index,
            self.config.n_film_groups,
        )

        sample_one_hot = F.one_hot(sample_index, num_classes=self.config.n_samples).float()
        film_one_hot = F.one_hot(film_index, num_classes=self.config.n_film_groups).float()
        membership = sample_one_hot.transpose(0, 1) @ film_one_hot
        membership_totals = membership.sum(dim=1, keepdim=True)
        film_weights = membership / membership_totals.clamp(min=1.0)
        sample_targets = film_weights @ film_means

        valid_samples = (sample_counts > 0) & (membership_totals.squeeze(-1) > 0)
        if not torch.any(valid_samples):
            return z_joint.new_zeros(())

        diffs = sample_means[valid_samples] - sample_targets[valid_samples]
        weights = sample_counts[valid_samples]
        weights = weights / weights.sum().clamp(min=1.0)
        per_sample = diffs.pow(2).mean(dim=-1)
        return torch.sum(weights * per_sample)

    def set_training_phase(self, phase: str) -> None:
        """Set the active training phase (`joint` or `cell_pretrain`)."""
        if phase not in {"joint", "cell_pretrain"}:
            raise ValueError(f"Unknown training phase '{phase}'.")
        self.training_phase = phase

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run full forward pass and return latent variables and decoder outputs."""
        counts = batch["counts"]
        sample_index = batch["sample_index"]
        film_index = batch.get("film_index", sample_index)
        edge_index = batch["edge_index"]
        library_size = batch["library_size"]

        z_cell, z_cell_mean, z_cell_logvar = self.cell_encoder(counts, sample_index)
        if self.training_phase == "cell_pretrain":
            z_spatial = torch.zeros(
                (z_cell.shape[0], self.config.n_latent_spatial),
                device=z_cell.device,
                dtype=z_cell.dtype,
            )
            z_spatial_mean = torch.zeros_like(z_spatial)
            z_spatial_logvar = torch.zeros_like(z_spatial)
            gamma = torch.ones_like(z_spatial)
            beta = torch.zeros_like(z_spatial)
            z_spatial_modulated = z_spatial
        else:
            z_spatial, z_spatial_mean, z_spatial_logvar = self.spatial_encoder(z_cell, edge_index)
            z_spatial_film, gamma, beta = self.sample_embedding(film_index, z_spatial)
            z_spatial_modulated = z_spatial + self.config.film_scale * (z_spatial_film - z_spatial)
        z_joint = torch.cat([z_cell, z_spatial_modulated], dim=-1)
        mu, theta = self.decoder(z_joint, sample_index, library_size)

        reversed_z = self.gradient_reversal(z_cell)
        sample_logits = self.sample_discriminator(reversed_z)
        reversed_joint = self.gradient_reversal(z_joint)
        joint_sample_logits = self.joint_discriminator(reversed_joint)

        return {
            "z_cell": z_cell,
            "z_cell_mean": z_cell_mean,
            "z_cell_logvar": z_cell_logvar,
            "z_spatial": z_spatial,
            "z_spatial_mean": z_spatial_mean,
            "z_spatial_logvar": z_spatial_logvar,
            "z_spatial_modulated": z_spatial_modulated,
            "gamma": gamma,
            "beta": beta,
            "z_joint": z_joint,
            "mu": mu,
            "theta": theta,
            "sample_logits": sample_logits,
            "joint_sample_logits": joint_sample_logits,
        }

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self.forward(batch)
        counts = batch["counts"]
        sample_index = batch["sample_index"]
        film_index = batch.get("film_index", sample_index)

        recon = negative_binomial_nll(counts, outputs["mu"], outputs["theta"]).mean()
        if self.config.normalize_recon_by_genes:
            recon = recon / float(self.config.n_genes)
        kl_cell = kl_standard_normal(outputs["z_cell_mean"], outputs["z_cell_logvar"]).mean()
        if self.training_phase == "cell_pretrain":
            kl_spatial = torch.zeros_like(kl_cell)
        else:
            kl_spatial = kl_standard_normal(outputs["z_spatial_mean"], outputs["z_spatial_logvar"]).mean()
        adv = F.cross_entropy(outputs["sample_logits"], sample_index)

        if self.training_phase == "cell_pretrain":
            adv_weight = 0.0
        else:
            adv_weight = adversarial_weight(
                epoch=int(self.current_epoch),
                lambda_adv=self.config.lambda_adv,
                warmup_epochs=self.config.adv_warmup_epochs,
                ramp_epochs=self.config.adv_ramp_epochs,
            )
        if self.training_phase == "cell_pretrain":
            adv_joint_weight = 0.0
        else:
            adv_joint_weight = adversarial_weight(
                epoch=int(self.current_epoch),
                lambda_adv=self.config.lambda_adv_joint,
                warmup_epochs=self.config.adv_warmup_epochs,
                ramp_epochs=self.config.adv_ramp_epochs,
            )
        adv_joint = F.cross_entropy(outputs["joint_sample_logits"], sample_index)
        film_reg = ((outputs["gamma"] - 1.0).pow(2) + outputs["beta"].pow(2)).mean()
        if self.training_phase == "cell_pretrain":
            sample_align = torch.zeros_like(kl_cell)
        else:
            sample_align = self._sample_to_film_alignment_loss(
                outputs["z_joint"],
                sample_index,
                film_index,
            )

        # Beta-KL annealing: gradually increase the KL weight from 0 to beta_max
        # to avoid posterior collapse early in training.
        epoch = int(self.current_epoch)
        beta1_eff = beta_kl_schedule(
            epoch,
            beta_max=self.config.beta1,
            warmup_epochs=self.config.kl_warmup_epochs,
            ramp_epochs=self.config.kl_ramp_epochs,
        )
        beta2_eff = beta_kl_schedule(
            epoch,
            beta_max=self.config.beta2,
            warmup_epochs=self.config.kl_warmup_epochs,
            ramp_epochs=self.config.kl_ramp_epochs,
        )
        total = (
            recon
            + beta1_eff * kl_cell
            + beta2_eff * kl_spatial
            + adv_weight * adv
            + adv_joint_weight * adv_joint
            + self.config.lambda_film_reg * film_reg
            + self.config.lambda_sample_align * sample_align
        )

        self.log(f"{stage}_loss", total, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f"{stage}_recon", recon, on_epoch=True, on_step=False)
        self.log(f"{stage}_kl_cell", kl_cell, on_epoch=True, on_step=False)
        self.log(f"{stage}_kl_spatial", kl_spatial, on_epoch=True, on_step=False)
        self.log(f"{stage}_adv", adv, on_epoch=True, on_step=False)
        self.log(f"{stage}_adv_joint", adv_joint, on_epoch=True, on_step=False)
        self.log(f"{stage}_film_reg", film_reg, on_epoch=True, on_step=False)
        self.log(f"{stage}_sample_align", sample_align, on_epoch=True, on_step=False)
        self.log("adv_weight", float(adv_weight), on_epoch=True, on_step=False)
        self.log("adv_joint_weight", float(adv_joint_weight), on_epoch=True, on_step=False)
        self.log("beta1_eff", float(beta1_eff), on_epoch=True, on_step=False)
        self.log("beta2_eff", float(beta2_eff), on_epoch=True, on_step=False)
        return total

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        """Compute and return training loss."""
        return self._shared_step(batch, stage="train")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    @torch.no_grad()
    def encode_latent(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode deterministic latent embeddings using posterior means.

        BUG FIX: the previous implementation called self.forward() and returned
        reparameterized *samples* (z_cell, z_spatial).  Samples are stochastic —
        running get_latent() twice would produce different embeddings, making
        downstream analysis (UMAP, clustering) non-reproducible.

        We now explicitly compute the posterior means and apply FiLM to the
        spatial mean, giving fully deterministic embeddings at inference time.
        """
        self.eval()
        counts = batch["counts"]
        sample_index = batch["sample_index"]
        film_index = batch.get("film_index", sample_index)
        edge_index = batch["edge_index"]

        # Use posterior means — not stochastic reparameterised samples.
        _, z_cell_mean, _ = self.cell_encoder(counts, sample_index)

        if self.training_phase == "joint":
            # Run spatial encoder on the cell mean for consistent determinism.
            _, z_spatial_mean, _ = self.spatial_encoder(z_cell_mean, edge_index)
            # Apply FiLM to the spatial mean (not a noisy sample).
            z_spatial_film, _, _ = self.sample_embedding(film_index, z_spatial_mean)
            z_spatial_modulated = z_spatial_mean + self.config.film_scale * (
                z_spatial_film - z_spatial_mean
            )
        else:
            # cell_pretrain: spatial component is not yet trained.
            z_spatial_modulated = torch.zeros(
                (z_cell_mean.shape[0], self.config.n_latent_spatial),
                device=z_cell_mean.device,
                dtype=z_cell_mean.dtype,
            )

        z_joint = torch.cat([z_cell_mean, z_spatial_modulated], dim=-1)
        return {
            "z_cell": z_cell_mean,
            "z_spatial": z_spatial_modulated,
            "z_joint": z_joint,
        }


def to_device_batch(batch: dict[str, torch.Tensor], module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Move tensors in batch to the same device as module."""
    device = next(module.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def make_train_config(n_genes: int, n_samples: int, n_film_groups: int, **kwargs: Any) -> TrainConfig:
    """Create a training config from keyword overrides."""
    config = TrainConfig(n_genes=n_genes, n_samples=n_samples, n_film_groups=n_film_groups)
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown training config key '{key}'.")
        setattr(config, key, value)
    return config
