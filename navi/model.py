"""Public API for NAVI (Neighborhood-Aware Variational Inference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from anndata import AnnData

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except Exception:  # pragma: no cover - fallback for older installations
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

from .graph import build_spatial_graph
from .train import GraphDataModule, NAVILightningModule, make_train_config, to_device_batch


def _to_dense_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


class NAVI:
    """Neighborhood-Aware Variational Inference model."""

    def __init__(
        self,
        sample_key: str = "sample_id",
        film_key: str | None = None,
        spatial_key: str = "spatial",
        counts_layer: str = "counts",
        k_neighbors: int = 15,
        use_squidpy: bool = True,
        n_latent_cell: int = 32,
        n_latent_spatial: int = 32,
        cell_hidden_dims: tuple[int, ...] = (256, 128),
        spatial_hidden_dim: int = 64,
        gat_heads: int = 4,
        decoder_hidden_dims: tuple[int, ...] = (128,),
        sample_embedding_dim: int = 16,
        dropout: float = 0.1,
        normalize_recon_by_genes: bool = True,
        beta1: float = 1.0,
        beta2: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_adv_joint: float = 0.0,
        lambda_film_reg: float = 0.0,
        lambda_sample_align: float = 0.0,
        film_scale: float = 1.0,
        adv_warmup_epochs: int = 5,
        adv_ramp_epochs: int = 10,
        kl_warmup_epochs: int = 0,
        kl_ramp_epochs: int = 0,
        training_mode: str = "joint",
        cell_pretrain_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: int | str = 1,
        seed: int = 0,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Initialize model and training hyperparameters."""
        self.sample_key = sample_key
        self.film_key = film_key or sample_key
        self.spatial_key = spatial_key
        self.counts_layer = counts_layer
        self.k_neighbors = k_neighbors
        self.use_squidpy = use_squidpy
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.training_mode = training_mode
        self.cell_pretrain_epochs = cell_pretrain_epochs

        self._config_overrides = {
            "n_latent_cell": n_latent_cell,
            "n_latent_spatial": n_latent_spatial,
            "cell_hidden_dims": cell_hidden_dims,
            "spatial_hidden_dim": spatial_hidden_dim,
            "gat_heads": gat_heads,
            "decoder_hidden_dims": decoder_hidden_dims,
            "sample_embedding_dim": sample_embedding_dim,
            "dropout": dropout,
            "normalize_recon_by_genes": normalize_recon_by_genes,
            "beta1": beta1,
            "beta2": beta2,
            "lambda_adv": lambda_adv,
            "lambda_adv_joint": lambda_adv_joint,
            "lambda_film_reg": lambda_film_reg,
            "lambda_sample_align": lambda_sample_align,
            "film_scale": film_scale,
            "adv_warmup_epochs": adv_warmup_epochs,
            "adv_ramp_epochs": adv_ramp_epochs,
            "kl_warmup_epochs": kl_warmup_epochs,
            "kl_ramp_epochs": kl_ramp_epochs,
            "training_mode": training_mode,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }

        self._module: NAVILightningModule | None = None
        self._trainer: pl.Trainer | None = None
        self._sample_to_idx: dict[str, int] | None = None
        self._ordered_samples: list[str] | None = None
        self._film_to_idx: dict[str, int] | None = None
        self._ordered_film: list[str] | None = None
        self.graph_metadata_: dict[str, Any] | None = None

    @property
    def cell_encoder(self) -> torch.nn.Module:
        """Return fitted cell encoder module."""
        if self._module is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._module.cell_encoder

    @property
    def spatial_encoder(self) -> torch.nn.Module:
        """Return fitted spatial encoder module."""
        if self._module is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._module.spatial_encoder

    def fit(self, adata: AnnData) -> "NAVI":
        """Fit the model on AnnData counts and spatial graph."""
        if self.training_mode not in {"joint", "staged"}:
            raise ValueError("training_mode must be either 'joint' or 'staged'.")
        self._validate_adata(adata)
        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)

        batch = self._build_batch(adata, sample_mapping=None)
        n_genes = int(batch["counts"].shape[1])
        n_samples = int(batch["sample_index"].max().item()) + 1
        n_film_groups = int(batch["film_index"].max().item()) + 1

        config = make_train_config(
            n_genes=n_genes,
            n_samples=n_samples,
            n_film_groups=n_film_groups,
            **self._config_overrides,
        )
        module = NAVILightningModule(config=config)
        train_batch = {
            "counts": batch["counts"],
            "sample_index": batch["sample_index"],
            "film_index": batch["film_index"],
            "edge_index": batch["edge_index"],
            "library_size": batch["library_size"],
        }
        data_module = GraphDataModule(batch=train_batch)

        checkpoint_dir = self.checkpoint_dir or str(Path.cwd() / "checkpoints")
        if self.training_mode == "joint":
            module.set_training_phase("joint")
            callbacks = [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="navi-{epoch:02d}-{train_loss:.4f}",
                    monitor="train_loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                )
            ]
            trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=self.devices,
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=True,
                enable_model_summary=True,
            )
            trainer.fit(module, datamodule=data_module)
        else:
            if self.cell_pretrain_epochs < 1 or self.cell_pretrain_epochs >= self.max_epochs:
                raise ValueError(
                    "For training_mode='staged', cell_pretrain_epochs must be >=1 and < max_epochs."
                )
            pretrain_callbacks = [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="navi-pretrain-{epoch:02d}-{train_loss:.4f}",
                    monitor="train_loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                )
            ]
            module.set_training_phase("cell_pretrain")
            pretrain_trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=self.devices,
                max_epochs=self.cell_pretrain_epochs,
                callbacks=pretrain_callbacks,
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=True,
                enable_model_summary=True,
            )
            pretrain_trainer.fit(module, datamodule=data_module)

            joint_callbacks = [
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="navi-joint-{epoch:02d}-{train_loss:.4f}",
                    monitor="train_loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                )
            ]
            module.set_training_phase("joint")
            trainer = pl.Trainer(
                accelerator=self.accelerator,
                devices=self.devices,
                max_epochs=self.max_epochs - self.cell_pretrain_epochs,
                callbacks=joint_callbacks,
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=True,
                enable_model_summary=True,
            )
            trainer.fit(module, datamodule=data_module)

        self._module = module
        self._trainer = trainer
        self._sample_to_idx = batch["sample_to_idx"]
        self._ordered_samples = batch["ordered_samples"]
        self._film_to_idx = batch["film_to_idx"]
        self._ordered_film = batch["ordered_film"]
        self.graph_metadata_ = batch["graph_metadata"]

        # Populate latent embeddings for the training object.
        self.get_latent(adata)
        return self

    @torch.no_grad()
    def get_latent(self, adata: AnnData) -> dict[str, np.ndarray]:
        """Return latent embeddings and store them in AnnData `.obsm`."""
        if self._module is None or self._sample_to_idx is None or self._film_to_idx is None:
            raise RuntimeError("Call fit() before get_latent().")

        batch = self._build_batch(
            adata,
            sample_mapping=self._sample_to_idx,
            film_mapping=self._film_to_idx,
        )
        batch_tensors = {
            "counts": batch["counts"],
            "sample_index": batch["sample_index"],
            "film_index": batch["film_index"],
            "edge_index": batch["edge_index"],
            "library_size": batch["library_size"],
        }
        batch_tensors = to_device_batch(batch_tensors, self._module)
        encoded = self._module.encode_latent(batch_tensors)

        z_cell = encoded["z_cell"].detach().cpu().numpy()
        z_spatial = encoded["z_spatial"].detach().cpu().numpy()
        z_joint = encoded["z_joint"].detach().cpu().numpy()

        adata.obsm["X_z_cell"] = z_cell
        adata.obsm["X_z_spatial"] = z_spatial
        adata.obsm["X_z_joint"] = z_joint
        return {"z_cell": z_cell, "z_spatial": z_spatial, "z_joint": z_joint}

    def _validate_adata(self, adata: AnnData) -> None:
        if self.sample_key not in adata.obs:
            raise KeyError(f"Missing required obs key '{self.sample_key}'.")
        if self.film_key not in adata.obs:
            raise KeyError(f"Missing required obs key '{self.film_key}' for FiLM conditioning.")
        if self.spatial_key not in adata.obsm:
            raise KeyError(f"Missing required obsm key '{self.spatial_key}'.")

        counts_matrix = adata.layers[self.counts_layer] if self.counts_layer in adata.layers else adata.X
        counts = _to_dense_array(counts_matrix)
        if np.any(counts < 0):
            raise ValueError("Counts matrix contains negative values.")
        if counts.shape[0] != adata.n_obs:
            raise ValueError("Counts matrix row count must match number of cells.")

    def _build_batch(
        self,
        adata: AnnData,
        sample_mapping: dict[str, int] | None,
        film_mapping: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        counts_matrix = adata.layers[self.counts_layer] if self.counts_layer in adata.layers else adata.X
        counts = _to_dense_array(counts_matrix)
        library_size = counts.sum(axis=1).astype(np.float32)

        edge_index, sample_index, sample_to_idx, ordered_samples, graph_metadata = build_spatial_graph(
            adata=adata,
            sample_key=self.sample_key,
            spatial_key=self.spatial_key,
            k_neighbors=self.k_neighbors,
            use_squidpy=self.use_squidpy,
        )
        film_values = adata.obs[self.film_key].to_numpy().astype(str)
        if film_mapping is None:
            film_to_idx: dict[str, int] = {}
            ordered_film: list[str] = []
            film_index = np.empty(adata.n_obs, dtype=np.int64)
            for i, label in enumerate(film_values):
                if label not in film_to_idx:
                    film_to_idx[label] = len(film_to_idx)
                    ordered_film.append(label)
                film_index[i] = film_to_idx[label]
        else:
            missing_film = sorted({label for label in film_values if label not in film_mapping})
            if missing_film:
                raise ValueError(
                    "Found unseen FiLM groups in get_latent(): "
                    + ", ".join(missing_film[:5])
                    + (" ..." if len(missing_film) > 5 else "")
                )
            film_to_idx = dict(film_mapping)
            ordered_film = sorted(film_mapping, key=film_mapping.get)
            film_index = np.asarray([film_mapping[label] for label in film_values], dtype=np.int64)

        if sample_mapping is not None:
            sample_ids = adata.obs[self.sample_key].to_numpy().astype(str)
            unknown = sorted({sample for sample in sample_ids if sample not in sample_mapping})
            if unknown:
                raise ValueError(
                    "Found unseen sample_id values in get_latent(): "
                    + ", ".join(unknown[:5])
                    + (" ..." if len(unknown) > 5 else "")
                )
            sample_index = np.asarray([sample_mapping[sample] for sample in sample_ids], dtype=np.int64)
            sample_to_idx = dict(sample_mapping)
            ordered_samples = sorted(sample_mapping, key=sample_mapping.get)

        batch = {
            "counts": torch.as_tensor(counts, dtype=torch.float32),
            "sample_index": torch.as_tensor(sample_index, dtype=torch.long),
            "film_index": torch.as_tensor(film_index, dtype=torch.long),
            "edge_index": edge_index.long(),
            "library_size": torch.as_tensor(library_size, dtype=torch.float32),
            "sample_to_idx": sample_to_idx,
            "ordered_samples": ordered_samples,
            "film_to_idx": film_to_idx,
            "ordered_film": ordered_film,
            "graph_metadata": graph_metadata,
        }
        return batch
