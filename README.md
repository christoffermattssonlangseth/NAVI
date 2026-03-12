# NAVI

Neighborhood-Aware Variational Inference (NAVI) is a hierarchical dual-encoder VAE for multi-sample spatial transcriptomics.  
It combines cell-intrinsic expression and local spatial context while keeping expression latents transferable across samples.

## Core Idea

NAVI learns three components:

- `z_cell`: cell-intrinsic latent from raw counts (NB reconstruction, sample-aware encoder, adversarially sample-invariant representation).
- `z_spatial`: local neighborhood latent from a per-sample kNN spatial graph (GATv2 encoder).
- FiLM conditioning: a learned group embedding (e.g. condition/timepoint) modulates `z_spatial`.

Final latent:

```text
z_joint = concat(z_cell, z_spatial_modulated)
```

## Input Requirements

Use an `AnnData` object with:

- counts in `adata.layers["counts"]` (preferred) or `adata.X`
- `adata.obs["sample_id"]` (or your chosen `sample_key`)
- `adata.obsm["spatial"]` with 2D coordinates
- optional conditioning column for FiLM (recommended): `condition`, `timepoint`, or custom

## Installation

### Option A: pip (existing environment)

```bash
pip install -e .[dev]
```

Optional spatial backend:

```bash
pip install -e .[spatial]
```

### Option B: conda environment (recommended)

```bash
conda create -n NAVI python=3.11 -y
conda activate NAVI
pip install -e .[dev]
pip install scanpy ipykernel
python -m ipykernel install --user --name NAVI --display-name "Python (NAVI)"
```

Note: `torch-geometric` can be platform/CUDA sensitive. If installation fails, install PyTorch and PyG wheels matching your system first, then rerun `pip install -e .`.

## Quickstart

```python
import scanpy as sc
from navi import NAVI

adata = sc.read_h5ad("path/to/data.h5ad")

# Recommended: FiLM groups should be biological cohorts, not sample_id.
if "condition" in adata.obs:
    adata.obs["film_group"] = adata.obs["condition"].astype(str)
elif "timepoint" in adata.obs:
    adata.obs["film_group"] = adata.obs["timepoint"].astype(str)
else:
    adata.obs["film_group"] = "global"

model = NAVI(
    sample_key="sample_id",
    film_key="film_group",
    spatial_key="spatial",
    counts_layer="counts",
    k_neighbors=15,
    n_latent_cell=32,
    n_latent_spatial=32,
    training_mode="joint",  # or "staged"
    max_epochs=60,
    accelerator="auto",     # "mps" / "gpu" / "cpu"
    devices=1,
)

model.fit(adata)
latents = model.get_latent(adata)
```

Stored outputs:

- `adata.obsm["X_z_cell"]`
- `adata.obsm["X_z_spatial"]`
- `adata.obsm["X_z_joint"]`

## API

Main entrypoint:

- `navi.NAVI`
  - `.fit(adata)`
  - `.get_latent(adata)`
  - `.cell_encoder` and `.spatial_encoder` properties for direct encoder access

Diagnostics helper:

- `navi.compute_disentanglement_metrics(embeddings, sample_labels, k=30, alpha=0.05)`

## Integration Tuning (when `z_joint` separates by sample)

If joint embeddings cluster by sample too strongly, start with:

```python
model = NAVI(
    film_key="condition",           # avoid sample_id here
    lambda_adv=1.0,
    lambda_adv_joint=2.0,
    lambda_sample_align=1.0,
    lambda_film_reg=1e-2,
    film_scale=0.0,                 # disable FiLM intensity initially
    normalize_recon_by_genes=True,
    beta2=0.5,
    kl_warmup_epochs=5,
    kl_ramp_epochs=20,
)
```

Then tune:

- increase `lambda_adv_joint` (e.g. `3.0-5.0`) for stronger sample invariance in `z_joint`
- increase `lambda_sample_align` (e.g. `2.0-5.0`) to reduce sample-level centroid drift
- keep `film_key` at condition/timepoint-level unless you explicitly want sample-specific context

## Apple Silicon (MPS)

MPS is supported through PyTorch Lightning:

```python
model = NAVI(accelerator="mps", devices=1)
```

If you hit unsupported ops, set:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Running Tests

```bash
pytest -q
```

## Repository Layout

```text
navi/
  model.py       # NAVI public API
  train.py       # Lightning module + training loop
  encoders.py    # Cell, spatial, FiLM/sample modules
  decoder.py     # Negative binomial decoder
  losses.py      # NB NLL, KL, schedules
  graph.py       # per-sample kNN graph construction
  diagnostics.py # LISI / kBET-style mixing diagnostics
```

## Example Files

- `example_usage.py` for a minimal script
- `output/jupyter-notebook/navi-custom-h5ad.ipynb` for notebook workflow
