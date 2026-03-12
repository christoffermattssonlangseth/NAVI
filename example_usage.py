"""Example usage of NAVI on an AnnData object."""

from __future__ import annotations

import scanpy as sc

from navi import NAVI


def main() -> None:
    """Load data, fit model, and extract latent embeddings."""
    adata = sc.read_h5ad("path/to/xenium_data.h5ad")

    model = NAVI(
        sample_key="sample_id",
        film_key="condition",  # use a biological grouping, not sample_id
        spatial_key="spatial",
        counts_layer="counts",
        k_neighbors=15,
        n_latent_cell=32,
        n_latent_spatial=32,
        normalize_recon_by_genes=True,
        lambda_adv_joint=2.0,
        lambda_sample_align=1.0,
        lambda_film_reg=1e-2,
        film_scale=0.0,
        max_epochs=100,
        accelerator="auto",
        devices=1,
    )
    model.fit(adata)
    latents = model.get_latent(adata)

    print("z_cell:", latents["z_cell"].shape)
    print("z_spatial:", latents["z_spatial"].shape)
    print("z_joint:", latents["z_joint"].shape)
    print("Stored obsm keys:", [key for key in adata.obsm.keys() if key.startswith("X_z_")])


if __name__ == "__main__":
    main()
