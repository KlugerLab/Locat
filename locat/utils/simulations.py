import anndata as ad
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.datasets import make_blobs


def create_anndata(matrix, cell_names=None, gene_names=None):
    adata = ad.AnnData(matrix)
    adata.obs_names = cell_names or [f"Cell_{i}" for i in range(adata.n_obs)]
    adata.var_names = gene_names or [f"Gene_{i}" for i in range(adata.n_vars)]
    return adata


def simulate_blob_data(
        n_samples: int = 5000,
        n_tests: int = 200,
        n_total: int = 50,
        seed: int = 0,
) -> AnnData:
    coords, clusts, centers = make_blobs(
        n_samples=[n_samples],
        n_features=2,
        centers=None,
        return_centers=True,
        random_state=0,
        cluster_std=[1.]
    )

    rng = np.random.default_rng(seed)

    # ----------------------------
    # Fixed radius, vary in/out fraction
    # ----------------------------
    radius0 = 0.5  # <-- FIXED radius
    fractions_in = np.linspace(1.0, 0.5, n_tests)  # 1.0 => all inside, 0.0 => all outside

    # precompute in/out sets for the fixed radius
    dists = np.sqrt(np.sum((coords - centers[0, :]) ** 2, axis=1))
    in_region = np.flatnonzero(dists < radius0)
    out_region = np.flatnonzero(dists >= radius0)

    genes = np.zeros((coords.shape[0], n_tests), dtype=np.uint8)

    for i, frac_in in enumerate(fractions_in):
        n_in = int(np.round(frac_in * n_total))
        n_out = n_total - n_in

        # cap by availability (just in case)
        n_in = min(n_in, len(in_region))
        n_out = min(n_out, len(out_region))

        # if caps changed total, top up from the other side if possible
        # (keeps total close to n_total when one side is too small)
        cur_total = n_in + n_out
        if cur_total < n_total:
            need = n_total - cur_total
            # try to add to whichever side still has room
            room_in = len(in_region) - n_in
            add_in = min(need, room_in)
            n_in += add_in
            need -= add_in

            room_out = len(out_region) - n_out
            add_out = min(need, room_out)
            n_out += add_out
            need -= add_out

        idx_in = rng.choice(in_region, n_in, replace=False) if n_in > 0 else np.array([], dtype=int)
        idx_out = rng.choice(out_region, n_out, replace=False) if n_out > 0 else np.array([], dtype=int)

        pos_idx = np.concatenate([idx_in, idx_out])
        genes[pos_idx, i] = 1

    adata = create_anndata(genes.astype(np.float64))
    adata.obsm["coords"] = coords.astype(np.float64)
    sc.pp.neighbors(adata, use_rep="coords", n_neighbors=30)
    return adata
