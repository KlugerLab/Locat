import pandas as pd
import scanpy as sc

def _ensure_neighbors_from_pcs(adata, n_neighbors=15, metric="euclidean", key_added=None):
    """
    Use adata.obsm['X_pca'] to build a kNN graph (if not already present).
    Stores:
      - adata.obsp['distances'], adata.obsp['connectivities']
    """
    #if 'X_pca' not in adata.obsm:
    #    raise ValueError("adata.obsm['X_pca'] not found. Run sc.tl.pca or supply PC coords.")

    # If already computed, keep it
    if 'connectivities' in adata.obsp and 'distances' in adata.obsp:
        return

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep='pca',
        method='umap',
        metric=metric,
        key_added=key_added  # None -> use defaults in .obsp
    )

def _laplacian_from_connectivities(W, normalized=True):
    """
    Build (normalized) graph Laplacian from a symmetric affinity/adjacency matrix W (csr).
    L_sym = I - D^{-1/2} W D^{-1/2}
    L = D - W  (if normalized=False)
    """
    if not sp.isspmatrix(W):
        raise ValueError("W must be a scipy sparse matrix")
    # Ensure symmetry (scanpy's connectivities is already symmetrized, but just in case)
    W = 0.5 * (W + W.T)
    d = np.asarray(W.sum(axis=1)).ravel()
    if normalized:
        # L_sym
        d_safe = np.where(d > 0, d, 1.0)
        inv_sqrt_d = 1.0 / np.sqrt(d_safe)
        Dm12 = sp.diags(inv_sqrt_d)
        L = sp.eye(W.shape[0], format='csr') - (Dm12 @ W @ Dm12)
    else:
        # Unnormalized
        D = sp.diags(d)
        L = (D - W).tocsr()
    return L

def _center(X):
    """Center each column (gene) to zero mean."""
    col_means = np.asarray(X.mean(axis=0)).ravel()
    Xc = X - col_means
    return Xc

def _rayleigh_smoothness_all_genes(X_cell_by_gene, L, center=True, eps=1e-12):
    """
    Compute Rayleigh quotient per gene: (x^T L x) / (x^T x)
    Works with sparse L and dense/sparse X. Returns np.array shape (n_genes,).
    """
    # Convert X to dense float64 for fast products; keep memory in mind
    if sp.issparse(X_cell_by_gene):
        X = X_cell_by_gene.toarray().astype(np.float64, copy=False)
    else:
        X = np.asarray(X_cell_by_gene, dtype=np.float64)

    if center:
        X = _center(X)

    # Compute L @ X (cells x genes)
    LX = (L @ X)

    # Numerator per gene: diag(X^T L X) = columnwise dot of X and LX
    num = np.einsum('ij,ij->j', X, LX, optimize=True)

    # Denominator per gene: ||x||^2
    den = np.einsum('ij,ij->j', X, X, optimize=True) + eps

    R = num / den
    # Non-negative by construction; numerical floor at 0
    R = np.maximum(R, 0.0)
    return R

def _spectral_entropy_all_genes(X_cell_by_gene, L_sym, m_eigs=64, center=True, eps=1e-12):
    """
    Approximate spectral entropy using the first m_eigs smallest-eigenvalue eigenvectors of L_sym.
    Returns np.array shape (n_genes,) with values in [0,1] (normalized by log(m_eigs)).
    """
    n = L_sym.shape[0]
    k = min(m_eigs, n - 1)  # keep at least one less than n to skip only-trivial case

    # Compute k smallest eigenpairs (L_sym is SPSD; 'SM' -> smallest magnitude)
    # Note: for large graphs, increase tolerance or use a smaller k.
    evals, evecs = eigsh(L_sym, k=k, which='SM')

    # (Optional) drop the first eigenvector if it's the (near) zero-eigenvalue DC component
    # Detect near-zero eigenvalue
    zero_like = np.isclose(evals[0], 0.0, atol=1e-9)
    U = evecs[:, 1:] if zero_like and k > 1 else evecs
    m = U.shape[1]

    # Prepare gene matrix
    if sp.issparse(X_cell_by_gene):
        X = X_cell_by_gene.toarray().astype(np.float64, copy=False)
    else:
        X = np.asarray(X_cell_by_gene, dtype=np.float64)


    if center:
        X = _center(X)

    # Graph Fourier coefficients for all genes at once: (m x n) @ (n x G) = (m x G)
    Xhat = U.T @ X  # shape (m, n_genes)

    # Power distribution per frequency
    power = Xhat**2
    power_sum = power.sum(axis=0, keepdims=True) + eps  # (1, G)
    p = power / power_sum

    # Entropy per gene (normalize by log(m) -> in [0,1])
    H = -(p * (np.log(p + eps))).sum(axis=0) / np.log(m + eps)
    return np.asarray(H).ravel()

def score_genes_with_graph_metrics(
    adata,
    n_neighbors=15,
    metric="euclidean",
    normalized_laplacian=True,
    center=True,
    m_eigs=64,
    use_connectivities=True,
    layer=None,           # e.g. "log1p" if you keep a log-normalized layer
    clip_negatives=True   # safety: ensure nonnegativity if needed
):
    """
    Adds two columns to adata.var:
      - 'rayleigh_smoothness'
      - 'spectral_entropy'

    Returns a DataFrame with the two scores.
    """
    # 1) Build neighbor graph from PC coords (if missing)
    _ensure_neighbors_from_pcs(adata, n_neighbors=n_neighbors, metric=metric)

    # 2) Choose the weight/adjacency to use
    key = 'connectivities' if use_connectivities else 'distances'
    if key not in adata.obsp:
        raise ValueError(f"adata.obsp['{key}'] not found after neighbors computation.")
    W = adata.obsp[key].tocsr()

    # 3) Laplacian
    L = _laplacian_from_connectivities(W, normalized=normalized_laplacian)

    # 4) Choose expression matrix (cells x genes)
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Requested layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]

    # Optional: enforce nonnegativity if you want (log-normalized should already be >=0)
    if clip_negatives:
        if sp.issparse(X):
            X = X.tocsr(copy=True)
            X.data = np.maximum(X.data, 0.0)
        else:
            X = np.maximum(X, 0.0)

    # 5) Scores
    rayleigh = _rayleigh_smoothness_all_genes(X, L, center=center)
    spectral_H = _spectral_entropy_all_genes(X, L, m_eigs=m_eigs, center=center)

    # 6) Store + return
    adata.var['rayleigh_smoothness'] = rayleigh
    adata.var['spectral_entropy'] = spectral_H
    return adata.var[['rayleigh_smoothness', 'spectral_entropy']].copy()
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# --- Helpers: Laplacian + eigenvectors (reuse from earlier) ---
def laplacian_from_connectivities(W, normalized=True):
    W = 0.5 * (W + W.T)
    d = np.asarray(W.sum(axis=1)).ravel()
    if normalized:
        d_safe = np.where(d > 0, d, 1.0)
        inv_sqrt_d = 1.0 / np.sqrt(d_safe)
        Dm12 = sp.diags(inv_sqrt_d)
        L = sp.eye(W.shape[0], format='csr') - (Dm12 @ W @ Dm12)
    else:
        L = sp.diags(d) - W
    return L.tocsr()

def smallest_eigvecs(L_sym, m_eigs=64):
    evals, evecs = eigsh(L_sym, k=min(m_eigs, L_sym.shape[0]-1), which='SM')
    U = evecs[:, 1:] if np.isclose(evals[0], 0.0, atol=1e-9) and evecs.shape[1] > 1 else evecs
    return U

# --- Scores for many signals at once ---
def rayleigh_batch(X, L, eps=1e-12):
    # X: (n_cells, B), L sparse (n_cells x n_cells)
    LX = L @ X                         # (n, B)
    num = np.sum(X * LX, axis=0)       # (B,)
    den = np.sum(X * X, axis=0) + eps
    R = np.maximum(num / den, 0.0)
    return R

def spectral_entropy_batch(X, U, eps=1e-12):
    # X: (n, B), U: (n, m)
    Y = U.T @ X                        # (m, B)
    power = Y * Y
    psum = np.sum(power, axis=0, keepdims=True) + eps
    p = power / psum
    H = -np.sum(p * (np.log(p + eps)), axis=0) / np.log(U.shape[1] + eps)
    return H

# --- Build size-specific null banks ---
def build_size_nulls(
    adata, W_key='connectivities', layer=None,
    normalized_laplacian=True, m_eigs=96, P=256, seed=0,
    value_pool=None, max_unique_k=200,
):
    rng = np.random.default_rng(seed)
    X = adata.layers[layer] if layer else adata.X
    if sp.issparse(X): X = X.tocsr()
    n_cells, n_genes = X.shape

    # --- Laplacian & eigvecs ---
    W = adata.obsp[W_key].tocsr()
    L = laplacian_from_connectivities(W, normalized=normalized_laplacian)
    U = smallest_eigvecs(L, m_eigs=m_eigs)

    # --- CORRECT per-gene nnz (columns), aligned to var_names ---
    if sp.issparse(X):
        nnz = np.asarray(X.getnnz(axis=0)).ravel()  # <- per column
    else:
        nnz = np.asarray((X > 0).sum(axis=0)).ravel()
    nnz = pd.Series(nnz, index=adata.var_names)     # keep alignment

    # --- value pool for nonzeros ---
    if value_pool is None:
        if sp.issparse(X):
            vals = X.data
            value_pool = vals[vals > 0]
        else:
            value_pool = X[X > 0].ravel()
    value_pool = np.asarray(value_pool, float)
    if value_pool.size == 0:
        raise ValueError("No positive values found to build the null pool.")

    # --- unique k’s (cap/bucket if too many) ---
    unique_k = np.unique(nnz.values)
    unique_k = unique_k[(unique_k > 0) & (unique_k <= n_cells)]  # guardrail
    if unique_k.size == 0:
        raise ValueError("All genes have 0 or >n_cells nonzeros after filtering.")

    if unique_k.size > max_unique_k:
        # bin ks into ~max_unique_k buckets
        bins = np.linspace(unique_k.min(), unique_k.max(), num=max_unique_k+1)
        bins = np.unique(bins.astype(int))
        to_bin = {k: bins[np.searchsorted(bins, k, side='right')-1] for k in unique_k}
        ks_for_null = np.unique(list(to_bin.values()))
    else:
        to_bin = {k: k for k in unique_k}
        ks_for_null = unique_k

    nulls = {}
    for k in ks_for_null:
        # k is guaranteed 1..n_cells
        cols = []
        for _ in range(P):
            idx = rng.choice(n_cells, size=k, replace=False)
            vals = rng.choice(value_pool, size=k, replace=(value_pool.size < k))
            x = np.zeros(n_cells, dtype=float)
            x[idx] = vals
            cols.append(x)
        Xnull = np.stack(cols, axis=1)
        Xnull -= Xnull.mean(axis=0, keepdims=True)   # optional
        Xnull = np.clip(Xnull, 0.0, None)

        R = rayleigh_batch(Xnull, L)
        H = spectral_entropy_batch(Xnull, U)
        nulls[k] = {"R": R, "H": H}

    return {"L": L, "U": U, "nnz": nnz, "nulls": nulls, "k_map": to_bin}

# --- Compare observed scores to null (per gene) ---
def size_matched_pvalues(scores_df, null_bank, two_sided=False):
    nnz = null_bank["nnz"]            # Series indexed by var_names
    k_map = null_bank["k_map"]
    nulls = null_bank["nulls"]

    # align to genes present in nnz/null bank
    genes = scores_df.index.intersection(nnz.index)
    obs = scores_df.loc[genes]
    k_g = np.array([k_map.get(int(nnz.loc[g]), None) for g in genes])
    mask = np.array([k in nulls for k in k_g])
    genes = genes[mask]; k_g = k_g[mask]; obs = obs.loc[genes]

    obs_R = obs['rayleigh_smoothness'].to_numpy()
    obs_H = obs['spectral_entropy'].to_numpy()

    p_R = np.empty_like(obs_R, float); p_H = np.empty_like(obs_H, float)
    pct_R = np.empty_like(obs_R, float); pct_H = np.empty_like(obs_H, float)

    for i, g in enumerate(genes):
        k = k_g[i]
        Rnull = nulls[k]["R"]; Hnull = nulls[k]["H"]
        if two_sided:
            p_R[i] = 2*min((Rnull >= obs_R[i]).mean(), (Rnull <= obs_R[i]).mean())
            p_H[i] = 2*min((Hnull >= obs_H[i]).mean(), (Hnull <= obs_H[i]).mean())
        else:
            p_R[i] = (Rnull <= obs_R[i]).mean()  # smaller = smoother
            p_H[i] = (Hnull <= obs_H[i]).mean()  # lower entropy = more structured
        pct_R[i] = (Rnull < obs_R[i]).mean()
        pct_H[i] = (Hnull < obs_H[i]).mean()

    out = scores_df.copy()
    out.loc[genes, "p_rayleigh_sizeNull"] = p_R
    out.loc[genes, "p_entropy_sizeNull"]  = p_H
    out.loc[genes, "pct_rayleigh_sizeNull"] = pct_R
    out.loc[genes, "pct_entropy_sizeNull"]  = pct_H
    return out