import numpy as np
import networkx as nx
from tqdm import tqdm
import scipy.sparse as sp
from scipy.stats import hypergeom
import community  # this is the `python-louvain` module

def _discard_small_modules(labels, min_module_size):
    """Relabel communities smaller than *min_module_size* to -1 (in place)."""
    if min_module_size <= 1:
        return labels
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    small = uniq[(counts < min_module_size) & (uniq != -1)]
    if small.size:
        labels = np.where(np.isin(labels, small), -1, labels)
    return labels


def cluster_genes(gene_expression, thresh=0.05, min_module_size=4):
    """
    Cluster localized genes based on overlap of their expression.

    Parameters
    ----------
    gene_expression : AnnData
        Subset of AnnData with only the genes to be clustered (in .var_names).
    thresh : float
        Bonferroni-corrected significance threshold for the pairwise
        hypergeometric test.
    min_module_size : int
        Communities with fewer than this many genes are discarded (relabeled
        to -1). Default is 4, matching the paper's Methods.

    Returns
    -------
    np.ndarray
        Cluster labels for each gene (in order of gene_expression.var_names).
        Genes in discarded (too-small) communities are labeled -1.
    """
    n_genes = gene_expression.shape[1]
    pdist = np.zeros((n_genes, n_genes))

    # Compute pairwise hypergeometric p-values
    for i in tqdm(range(n_genes - 1)):
        for j in range(i + 1, n_genes):
            x0 = gene_expression[:, i].X > 0
            x1 = gene_expression[:, j].X > 0
            x0 = np.asarray(x0).flatten()
            x1 = np.asarray(x1).flatten()
            k = np.sum(x0 & x1)
            M = len(x0)
            n = np.sum(x0)
            N = np.sum(x1)
            pdist[i, j] = 1 - hypergeom.cdf(k, M, n, N)
            pdist[j, i] = pdist[i, j]

    # Bonferroni correction
    n_tests = n_genes * (n_genes - 1) / 2
    threshold = thresh / n_tests

    # Build graph of significant co-expression
    G = nx.Graph()
    G.add_nodes_from(range(n_genes))
    for i in range(n_genes - 1):
        for j in range(i + 1, n_genes):
            if pdist[i, j] < threshold:
                G.add_edge(i, j)

    # Apply Louvain clustering
    partition = community.best_partition(G)

    # Map cluster labels to ordered list
    labels = np.array([partition.get(i, -1) for i in range(n_genes)])
    return _discard_small_modules(labels, min_module_size)


def cluster_genes_fast(
    adata_subset,
    alpha=0.05,
    use_bonferroni=True,
    chunk_size=200000,
    jacc_q=0.90,
    topk=20,
    resolution=2.0,
    lift_min=None,          # e.g. 2.5 or 3.0 to reduce glue; None disables
    weight_mode="loglift",  # "jacc" | "loglift" | "jacc_loglift" | "logp"
    min_module_size=4,      # communities smaller than this are discarded (labeled -1)
):
    """
    Vectorized version of :func:`cluster_genes`, with adaptive Jaccard/lift
    edge filtering, top-k edge pruning, and weighted Louvain clustering.

    Parameters
    ----------
    adata_subset : AnnData
        Subset of AnnData with only the genes to be clustered (in .var_names).
    min_module_size : int
        Communities with fewer than this many genes are discarded (relabeled
        to -1). Default is 4, matching the paper's Methods.

    Returns
    -------
    np.ndarray
        Cluster labels for each gene (in order of adata_subset.var_names).
        Genes in discarded (too-small) communities are labeled -1.
    """
    X = adata_subset.X
    Xb = X.astype(bool).tocsr() if sp.issparse(X) else sp.csr_matrix((X > 0).astype(np.uint8))
    M, G = Xb.shape
    if G <= 1:
        return np.zeros(G, dtype=int)

    n_pos = np.asarray(Xb.getnnz(axis=0)).ravel().astype(np.int64)
    K = (Xb.T @ Xb).astype(np.int32).toarray()

    iu = np.triu_indices(G, k=1)
    k_obs = K[iu].astype(np.int64)
    n_i_vals = n_pos[iu[0]]
    n_j_vals = n_pos[iu[1]]
    valid = (n_i_vals > 0) & (n_j_vals > 0)

    # hypergeom right-tail p-values on valid pairs
    pvals = np.ones_like(k_obs, dtype=float)
    valid_idx = np.where(valid)[0]
    for start in tqdm(range(0, len(valid_idx), chunk_size), desc="hypergeom tests", leave=False):
        sl = valid_idx[start:start + chunk_size]
        pvals[sl] = hypergeom.sf(
            np.maximum(k_obs[sl] - 1, 0),
            M,
            n_i_vals[sl],
            n_j_vals[sl],
        )

    n_tests = G * (G - 1) / 2
    thr = (alpha / n_tests) if use_bonferroni else alpha
    base = (pvals < thr) & valid
    if not np.any(base):
        return -np.ones(G, dtype=int)

    # effect sizes
    jacc = k_obs / (n_i_vals + n_j_vals - k_obs + 1e-9)
    lift = (k_obs * M) / (n_i_vals * n_j_vals + 1e-9)

    # adaptive cutoff among significant edges
    j_cut = np.quantile(jacc[base], jacc_q)
    sig_mask = base & (jacc >= j_cut)
    if lift_min is not None:
        sig_mask &= (lift >= lift_min)

    if not np.any(sig_mask):
        return -np.ones(G, dtype=int)

    # choose weights
    if weight_mode == "jacc":
        w = jacc[sig_mask].astype(np.float32)
    elif weight_mode == "loglift":
        w = np.log1p(lift[sig_mask]).astype(np.float32)
    elif weight_mode == "jacc_loglift":
        w = (jacc[sig_mask] * np.log1p(lift[sig_mask])).astype(np.float32)
    elif weight_mode == "logp":
        w = (-np.log10(np.clip(pvals[sig_mask], 1e-300, 1.0))).astype(np.float32)
        w = np.clip(w, 0, 50)
    else:
        raise ValueError("weight_mode must be one of: jacc, loglift, jacc_loglift, logp")

    rows, cols = iu[0][sig_mask], iu[1][sig_mask]
    A = sp.coo_matrix((w, (rows, cols)), shape=(G, G))
    A = (A + A.T).tocsr()

    # top-k prune
    def topk_prune(A, k=20):
        A = A.tocsr()
        r, c, d = [], [], []
        for i in range(A.shape[0]):
            s, e = A.indptr[i], A.indptr[i+1]
            if s == e:
                continue
            nbrs = A.indices[s:e]
            wts = A.data[s:e]
            if wts.size > k:
                keep = np.argpartition(wts, -k)[-k:]
                nbrs = nbrs[keep]
                wts = wts[keep]
            r.append(np.full(nbrs.size, i, dtype=np.int32))
            c.append(nbrs.astype(np.int32))
            d.append(wts.astype(np.float32))
        if not r:
            return sp.csr_matrix(A.shape)
        r = np.concatenate(r); c = np.concatenate(c); d = np.concatenate(d)
        Ap = sp.coo_matrix((d, (r, c)), shape=A.shape).tocsr()
        return Ap.maximum(Ap.T)

    A = topk_prune(A, k=topk)

    G_nx = nx.from_scipy_sparse_array(A)
    part = community.best_partition(
        G_nx,
        weight="weight",
        resolution=resolution,
        random_state=0
    )

    labels = np.full(G, -1, dtype=int)
    for node, lab in part.items():
        labels[node] = lab
    return _discard_small_modules(labels, min_module_size)