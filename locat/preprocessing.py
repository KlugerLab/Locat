import re

import numpy as np
import pandas as pd
import scipy.sparse as sp


#: Default regex patterns for pseudogene-like gene symbols.
DEFAULT_PSEUDO_PATTERNS = [
    r"^Gm\d+",
    r"Rik$",
    r"^AC\d+",
    r"^AA\d+",
    r"^A[0-9]{6,}",
    r"^Mir\d+",
    r"^Rpl\d*-\d+",
    r"^Rps\d*-\d+",
    r"^Linc",
]


def filter_genes(adata, pseudo_patterns=None, min_cell_frac=0.01):
    """Filter genes from an AnnData object.

    Removes pseudogene-like symbols matched by *pseudo_patterns* and genes
    expressed in fewer than *min_cell_frac* of cells.

    Parameters
    ----------
    adata:
        AnnData object whose ``.var_names`` are gene symbols and ``.X`` is an
        expression matrix (dense or sparse).
    pseudo_patterns:
        List of regex patterns identifying pseudogene-like symbols.  Defaults
        to :data:`DEFAULT_PSEUDO_PATTERNS`.
    min_cell_frac:
        Minimum fraction of cells that must express a gene (``X > 0``) for it
        to be retained.  Default is 0.01 (1 %).

    Returns
    -------
    AnnData
        A filtered copy of *adata*.
    """
    if pseudo_patterns is None:
        pseudo_patterns = DEFAULT_PSEUDO_PATTERNS

    pseudo_re = re.compile("|".join(f"(?:{p})" for p in pseudo_patterns))
    is_pseudo = pd.Index(adata.var_names.astype(str)).to_series().str.match(pseudo_re)
    adata = adata[:, ~is_pseudo.values].copy()

    if sp.issparse(adata.X):
        n_cells_expressed = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    else:
        n_cells_expressed = (adata.X > 0).sum(axis=0)

    keep = (n_cells_expressed / adata.n_obs) > min_cell_frac
    return adata[:, keep].copy()


def get_embedding(adata, key="X_pca", n_dims=None):
    """Extract a cell embedding from an AnnData object as a float64 array.

    Parameters
    ----------
    adata:
        AnnData object with the embedding stored in ``obsm``.
    key:
        Key in ``adata.obsm`` to extract (default: ``"X_pca"``).
    n_dims:
        Number of dimensions to keep. If None, all dimensions are returned.

    Returns
    -------
    np.ndarray
        2-D float64 array of shape ``(n_cells, n_dims)``.
    """
    X = adata.obsm[key]
    if n_dims is not None:
        X = X[:, :n_dims]
    return np.asarray(X, dtype=np.float64)
