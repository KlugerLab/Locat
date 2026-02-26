import math

import numba
import numpy as np
from loguru import logger
from scanpy import AnnData
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp
from scipy.stats import binom, betabinom, chi2
from sklearn.metrics.pairwise import euclidean_distances as edist
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from locat.wgmm import WGMM
from locat.wgmms import wgmm
from locat.rgmm import softbootstrap_gmm

class LOCATNullDistribution:
    mean = None
    std = None

    def __init__(self, mean_func, std_func):
        self.mean = mean_func
        self.std = std_func

    def to_zscore(self, raw_score, p):
        return (raw_score - self.mean(p)) / self.std(p)

    @classmethod
    def from_estimates(cls, p, means, stds):
        return cls(
            mean_func=PchipInterpolator(p, means),
            std_func=PchipInterpolator(p, stds),
        )


class LOCAT:
    """

    The main LOCAT class

    """
    _cell_dist = None
    _min_dist = None
    _knn = None
    _background_gmm: WGMM = None
    _background_pdf: np.ndarray = None
    _background_logpdf: np.ndarray = None
    _null_distribution: LOCATNullDistribution = None
    _X = None
    _n_components_waypoints = None
    _disable_progress_info = True

    def __init__(
        self,
        adata: AnnData,
        cell_embedding: np.ndarray,
        k: int,
        n_bootstrap_inits: int,
        show_progress=False,
        wgmm_dtype: str = "same",  # "same" | "float32" | "float64"
        knn=None,                 # <-- NEW: precomputed adjacency/connectivities
        knn_k: int | None = None, # <-- NEW: if computing, how many neighbors (defaults to k)
        knn_mode: str = "binary", # <-- NEW: "binary" or "connectivity"
    ):
        self._disable_progress_info = not show_progress
        self._adata = adata

        emb = np.asarray(cell_embedding)
        emb = (emb - emb.mean(0)) / (emb.std(0) + emb.dtype.type(1e-6))
        self._embedding = emb
        self._dtype = self._embedding.dtype
        self._wgmm_dtype = wgmm_dtype

        self._k = k
        self.n_cells, self.n_genes = self._adata.shape
        self.n_dims = self._embedding.shape[1]
        self.n_bootstrap_inits = n_bootstrap_inits
        self._knn = None
        self._knn_k = int(knn_k) if knn_k is not None else int(k)
        self._knn_mode = str(knn_mode)

        self._reg_covar = None
        if knn is not None:
            self.set_knn(knn)
    # ------------------------------------------------------------------
    # Basic geometry / regularization
    # ------------------------------------------------------------------
    @property
    def cell_dist(self) -> np.ndarray:
        if self._cell_dist is None:
            if not self._disable_progress_info:
                logger.info("recomputing cell-cell distance")
            self._cell_dist = edist(self._embedding)
        return self._cell_dist

    @property
    def min_dist(self) -> float:
        if self._min_dist is None:
            if not self._disable_progress_info:
                logger.info("recomputing min cell-cell distance")
            self._min_dist = np.mean(
                np.nanmin(
                    np.where(self.cell_dist > 0, self.cell_dist, np.nan),
                    axis=1,
                )
            )
        return self._min_dist

    def reg_covar(self, sample_size=None):
        if self._reg_covar is not None:
            return self._dtype.type(self._reg_covar)

        base = (self.min_dist ** (2 / self.n_dims)) / 6
        if sample_size is None:
            rc = base
        else:
            c = 1 - 2 / self.n_dims
            adj = np.sqrt((self.n_cells + c) / (sample_size + c))
            rc = base * adj

        rc = max(rc, 1e-4)
        return self._dtype.type(rc)

    # ------------------------------------------------------------------
    # Data access / options
    # ------------------------------------------------------------------
    @property
    def X(self):
        if self._X is None:
            X = self._adata.X
            if not isinstance(X, np.ndarray):
                X = X.toarray()
            self._X = X.astype(self._dtype, copy=False)
        return self._X

    def show_progress(self, show_progress=True):
        self._disable_progress_info = not show_progress

    # ------------------------------------------------------------------
    # Background GMM and LTST null
    # ------------------------------------------------------------------
    def background_n_components_init(self, weights_transform=None, min_points=10, n_reps=30):
        if not self._disable_progress_info:
            logger.info("Estimating number of GMM components")

        self._n_components_waypoints = np.zeros(shape=(3, 2))
        self._n_components_waypoints[0, 0] = min_points
        self._n_components_waypoints[1, 0] = int(np.sqrt(self.n_cells))
        self._n_components_waypoints[2, 0] = self.n_cells

        self._n_components_waypoints[0, 1] = 1

        bic_component_cost = self.n_dims * (self.n_dims + 3) / 2

        if weights_transform is not None:
            Xdense = self.X.copy()
            for i in range(self.n_genes):
                Xdense[:, i] = weights_transform(Xdense[:, i])
            weights = np.asarray(Xdense.sum(axis=1), dtype=self._dtype)
        else:
            weights = np.ones((self.n_cells,), dtype=self._dtype)

        weights = np.clip(weights, self._dtype.type(1e-6), np.inf).astype(
            self._dtype, copy=False
        )
        s = float(weights.sum())
        weights = (weights / s) if s > 0 else np.full_like(weights, 1.0 / len(weights))
        weights[~np.isfinite(weights)] = self._dtype.type(1.0 / len(weights))

        # Full-data BIC search for big-N waypoint
        min_search = 1
        max_search = 10
        keep_searching = True
        bic = []
        while keep_searching:
            for i in tqdm(
                range(min_search, max_search),
                desc=f"estimating BIC for {self._n_components_waypoints[2, 0]:.0f} cells",
                position=0,
                leave=True,
                disable=self._disable_progress_info,
            ):
                p = self.fit_wgmm(n_comp=i, weights=weights).pdf(self._embedding)
                bic.append(
                    bic_component_cost * i * np.log(self.n_cells)
                    - np.sum(2 * np.log(p[p > 0]))
                )
            self._n_components_waypoints[2, 1] = np.argmin(bic) + 1
            if self._n_components_waypoints[2, 1] >= len(bic) - 2:
                min_search = max_search
                max_search = max_search + 5
            else:
                keep_searching = False

        # sqrt(n_cells) waypoint
        n = np.max([min_points + 1, int(np.ceil(np.sqrt(self.n_cells)))])
        bic = np.zeros(shape=(int(self._n_components_waypoints[2, 1]), n_reps))

        o = np.arange(self.n_cells)
        wgs = np.zeros(shape=(self.n_cells, n_reps))
        for i_rep in range(n_reps):
            np.random.shuffle(o)
            wgs[o < n, i_rep] = weights[o < n]
        wgs = wgs / np.sum(wgs, axis=0)

        for i_rep in tqdm(
            range(n_reps),
            desc=f"estimating BIC for {self._n_components_waypoints[1, 0]:.0f} cells",
            position=0,
            leave=True,
            disable=self._disable_progress_info,
        ):
            for i in range(1, bic.shape[0] + 1):
                p = self.fit_wgmm(n_comp=i, weights=wgs[:, i_rep]).pdf(self._embedding)
                bic[i - 1, i_rep] = (
                    bic_component_cost * i * np.log(n) - np.sum(2 * np.log(p[p > 0]))
                )
        self._n_components_waypoints[1, 1] = np.argmin(np.median(bic, axis=1)) + 1

    def auto_bkg_components(self, n_points, weights_transform=None):
        if self._n_components_waypoints is None:
            self.background_n_components_init(weights_transform=weights_transform)

        n_components = np.interp(
            np.log(n_points),
            xp=np.log(self._n_components_waypoints[:, 0]),
            fp=np.log(self._n_components_waypoints[:, 1]),
        )
        return int(np.round(np.exp(n_components)))

    def _auto_n_effective_weights(self, weights=None, min_dist_cutoff=None):
        if min_dist_cutoff is None:
            min_dist_cutoff = self.min_dist * 3

        rweights = np.zeros(shape=(self.n_cells,))
        if weights is None:
            wg0 = rweights == 0
        else:
            wg0 = weights > 0

        if np.sum(wg0) > 5:
            # only compute the components to describe the cells that are close enough
            cell_include = (
                np.min(self.cell_dist[wg0, :][:, wg0] + np.eye(np.sum(wg0)) * 10, axis=1)
                < min_dist_cutoff
            )
            rweights[wg0] = cell_include.astype(float)
            return rweights
        return None

    def auto_n_components(self, coords, weights=None, indices=None, min_points_fraction=0.95):
        if weights is None:
            weights = np.ones(shape=(coords.shape[0],))

        if indices is None:
            if coords.shape[0] != self.n_cells:
                raise ValueError("A selection was made but the indices were not passed")
            indices = np.full_like(weights, fill_value=True)

        logpdf = self._background_logpdf[indices, :] * weights[:, None]
        id_component = np.argmax(logpdf, axis=1)

        component_counts = np.bincount(
            id_component[weights > 0], minlength=self._background_gmm.n_comp
        )
        n_counts = np.sum(component_counts)
        component_counts[component_counts < 5] = 0
        if np.sum(component_counts) < 10:
            return 1
        else:
            component_counts = component_counts / np.sum(component_counts)
            estimated_n_components = (
                np.min(
                    np.flatnonzero(
                        np.cumsum(np.sort(component_counts)[::-1]) > min_points_fraction
                    )
                )
                + 1
            )
            estimated_n_components = int(np.floor(np.sqrt(estimated_n_components + 1)) + 1)
            return int(
                np.max(
                    [
                        1,
                        np.min(
                            [
                                self._background_gmm.n_comp,
                                n_counts / 5,
                                estimated_n_components,
                            ]
                        ),
                    ]
                )
            )
    def set_knn(self, knn):
        """
        Store a precomputed KNN graph.
    
        Accepts:
          - scipy sparse (csr/csc/coo) adjacency or connectivities
          - dense numpy array adjacency/connectivities
    
        Expected shape: (n_cells, n_cells)
        """
        if sp.issparse(knn):
            K = knn.tocsr()
        else:
            K = np.asarray(knn)
    
        if K.shape != (self.n_cells, self.n_cells):
            raise ValueError(f"knn must have shape {(self.n_cells, self.n_cells)}, got {K.shape}")
    
        # zero diagonal to avoid self-neighbor artifacts
        if sp.issparse(K):
            K = K.tolil()
            K.setdiag(0.0)
            K = K.tocsr()
            K.eliminate_zeros()
        else:
            np.fill_diagonal(K, 0.0)
    
        self._knn = K
    
    
    def knn(self):
        """
        Return a KNN adjacency/connectivity matrix.
    
        If a KNN was provided at init (or via set_knn), returns it.
        Otherwise computes one from the embedding.
        """
        if self._knn is not None:
            return self._knn
    
        k = int(self._knn_k)
        if k <= 0:
            raise ValueError("knn_k must be >= 1")
    
        # Build kNN from embedding (exclude self by taking k+1 then dropping self)
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, self.n_cells), metric="euclidean")
        nbrs.fit(self._embedding)
        dists, inds = nbrs.kneighbors(self._embedding, return_distance=True)
    
        # Drop self neighbor (usually first column)
        inds = inds[:, 1: k + 1]
    
        rows = np.repeat(np.arange(self.n_cells), inds.shape[1])
        cols = inds.reshape(-1)
    
        if self._knn_mode == "binary":
            data = np.ones_like(cols, dtype=self._dtype)
        elif self._knn_mode == "connectivity":
            # simple distance-based weights; you can swap for something else
            dd = dists[:, 1: k + 1].reshape(-1)
            data = (1.0 / (dd + 1e-6)).astype(self._dtype, copy=False)
        else:
            raise ValueError("knn_mode must be 'binary' or 'connectivity'")
    
        K = sp.csr_matrix((data, (rows, cols)), shape=(self.n_cells, self.n_cells))
        K.eliminate_zeros()
        self._knn = K
        return self._knn


    def background_pdf(
        self,
        n_comp=None,
        reps=10,
        total_counts_weight=True,
        weights_transform=None,
        force_refresh=False,
    ):
        if (self._background_pdf is None) or force_refresh:
            if not self._disable_progress_info:
                logger.info("fitting background PDF")

            if n_comp is None:
                if self._n_components_waypoints is None:
                    self.background_n_components_init(weights_transform=weights_transform)
                n_comp = np.interp(
                    np.log(self.n_cells),
                    xp=np.log(self._n_components_waypoints[:, 0]),
                    fp=np.log(self._n_components_waypoints[:, 1]),
                )
                n_comp = int(np.round(np.exp(n_comp)))
                if not self._disable_progress_info:
                    logger.info(f"Using {n_comp} components")

            if weights_transform is not None:
                Xdense = self.X.copy()
                for i in range(self.n_genes):
                    Xdense[:, i] = weights_transform(Xdense[:, i])
                weights = np.asarray(Xdense.sum(axis=1), dtype=self._dtype)
            else:
                weights = np.ones((self.n_cells,), dtype=self._dtype)

            weights = np.clip(weights, self._dtype.type(1e-6), np.inf).astype(
                self._dtype, copy=False
            )
            s = float(weights.sum())
            weights = (weights / s) if s > 0 else np.full_like(weights, 1.0 / len(weights))
            weights[~np.isfinite(weights)] = self._dtype.type(1.0 / len(weights))

            self._background_pdf = np.zeros(shape=(self.n_cells,))
            self._background_logpdf = np.zeros(shape=(self.n_cells, n_comp))

            background_gmm = None
            for _ in tqdm(
                range(reps),
                desc="fitting background",
                position=0,
                leave=True,
                disable=self._disable_progress_info,
            ):
                background_gmm = self.fit_wgmm(n_comp, weights=weights)
                self._background_pdf += background_gmm.pdf(self._embedding)
                self._background_logpdf += background_gmm.loglikelihood_by_component(
                    self._embedding, np.ones(shape=self.n_cells)
                )
            self._background_gmm = background_gmm
            self._background_pdf /= reps
            self._background_logpdf /= reps

            self.estimate_null_parameters()
        return self._background_pdf

    def signal_gmm(self, weights, n_comp=None):
        if n_comp is None:
            comp_weights = self._auto_n_effective_weights(weights)
            if comp_weights is None:
                n_comp = 1
            else:
                loc_indices = comp_weights > 0
                n_comp = self.auto_n_components(
                    self._embedding[loc_indices, :],
                    weights[loc_indices],
                    indices=loc_indices,
                )

        return self.fit_wgmm(n_comp, weights=weights)

    def fit_wgmm(self, n_comp, weights=None) -> WGMM:
        if weights is None:
            weights = np.ones(shape=(self.n_cells,))

        pis, mus, sigmas, _ = wgmm(
            self._embedding,
            raw_weights=weights,
            n_components=n_comp,
            n_inits=1,
            reg_covar=self.reg_covar(),
        )
        pis = np.array(pis)
        mus = np.array(mus)
        sigmas = np.array(sigmas)
        return WGMM(pis, mus, sigmas)

    def estimate_null_parameters(self, fractions=None, n_reps=50):
        """
        Estimate LTST null mean/std as a function of expression fraction p
        using random pseudo-genes:
          - pick random expressing cells at frequency p
          - fit signal GMM with the *same* pipeline as real genes
          - compute LTST exactly as in gmm_scan_new
        """
        if fractions is None:
            fractions = 10 ** np.linspace(np.log10(10 / self.n_cells), 0, 7)

        f0 = self.background_pdf()
        scores = []

        for frac in tqdm(
            fractions,
            desc="null distribution parameters (perm. pseudo-genes)",
            position=0,
            leave=True,
            disable=self._disable_progress_info,
        ):
            n_pos = max(5, int(round(frac * self.n_cells)))
            if n_pos >= self.n_cells:
                n_pos = self.n_cells - 1

            ltst_vals = []

            for _ in range(n_reps):
                mask = np.zeros(self.n_cells, dtype=bool)
                mask[np.random.choice(self.n_cells, n_pos, replace=False)] = True
                gene_prior = mask.astype(self._dtype)

                comp_gene_prior = self._auto_n_effective_weights(gene_prior)
                if comp_gene_prior is None:
                    n_comp = 1
                else:
                    loc_indices = comp_gene_prior > 0
                    n_comp = self.auto_n_components(
                        self._embedding[loc_indices, :],
                        gene_prior[loc_indices],
                        loc_indices,
                    )
                gmm1 = self.signal_gmm(weights=gene_prior, n_comp=n_comp)

                i1 = gene_prior > 0
                loc_f1 = gmm1.pdf(self._embedding[i1, :])
                p1 = float(np.mean(i1))

                ltst_score = ltst_score_func(f0[i1], loc_f1, p1)

                w_expr = gene_prior[i1]
                w_expr = w_expr / (w_expr.sum() if w_expr.sum() > 0 else 1.0)
                ltst_vals.append(float(np.dot(w_expr, ltst_score)))

            ltst_vals = np.asarray(ltst_vals, dtype=float)
            scores.append(
                [
                    frac,
                    float(np.mean(ltst_vals)),
                    float(np.std(ltst_vals) + 1e-9),
                ]
            )

        scores = np.asarray(scores)
        self._null_distribution = LOCATNullDistribution.from_estimates(
            p=scores[:, 0],
            means=scores[:, 1],
            stds=scores[:, 2],
        )

    # ------------------------------------------------------------------
    # Depletion-style localization scan
    # ------------------------------------------------------------------
    def localization_pval_dep_scan(
        self,
        gmm1,
        gene_prior,
        *,
        c_values=None,
        soft_bound=None,  # default computed from n: max((n-1)/n, 0.99)
        min_p0_abs=0.10,
        min_expected=30,
        min_abs_deficit=0.02,
        n_trials_cap=500,
        weight_mode="amount",
        p_floor=1e-12,
        n_eff_scale=1.0,
        rho_bb=0.2,  # >0 enables Beta–Binomial tail
        eps_rel=0.01,
        debug=False,
        debug_store_masks=False,
        debug_max_cells=5000,
    ):
        if self._background_gmm is None:
            _ = self.background_pdf()

        X = self._embedding
        n = int(X.shape[0])
        if soft_bound is None:
            soft_bound = max((n - 1) / max(n, 1), 0.99)

        gp = np.asarray(gene_prior, float)
        if weight_mode == "binary":
            w_obs = (gp > 0).astype(float)
        elif weight_mode == "amount":
            w_obs = np.clip(gp, 0.0, np.inf)
        else:
            raise ValueError("weight_mode must be 'amount' or 'binary'")

        # Kish n_eff with tempering + cap
        sw = float(np.sum(w_obs))
        n_eff = 0.0 if sw <= 0 else (sw * sw) / max(float(np.sum(w_obs * w_obs)), 1e-12)
        n_eff *= float(n_eff_scale)
        if n_trials_cap is not None:
            n_eff = min(n_eff, float(n_trials_cap))
        n_eff = max(1.0, n_eff)
        n_trials = int(round(n_eff))

        f0_x = np.clip(self._background_gmm.pdf(X), 1e-300, np.inf)
        f1_x = np.clip(gmm1.pdf(X), 1e-300, np.inf)
        w0 = f0_x / float(np.sum(f0_x))

        w_obs_alpha = w_obs / (sw if sw > 0 else 1.0)
        w0_alpha = w0

        if c_values is None:
            c_values = np.concatenate([[1.0], np.geomspace(1.05, 3.0, 12)])

        best_logp = None
        best = {"c": None, "k_obs": None, "p0": None, "obs_prop": None}
        scanned = 0
        tested = 0

        per_c = [] if debug else None

        for c in c_values:
            c = float(c)
            in_R_mask = f0_x > c * f1_x

            p0_abs = float(np.sum(w0_alpha * in_R_mask))
            reason = None
            if p0_abs < min_p0_abs:
                reason = "fail:min_p0_abs"
            if n_eff * p0_abs < min_expected:
                reason = "fail:min_expected"
            if p0_abs > soft_bound:
                reason = "fail:soft_bound"

            obs_prop = float(np.sum(w_obs_alpha * in_R_mask))
            if reason is None:
                if (p0_abs - obs_prop) < min_abs_deficit:
                    reason = "fail:min_abs_deficit"
                elif obs_prop > (p0_abs / c) * (1.0 - float(eps_rel)):
                    reason = "fail:c_bound" #this checks whether observed f1 density in the region is at an equal or lower proportion than the observed f0 density in the region * c (where c is the contrast). If the region is larger than expectation, reject the gene.

            if debug:
                rec = {
                    "c": c,
                    "p0_abs": p0_abs,
                    "obs_prop": obs_prop,
                    "n_eff": float(n_eff),
                    "n_eff_expected": float(n_eff * p0_abs),
                    "reason": reason,
                }
                if debug_store_masks:
                    ncap = min(int(debug_max_cells), n)
                    rec["in_R_idx"] = np.flatnonzero(in_R_mask[:ncap]).astype(int)
                    if weight_mode == "binary":
                        expr_mask = gp > 0
                        rec["expr_in_R_count"] = int(np.sum(expr_mask & in_R_mask))
                per_c.append(rec)

            if reason is not None:
                continue

            scanned += 1
            tested += 1

            k_eff = int(np.rint(obs_prop * n_trials))
            p0_clip = np.clip(p0_abs, 1e-12, 1 - 1e-12)

            if rho_bb and rho_bb > 0.0:
                ab_sum = max(1.0 / float(rho_bb) - 1.0, 2.0)
                alpha = float(p0_clip * ab_sum)
                beta = float((1.0 - p0_clip) * ab_sum)
                p_raw = float(betabinom.cdf(k_eff, n_trials, alpha, beta))
                logp_raw = np.log(max(p_raw, np.finfo(float).tiny))
            else:
                lFkm1 = binom.logcdf(k_eff - 1, n_trials, p0_clip)
                lpk = binom.logpmf(k_eff, n_trials, p0_clip) + np.log(0.5)
                logp_raw = logsumexp([lFkm1, lpk])

            if (best_logp is None) or (logp_raw < best_logp):
                best_logp = float(logp_raw)
                best.update(
                    {"c": c, "k_obs": k_eff, "p0": p0_abs, "obs_prop": obs_prop}
                )

        if tested == 0:
            out = {
                "p_value": 1.0,
                "raw_min_p": 1.0,
                "log_p_single": 0.0,
                "log_p_sidak": 0.0,
                "neglog10_p_single": 0.0,
                "neglog10_p_sidak": 0.0,
                "best_c": None,
                "k_obs_eff": None,
                "p0_abs": None,
                "obs_prop": None,
                "scanned": int(scanned),
                "tested": int(tested),
                "sidak_penalty": 1,
                "n": n,
                "n_eff": float(n_eff),
                "guards": {
                    "min_p0_abs": float(min_p0_abs),
                    "min_expected": float(min_expected),
                    "min_abs_deficit": float(min_abs_deficit),
                    "soft_bound": float(soft_bound),
                    "eps_rel": float(eps_rel),
                },
                "n_eff_scale": float(n_eff_scale),
                "rho_bb": float(rho_bb),
            }
            if debug:
                out["per_c"] = per_c
            return out

        m_eff = tested
        sidak_logp = _logsidak_from_logp(best_logp, m_eff)

        p_value = float(np.exp(sidak_logp))
        raw_min_p = float(np.exp(best_logp))
        p_value = _safe_p(max(p_value, p_floor))
        raw_min_p = _safe_p(max(raw_min_p, p_floor))

        out = {
            "p_value": p_value,
            "raw_min_p": raw_min_p,
            "log_p_single": float(best_logp),
            "log_p_sidak": float(sidak_logp),
            "neglog10_p_single": float(-best_logp / np.log(10)),
            "neglog10_p_sidak": float(-sidak_logp / np.log(10)),
            "best_c": best["c"],
            "k_obs_eff": best["k_obs"],
            "p0_abs": best["p0"],
            "obs_prop": best["obs_prop"],
            "scanned": int(scanned),
            "tested": int(tested),
            "sidak_penalty": 1,
            "n": n,
            "n_eff": float(n_eff),
            "guards": {
                "min_p0_abs": float(min_p0_abs),
                "min_expected": float(min_expected),
                "min_abs_deficit": float(min_abs_deficit),
                "soft_bound": float(soft_bound),
                "eps_rel": float(eps_rel),
            },
            "n_eff_scale": float(n_eff_scale),
            "rho_bb": float(rho_bb),
        }
        if debug:
            out["per_c"] = per_c
        return out

    # ------------------------------------------------------------------
    # Main scan used in practice
    # ------------------------------------------------------------------
    def bic_score(self, gmm1, gene_prior):
        bic_component_cost = self.n_dims * (self.n_dims + 3) / 2
        p = gmm1.pdf(self._embedding[gene_prior > 0, :])
        n_cells = np.sum(gene_prior > 0)
        bic = (
            bic_component_cost * gmm1.n_comp * np.log(np.sum(gene_prior > 0))
            - 2 * np.sum(p[p > 0])
        )
        return bic / n_cells

    def gmm_scan_new(
        self,
        genes=None,
        weights_transform=None,
        zscore_thresh=None,
        max_freq=0.5,
        verbose=False,
        n_bootstrap_inits=None,
        # Depletion-scan defaults
        rc_c_values=None,  # default inside method
        rc_min_p0_abs=0.10, #minimum proportion of f0 density in depleted region required for the region pval to be estimated
        rc_min_expected=30, #minimum expected cells in depleted region required for the region pval to be estimated
        rc_min_abs_deficit=0.02, #minimum absolute difference in f1(x) - f0(x) for all x in depleted region
        rc_n_trials_cap=500, #maximum effective sample size
        rc_soft_bound=1.0, #this is unused/can be removed
        rc_n_eff_scale=0.5, #scaling factor for effective sample sizes -- can be tweaked to stabilize pvalues across various gene sample sizes
        rc_p_floor=1e-12, # this is just model precision, can be ignored
        rc_rho_bb=0.02, #this is the strength of the beta binomial (0.0 is standard binomial, set at 0.02-0.05 for wider tails)
        rc_weight_mode="amount",
        rc_eps_rel=0.01,
    ):
        if verbose:
            print("gmm_scan_new: using depletion scan for localization_pval (localization_pval_dep_scan)")

        if n_bootstrap_inits is not None:
            self.n_bootstrap_inits = int(n_bootstrap_inits)

        locally_enriched = dict()
        gzeros, freqzeros, zzeros = [], [], []
        inclgenes = self.get_genes_indices(genes)

        f0 = self.background_pdf(weights_transform=weights_transform)
        for i_gene in tqdm(
            inclgenes,
            desc="scanning genes",
            position=0,
            leave=True,
            disable=self._disable_progress_info,
        ):
            gene_prior = self.get_gene_prior(i_gene, weights_transform)
            try:
                if np.sum(gene_prior) == 0:
                    gzeros.append(self._adata.var_names[i_gene])
                    continue
                if np.mean(gene_prior > 0) > max_freq:
                    freqzeros.append(self._adata.var_names[i_gene])
                    continue

                comp_gene_prior = self._auto_n_effective_weights(gene_prior)
                if comp_gene_prior is None:
                    n_comp = 1
                else:
                    loc_indices = comp_gene_prior > 0
                    n_comp = self.auto_n_components(
                        self._embedding[loc_indices, :],
                        gene_prior[loc_indices],
                        loc_indices,
                    )
                gmm1 = self.signal_gmm(weights=gene_prior, n_comp=n_comp)

                i1 = gene_prior > 0
                loc_f1 = gmm1.pdf(self._embedding[i1, :])
                p1 = np.mean(i1)
                sample_size = p1 * self.n_cells

                ltst_score = ltst_score_func(f0[i1], loc_f1, p1)
                sens_score = sens_score_func(f0[i1], loc_f1, i1[i1])

                zscore = np.dot(gene_prior[i1], ltst_score) / np.sum(gene_prior[i1])
                zscore = self._null_distribution.to_zscore(zscore, p1)
                if (zscore_thresh is not None) and (zscore < zscore_thresh):
                    zzeros.append(self._adata.var_names[i_gene])
                    continue

                cs_res = self.localization_pval_dep_scan(
                    gmm1,
                    gene_prior,
                    debug=True,
                    c_values=(
                        rc_c_values
                        if rc_c_values is not None
                        else np.concatenate([[1.0], np.geomspace(1.05, 3.0, 12)])
                    ),
                    soft_bound=rc_soft_bound,
                    min_p0_abs=rc_min_p0_abs,
                    min_expected=rc_min_expected,
                    min_abs_deficit=rc_min_abs_deficit,
                    n_trials_cap=rc_n_trials_cap,
                    weight_mode=rc_weight_mode,
                    p_floor=rc_p_floor,
                    n_eff_scale=rc_n_eff_scale,
                    rho_bb=rc_rho_bb,
                    eps_rel=rc_eps_rel,
                )
                localization_pval = _safe_p(cs_res["p_value"])

                concentration_pval = _safe_p(float(normal_sf(zscore, 0.0, 1.0)))

                p_cauchy = cauchy_combine([localization_pval, concentration_pval])
                p_size = _safe_p(1.0 - np.exp(-1.0 / (sample_size + 1.0)))
                p_sens = _safe_p(1.0 - (sens_score + 1e-9))
                p_final = 1.0 - (1.0 - p_cauchy) * (1.0 - 0.05 * p_size) * (
                    1.0 - 0.12 * p_sens
                )
                p_final = float(smooth_qvals(np.array([_safe_p(p_final)]))[0])

                locally_enriched[self._adata.var_names[i_gene]] = {
                    "bic": self.bic_score(gmm1, gene_prior),
                    "zscore": zscore,
                    "sens_score": sens_score,
                    "localization_pval": localization_pval,
                    "concentration_pval": concentration_pval,
                    "pval": p_final,
                    "n_components": n_comp,
                    "sample_size": sample_size,
                    "depl_scan": cs_res,
                }

            except ValueError as e:
                if verbose:
                    logger.info(e)

        if verbose:
            print("gzeros:", len(gzeros), "freqzeros:", len(freqzeros), "zzeros:", len(zzeros))
        return locally_enriched

    # ------------------------------------------------------------------
    # Small helpers still used by gmm_scan_new
    # ------------------------------------------------------------------
    def get_genes_indices(self, genes):
        inclgenes = range(self.n_genes)
        list_genes = self._adata.var_names.tolist()
        if genes is not None:
            inclgenes = [list_genes.index(i) for i in genes]
        return inclgenes

    def get_gene_prior(self, i_gene, weights_transform):
        gene_prior = self.X[:, i_gene]
        if weights_transform is not None:
            gene_prior = weights_transform(gene_prior)
        return gene_prior
    
    def signal_pdf(self, weights, n_comp=None):
        gmm = self.signal_gmm(weights=weights, n_comp=n_comp, )
        return gmm.pdf(self._embedding)

    def gmm_loglikelihoodtest(self, genes=None, weights_transform=None, max_freq=0.5):
        res = dict()
        bkg_df = self.auto_bkg_components(self.n_cells)
        log_bkg_pdf = self.background_pdf(weights_transform=weights_transform)
        bkg_pdf_gt0 = log_bkg_pdf > 0
        log_bkg_pdf = np.where(bkg_pdf_gt0, np.log(log_bkg_pdf), 0)

        inclgenes = self.get_genes_indices(genes)

        for i_gene in tqdm(inclgenes, desc=f'scanning genes',
                           position=0, leave=True, disable=self._disable_progress_info):
            gene_prior = self.get_gene_prior(i_gene, weights_transform)
            gene_prior_gt0 = gene_prior > 0
            n_gene_prior = np.sum(gene_prior_gt0)
            pv = 1.0
            df = None
            lr = None
            n_comp = 0
            sample_size = None

            if (n_gene_prior > 5) & ((n_gene_prior / len(gene_prior)) < max_freq):
                comp_gene_prior = self._auto_n_effective_weights(gene_prior)
                if comp_gene_prior is None:
                    n_comp = 1
                else:
                    loc_indices = comp_gene_prior > 0
                    n_comp = self.auto_n_components(
                        self._embedding[loc_indices, :],
                        gene_prior[loc_indices],
                        indices=loc_indices,  # <-- required
                    )
                
        
                f1 = self.signal_pdf(weights=gene_prior, n_comp=n_comp)

                df = (bkg_df - n_comp) + 1
                ix = gene_prior_gt0 & (f1 > 0)
                if np.sum(ix) > 5:
                    ix = ix & bkg_pdf_gt0
                    sample_size = np.sum(ix)

                    if sample_size == 0:
                        pv = 0

                    elif sample_size > 5:
                        lr = self.calc_lratio(f1, ix, log_bkg_pdf, sample_size)
                        pv = 1 - chi2.cdf(lr, df)

            res[self._adata.var_names[i_gene]] = {
                'llratio_pvalue': pv,
                'llratio_sample_size': sample_size,
                'llratio_stat': lr,
                'llratio_df': df,
                'n_comp': n_comp,
            }

        return res

    def gmm_local_pvalue(
        self,
        genes=None,
        n_comp=None,
        weights_transform=None,
        alpha=0.05,
        n_inits=100,
        normalize_knn=True,
        eps=1e-12,
    ):
        from scipy.stats import rankdata
        from sklearn.metrics import roc_curve
        import scipy.sparse as sp
        import numpy as np
    
        f0 = np.asarray(self.background_pdf(weights_transform=weights_transform), dtype=self._dtype).ravel()
        K = self.knn()
    
        # Optional: row-normalize K so every cell has comparable neighborhood "mass"
        if normalize_knn:
            if sp.issparse(K):
                deg = np.asarray(K.sum(axis=1)).ravel()
                inv = 1.0 / np.maximum(deg, eps)
                Kuse = sp.diags(inv).dot(K)
            else:
                deg = K.sum(axis=1)
                Kuse = K / np.maximum(deg[:, None], eps)
        else:
            Kuse = K
    
        locally_enriched = {}
        inclgenes = self.get_genes_indices(genes)
    
        for i_gene in tqdm(
            inclgenes,
            desc="scanning genes",
            position=0,
            leave=True,
            disable=self._disable_progress_info,
        ):
            gene_prior = self.get_gene_prior(i_gene, weights_transform)
            gp = np.asarray(gene_prior, dtype=self._dtype).ravel()
    
            sw = float(gp.sum())
            if sw <= 0:
                continue  # or store null result if you prefer
    
            # signal pdf + LTST
            f1 = np.asarray(self.signal_pdf(weights=gp, n_comp=n_comp), dtype=self._dtype).ravel()
            p = float(np.mean(gp > 0))
            ltst_score = ltst_score_func(f0, f1, p)  # (n_cells,)
    
            zscore = float(np.dot(gp, ltst_score) / sw)
    
            # random pseudo-genes (must return f2: (n_cells, n_inits), rweights: (n_cells, n_inits))
            f2, rweights = self.random_pdf(weights=gp, n_comp=n_comp, n_inits=n_inits)
            f2 = np.asarray(f2, dtype=self._dtype)
            rweights = np.asarray(rweights, dtype=self._dtype)
    
            # p2 is per-init expression fraction
            p2 = np.mean(rweights > 0, axis=0).astype(self._dtype, copy=False)  # (n_inits,)
    
            ltst_score2 = ltst_score_func(f0[:, None], f2, p2)  # (n_cells, n_inits)
    
            # empirical zscore p-value (avoid 0/1 extremes)
            z2 = (np.sum(rweights * ltst_score2, axis=0) / sw).astype(np.float64)  # (n_inits,)
            z_p = (1.0 + np.sum(z2 >= zscore)) / (n_inits + 1.0)
    
            # neighborhood-smoothed statistics
            if sp.issparse(Kuse):
                wstat1 = np.asarray(Kuse.dot(ltst_score)).ravel()
                wstat2 = np.asarray(Kuse.dot(ltst_score2))
            else:
                wstat1 = (Kuse @ ltst_score).ravel()
                wstat2 = (Kuse @ ltst_score2)
    
            # empirical p-values per cell: rank among [wstat1, wstat2...]
            # shape: (n_cells, n_inits+1)
            M = np.concatenate([wstat1[:, None], wstat2], axis=1)
            ranks = rankdata(-M, axis=1, method="average")[:, 0]  # rank of observed in each row
            emp_p = ranks / (n_inits + 1.0)  # in (0,1]
    
            # pick a cutoff from ROC if there is signal
            empirical_h1 = emp_p < (alpha / self.n_cells)
    
            wstat1_cutoff = 0.0
            if np.any(empirical_h1) and np.any(~empirical_h1):
                fpr, tpr, cuts = roc_curve(empirical_h1.astype(int), wstat1)
                # maximize balanced accuracy ( (tpr + (1-fpr)) / 2 )
                j = np.argmax(((1 - fpr) + tpr) / 2.0)
                wstat1_cutoff = float(cuts[j])
    
            local_res = {
                "wstat_cutoff": wstat1_cutoff,
                "wstat_alpha": float(alpha),
                "wstat_pvalues": emp_p.astype(np.float32, copy=False),
                "zscore": float(zscore),
                "zscore_pvalue": float(z_p),
                "wstat_repetitions": int(n_inits),
            }
    
            sig_idx = np.flatnonzero(wstat1 > wstat1_cutoff)
            local_res["wstat_significant"] = sig_idx if sig_idx.size else None
            local_res["wstat_significant_clusters"] = None  # placeholder
    
            locally_enriched[self._adata.var_names[i_gene]] = local_res
    
        return locally_enriched

    
    def gmm_local_scan(self,
                       genes=None,
                       weights_transform=None,
                       zscore_thresh=None,
                       max_freq=0.5):

        locally_enriched = dict()
        if zscore_thresh is None:
            zscore_thresh = 1.0
        K = self.knn()
        
        if sp.issparse(K):
            knn_neis = np.asarray(K.sum(axis=1)).ravel()
            inv = 1.0 / np.maximum(knn_neis, 1e-12)
            # row-normalize: D^{-1} K
            knn_norm = sp.diags(inv).dot(K)
        else:
            knn_neis = K.sum(axis=1)
            knn_norm = K / np.maximum(knn_neis[:, None], 1e-12)

        inclgenes = self.get_genes_indices(genes)

        f0 = self.background_pdf(weights_transform=weights_transform)
        f0 = np.asarray(f0, dtype=self._dtype)
        for i_gene in tqdm(inclgenes, desc=f'scanning genes',
                           position=0, leave=True, disable=self._disable_progress_info):
            gene_prior = self.get_gene_prior(i_gene, weights_transform)
            try:
                if np.sum(gene_prior) == 0:
                    continue

                if np.mean(gene_prior > 0) > max_freq:
                    continue

                comp_gene_prior = self._auto_n_effective_weights(gene_prior)
                if comp_gene_prior is None:
                    n_comp = 1
                else:
                    loc_indices = comp_gene_prior > 0
                    n_comp = self.auto_n_components(
                        self._embedding[loc_indices, :],
                        gene_prior[loc_indices],
                        indices=loc_indices,          # <-- THIS is the missing piece
                    )


                f1 = self.signal_pdf(weights=gene_prior, n_comp=n_comp)
                expr = (gene_prior > 0).astype(self._dtype, copy=False)
                
                # p1 = local expression fraction around each cell
                if sp.issparse(knn_norm):
                    p1 = knn_norm.dot(expr)
                else:
                    p1 = knn_norm @ expr
                p1 = np.asarray(p1, dtype=self._dtype).ravel()
                f1 = np.asarray(f1, dtype=self._dtype).ravel()
                # compute ltst per-cell then smooth by knn
                ltst = ltst_score_func(
                    np.asarray(f0, dtype=self._dtype),
                    np.asarray(f1, dtype=self._dtype),
                    p1,
                )
                
                if sp.issparse(knn_norm):
                    local_zscore = knn_norm.dot(ltst)
                else:
                    local_zscore = knn_norm @ ltst
                local_zscore = np.asarray(local_zscore, dtype=self._dtype).ravel()

                local_zscore = self._null_distribution.to_zscore(local_zscore, p1)
                
                K = self.knn()
                expr_mask = (gene_prior > 0).astype(np.float32)           # (n,)
                sig_mask  = (f1 > f0).astype(np.float32)                 # (n,)
                
                if sp.issparse(K):
                    expr_neighbors = K.dot(expr_mask)                    # how many expressing neighbors (weighted)
                    sig_expr_neighbors = K.dot(expr_mask * sig_mask)     # how many expressing-and-signal neighbors
                else:
                    expr_neighbors = K @ expr_mask
                    sig_expr_neighbors = K @ (expr_mask * sig_mask)
                
                local_lscore = sig_expr_neighbors / np.maximum(expr_neighbors, 1e-12)
                local_lscore = np.asarray(local_lscore).ravel()
                
                                
                # rate
                # local_zscore = local_zscore * knn_neis / self.n_cells
                assert K.shape == (self.n_cells, self.n_cells), (K.shape, self.n_cells)
                assert f0.shape == (self.n_cells,), (f0.shape, self.n_cells)
                assert f1.shape == (self.n_cells,), (f1.shape, self.n_cells)


                localization_pval = localization_pvalue_nn_func(gene_prior, f1, f0, K)
                concentration_pval = norm.sf(local_zscore)
                #localization_pval = 1.0
                #concentration_pval = norm.sf(local_zscore)


                if np.any(local_zscore > zscore_thresh):
                    locally_enriched[self._adata.var_names[i_gene]] = {
                        'enriched': np.flatnonzero(local_zscore > zscore_thresh),
                        'local_zscore': local_zscore,
                        'local_lscore': local_lscore,
                        'localization_pval': localization_pval,
                        'concentration_pval': concentration_pval,
                        'local_pvalue': localization_pval + concentration_pval - localization_pval*concentration_pval,
                        'n_components': n_comp
                    }
            except Exception as e:
                logger.exception(e)   # prints full traceback
                raise                # optional: stop on first failure

        return locally_enriched

    @staticmethod
    def calc_lratio(f1, ix, log_bkg_pdf, sample_size, eps=1e-300):
        """
        Per-cell LRT contribution on expressing cells:
          -2 * sum_{i in ix} ( log f0(i) - log f1(i) ) / sample_size
        """
        # ensure 1D arrays
        f1 = np.asarray(f1).ravel()
        ix = np.asarray(ix, dtype=bool).ravel()
        log_bkg_pdf = np.asarray(log_bkg_pdf).ravel()
    
        # log f1 safely (no mutation)
        log_f1 = np.log(np.clip(f1, eps, np.inf))
    
        denom = float(sample_size) if sample_size > 0 else 1.0
        return float((-2.0 * np.sum((log_bkg_pdf[ix] - log_f1[ix])) ) / denom)

    def random_pdf(
        self,
        weights,
        n_comp=None,
        n_inits=300,
        buckets=None,
    ):
        import numpy as np
        from scipy.stats import multivariate_normal as mnorm  # <-- add back
    
        weights = np.asarray(weights, dtype=self._dtype).ravel()
    
        if n_comp is None:
            comp_weights = self._auto_n_effective_weights(weights)
            if comp_weights is None:
                n_comp = 1
            else:
                loc_indices = comp_weights > 0
                n_comp = self.auto_n_components(
                    self._embedding[loc_indices, :],
                    weights[loc_indices],
                    indices=loc_indices,          # <-- IMPORTANT
                )
    
        pis, mus, sigmas, rweights = softbootstrap_gmm(
            self._embedding,
            raw_weights=weights,
            n_components=n_comp,
            reg_covar=self.reg_covar(),
            n_inits=n_inits,
            buckets=buckets,
        )
    
        pis = np.asarray(pis)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        rweights = np.asarray(rweights)
    
        dtot = np.zeros((self.n_cells, n_inits), dtype=np.float64)
    
        # NOTE: this loop is slow but correct; optimize later if needed
        for j in range(n_inits):
            for i in range(n_comp):
                w = pis[j, i]
                if w == 0:
                    continue
                c0 = mnorm(mean=mus[j, i], cov=sigmas[j, i], allow_singular=True)
                dtot[:, j] += w * c0.pdf(self._embedding)
    
        return dtot.astype(self._dtype, copy=False), rweights.astype(self._dtype, copy=False)


# ----------------------------------------------------------------------
# JIT-accelerated scoring helpers
# ----------------------------------------------------------------------
@numba.jit(nopython=True)
def ltst_score_func(f0, f1, p):
    q = np.sqrt(p)
    return (q * (1 - q)) * (f1 - f0) / (f1 + f0)


@numba.jit(nopython=True)
def sens_score_func(f0, f1, i):
    return np.mean(f1[i > 0] > f0[i > 0])

import numpy as np
import scipy.sparse as sp

def localization_pvalue_nn_func(x1, f1, f0, nn):
    """
    Sparse-safe rewrite of the original localization_pvalue_nn_func.

    Preserves:
      - i1/obs1/obs2 definitions
      - n and o as neighborhood (weighted) counts / signed balance
      - f2 weighting for global mu_hat, p1, p0
      - effective_n, sd_hat
      - per-cell p via normal_sf(o[i], mu_hat, sd_hat[i])

    Works for:
      - nn sparse CSR/CSC/COO
      - nn dense numpy array
    """
    x1 = np.asarray(x1)
    f1 = np.asarray(f1)
    f0 = np.asarray(f0)

    # boolean masks
    i1 = (x1 > 0)
    obs1 = ((f1 > f0) & i1)
    obs2 = ((f1 < f0) & i1)

    # convert to float vectors for dot products
    i1_f   = i1.astype(np.float32)
    obs1_f = obs1.astype(np.float32)
    obs2_f = obs2.astype(np.float32)

    # neighborhood weighted counts
    if sp.issparse(nn):
        # ensure CSR for fast dot
        nn = nn.tocsr()
        n = nn.dot(i1_f)
        o = nn.dot(obs1_f) - nn.dot(obs2_f)
    else:
        n = nn @ i1_f
        o = (nn @ obs1_f) - (nn @ obs2_f)

    n = np.asarray(n).ravel()
    o = np.asarray(o).ravel()

    # normalize signed balance
    o = o / np.clip(n, 1.0, np.inf)

    # same f2 + global mu_hat logic as original
    f2 = 2.0 * (f1 * f0) / np.clip(f1 + f0, 1e-12, np.inf)

    denom = float(np.sum(f2))
    if denom <= 0 or not np.isfinite(denom):
        # degenerate fallback: no information => p=1 everywhere
        return np.ones(shape=(nn.shape[0],), dtype=np.float64)

    p1 = float(np.sum((f1 > f0) * f2) / denom)
    p0 = float(np.sum((f1 < f0) * f2) / denom)
    mu_hat = p1 - p0

    effective_n = np.clip(n * (1.0 + 2.0 * np.abs(o / 2.0)), 1.0, np.inf)
    sd_hat = np.sqrt(((p1 + p0) - ((p1 - p0) ** 2)) / effective_n)

    # compute p only where n>0 (same behavior as your loop)
    p = np.ones(shape=(nn.shape[0],), dtype=np.float64)
    idx = np.flatnonzero(n > 0)

    # vectorized normal_sf (your normal_sf is scalar-numba; keep loop or vectorize in numpy)
    # We'll keep the loop for correctness with your numba normal_sf signature.
    for i in idx:
        p[i] = normal_sf(float(o[i]), mu=mu_hat, sigma=float(sd_hat[i]))

    return p

    
@numba.njit
def normal_sf(x, mu, sigma):
    """
    Normal survival function SF = 1 - CDF, numba-jitted.
    """
    z = (x - mu) / sigma
    return 0.5 * math.erfc(z * 0.7071067811865476)  # = 1 - Phi(z)


def _logsidak_from_logp(logp_min: float, m_eff: int) -> float:
    """
    Sidák combine in log-space:
      log p_sidák = log(1 - (1 - p_min)^m_eff)
                  = log(1 - exp(m_eff * log(1 - p_min)))
    with log(1 - p_min) = log1p(-exp(logp_min)).
    """
    if m_eff <= 1:
        return float(logp_min)
    l1mp = np.log1p(-np.exp(logp_min))
    return float(np.log1p(-np.exp(m_eff * l1mp)))


@numba.njit
def smooth_qvals(x):
    return np.where(
        x <= 0.99,
        x,
        1 - 0.01 / (1 + 100 * (x - 0.99) + 10000 * (x - 0.99) ** 2),
    )


def _safe_p(p, eps=1e-15):
    if not np.isfinite(p):
        return 1.0 - eps
    return float(min(max(p, eps), 1.0 - eps))


def cauchy_combine(pvals, weights=None):
    """
    Robust p-value combiner for dependent tests (Cauchy combination).
    Liu & Xie (2020).
    """
    ps = np.array([_safe_p(p) for p in pvals], dtype=np.float64)
    if weights is None:
        weights = np.ones_like(ps)
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() if w.sum() > 0 else 1.0)

    t = np.sum(w * np.tan((0.5 - ps) * np.pi))
    pc = 0.5 - np.arctan(t) / np.pi
    return _safe_p(pc)


def summarize_rc_debug(cs_res, top=8):
    """
    Convenience helper to inspect per-threshold diagnostics from
    localization_pval_dep_scan(..., debug=True).
    """
    if "per_c" not in cs_res:
        print("No per-threshold diagnostics captured. Run with debug=True.")
        return
    rows = cs_res["per_c"]
    if not rows:
        print("No thresholds scanned.")
        return
    from collections import Counter

    reasons = Counter(r.get("reason") for r in rows)
    print("Reason counts:", dict(reasons))

    cand = [r for r in rows if r.get("reason") is not None]
    cand.sort(key=lambda r: r["n_eff_expected"], reverse=True)
    print(f"\nTop {min(top, len(cand))} failing thresholds by n_eff_expected:")
    for r in cand[:top]:
        print(
            f"  c={r['c']:.3f}  p0_abs={r['p0_abs']:.3f}  "
            f"obs_prop={r['obs_prop']:.3f}  n_eff_expected={r['n_eff_expected']:.1f}  "
            f"reason={r['reason']}"
        )
