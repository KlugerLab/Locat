import numpy as np
from scipy.stats import multivariate_normal as mnorm
from dataclasses import dataclass


@dataclass
class WGMM:
    pis: np.ndarray
    mus: np.ndarray
    sigmas: np.ndarray

    @property
    def n_comp(self) -> int:
        return len(self.pis)

    def pdf(self, coords):
        pdf = np.zeros(shape=(coords.shape[0],))
        for i in range(self.n_comp):
            c0 = mnorm(mean=self.mus[i], cov=self.sigmas[i])
            pdf += self.pis[i] * c0.pdf(coords)
        return pdf

    def loglikelihood_by_component(self, coords, weights):
        # estimate the pdf of each component separately
        logpdf = np.zeros(shape=(coords.shape[0], self.n_comp))
        for i in range(self.n_comp):
            c0 = mnorm(mean=self.mus[i], cov=self.sigmas[i])
            logpdf[:, i]= c0.logpdf(coords) * weights

        # estimate the log-likelihood of each component
        return logpdf

    def mahalanobis_dist(self, coords):
        """
        Compute Mahalanobis distance from each point to each component's peak (vectorized).

        Parameters:
        -----------
        coords : np.ndarray
            Input coordinates array with shape (n_points, n_dimensions)

        Returns:
        --------
        np.ndarray
            Mahalanobis distances with shape (n_points, n_components)
            Each row represents a point, each column represents a component
        """
        n_points = coords.shape[0]
        distances = np.zeros((n_points, self.n_comp))

        for i in range(self.n_comp):
            # Get mean and covariance for component i
            mean = self.mus[i]
            cov = self.sigmas[i]

            # Compute inverse of covariance matrix with regularization
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Add small regularization term for numerical stability
                reg_cov = cov + np.eye(cov.shape[0]) * 1e-6
                inv_cov = np.linalg.inv(reg_cov)

            # Vectorized computation for all points
            diff = coords - mean[None,:]  # Broadcasting: (n_points, n_dims) - (n_dims,)

            # Compute quadratic form: (x-μ)ᵀ Σ⁻¹ (x-μ) for each point
            quad_form = np.sum((diff @ inv_cov) * diff, axis=1)
            distances[:, i] = np.sqrt(quad_form)

        return distances

    def loglikelihood_truncated(self, coords, weights, top_n_components=None):
        if top_n_components is None:
            top_n_components = self.n_comp

        # speeding up
        coords = coords[weights>0]
        weights = weights[weights>0]

        # top n components by number of explained points
        logpdf = self.loglikelihood_by_component(coords=coords, weights=weights)

        # Get indices of top n components using argpartition (efficient)
        if top_n_components<self.n_comp:
            top_n_indices = np.argpartition(np.sum(logpdf, axis=0), top_n_components)[-top_n_components:]
        else:
            top_n_indices = np.arange(top_n_components)


        # Sort by descending count
        top_n_pis = self.pis[top_n_indices]/np.sum(self.pis[top_n_indices])
        return np.log(np.sum(np.exp(np.sum(logpdf[:, top_n_indices], axis=0)) * top_n_pis))
