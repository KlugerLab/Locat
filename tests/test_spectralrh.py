import unittest
import numpy as np

from locat.spectralrh import score_genes_with_graph_metrics
from locat.utils.simulations import simulate_blob_data


class SpectralRHTestCase(unittest.TestCase):

    def test_score_genes_with_graph_metrics(self):
        adata = simulate_blob_data(n_samples=500, n_tests=5, n_total=50)

        scores = score_genes_with_graph_metrics(adata, m_eigs=16)

        # both scores are reported for every gene
        self.assertEqual(list(scores.columns), ['rayleigh_smoothness', 'spectral_entropy'])
        self.assertEqual(len(scores), adata.n_vars)

        # results are also stashed on adata.var
        np.testing.assert_array_equal(scores['rayleigh_smoothness'].to_numpy(),
                                       adata.var['rayleigh_smoothness'].to_numpy())
        np.testing.assert_array_equal(scores['spectral_entropy'].to_numpy(),
                                       adata.var['spectral_entropy'].to_numpy())

        # rayleigh quotient is non-negative by construction (clipped)
        self.assertTrue((scores['rayleigh_smoothness'] >= 0).all())

        # normalized spectral entropy lies in [0, 1]
        self.assertTrue((scores['spectral_entropy'] >= 0).all())
        self.assertTrue((scores['spectral_entropy'] <= 1 + 1e-8).all())


if __name__ == '__main__':
    unittest.main()
