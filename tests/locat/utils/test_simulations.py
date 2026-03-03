import unittest
import jax
import numpy as np

from locat.locat import LOCAT
from locat.utils.simulations import simulate_blob_data


class AlgorithmTestCase(unittest.TestCase):
    def test_jax_devices(self):
        jax_devices = jax.devices()
        self.assertIsNotNone(jax_devices)

    def test_simulated_data(self):
        adata = simulate_blob_data(n_samples=5000, n_tests=200)

        locat = LOCAT(adata, adata.obsm["coords"], 20,
                      show_progress=False, n_bootstrap_inits=100, knn=adata.obsp["connectivities"])
        locat.background_pdf()
        locat_results = locat.gmm_scan_new(zscore_thresh=-np.inf, max_freq=1., rc_min_abs_deficit=0.,
                                           rc_min_expected=0., rc_min_p0_abs=0.,
                                           rc_n_trials_cap=np.sqrt(adata.shape[0]), rc_n_eff_scale=0.9)

        gene_0_results = locat_results['Gene_0']

        self.assertEqual('Gene_0', gene_0_results.gene_name)
        self.assertAlmostEqual(-1.69827, gene_0_results.bic, places=4)
        self.assertLessEqual(0., gene_0_results.zscore) # This changes randomly
        self.assertAlmostEqual(1.0, gene_0_results.sens_score, places=5)
        self.assertAlmostEqual(1e-12, gene_0_results.depletion_pval, places=12)
        self.assertAlmostEqual(1e-15, gene_0_results.concentration_pval, places=15)
        self.assertAlmostEqual(0.000970, gene_0_results.pval, places=5)
        self.assertEqual(1, gene_0_results.K_components)
        self.assertEqual(50, gene_0_results.sample_size)


if __name__ == '__main__':
    unittest.main()
