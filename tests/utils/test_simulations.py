import unittest
import numpy as np

from locat.locat import LOCAT
from locat.utils.simulations import simulate_blob_data


class SimulationTestCase(unittest.TestCase):

    def test_simulated_data(self):
        np.random.seed(0)
        adata = simulate_blob_data(n_samples=500, n_tests=20, n_total=50)
        gene_name = 'Gene_0'

        locat = LOCAT(adata, adata.obsm["coords"], k=10,
                      show_progress=False, knn=adata.obsp["connectivities"])
        locat_results = locat.gmm_scan(genes=[gene_name])

        gene_0_results = locat_results[gene_name]

        self.assertEqual(gene_name, gene_0_results.gene_name)
        self.assertAlmostEqual(-1.512298, gene_0_results.bic, places=4)
        self.assertLessEqual(0., gene_0_results.zscore) # This changes randomly
        self.assertAlmostEqual(1.0, gene_0_results.sens_score, places=5)
        self.assertLess(gene_0_results.depletion_pval, 1e-6)
        self.assertAlmostEqual(1e-15, gene_0_results.concentration_pval, places=14)
        self.assertLess(gene_0_results.pval, 1e-2)
        self.assertEqual(1, gene_0_results.K_components)
        self.assertEqual(50, gene_0_results.sample_size)


if __name__ == '__main__':
    unittest.main()
