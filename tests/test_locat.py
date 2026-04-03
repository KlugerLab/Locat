import unittest
import jax
import numpy as np

from locat.locat import logsidak_from_logp, _safe_p


class LocatTestCase(unittest.TestCase):
    def test_jax_devices(self):
        jax_devices = jax.devices()
        self.assertIsNotNone(jax_devices)


    def test_logp_from_sidak(self):
        self.assertAlmostEqual(logsidak_from_logp(0.12345, 1), 0.12345 , places=5)
        self.assertAlmostEqual(logsidak_from_logp(np.log(1), 2), np.log(1) , places=8)
        self.assertAlmostEqual(logsidak_from_logp(np.log(0), 2), np.log(0) , places=8)
        self.assertAlmostEqual(logsidak_from_logp(-10, 10), -7.697619202824944 , places=8)

        self.assertAlmostEqual(np.exp(logsidak_from_logp(np.log(0.01), 8)), 0.077255305572 , places=8)

        self.assertAlmostEqual(_safe_p(logsidak_from_logp(-56.24362650, 8)), 1-1e-15 ,
                               places=8, msg="very small p-values are clipped in the pipeline" )

        self.assertAlmostEqual(logsidak_from_logp(-56.24362650, 8), -54.164184960 ,
                               places=8, msg="very small values should not underflow")

if __name__ == '__main__':
    unittest.main()
