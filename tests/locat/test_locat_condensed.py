import unittest
import jax


class AlgorithmTestCase(unittest.TestCase):
    def test_jax_devices(self):
        jax_devices = jax.devices()
        self.assertIsNotNone(jax_devices)


if __name__ == '__main__':
    unittest.main()
