import unittest

import torch

from forecasting.losses import horizon_weights


class TestHorizonWeights(unittest.TestCase):
    def test_length_and_endpoints(self):
        w = horizon_weights(5, floor=0.2)
        self.assertEqual(w.shape, (5,))
        self.assertAlmostEqual(w[0].item(), 1.0, places=6)
        self.assertAlmostEqual(w[-1].item(), 0.2, places=6)

    def test_linear_midpoint(self):
        w = horizon_weights(5, floor=0.0)
        # k=2 of t=5 -> 1 - 1.0 * 2/4 = 0.5
        self.assertAlmostEqual(w[2].item(), 0.5, places=6)

    def test_floor_one_is_uniform(self):
        w = horizon_weights(7, floor=1.0)
        self.assertTrue(torch.allclose(w, torch.ones(7)))

    def test_monotonically_non_increasing(self):
        w = horizon_weights(10, floor=0.2)
        self.assertTrue(torch.all(w[1:] <= w[:-1] + 1e-9))


if __name__ == "__main__":
    unittest.main()
