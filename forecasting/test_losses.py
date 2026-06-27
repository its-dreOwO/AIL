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


from forecasting.losses import mpjpe_loss
from forecasting.config import POSE_DIM


class TestWeightedMpjpe(unittest.TestCase):
    def _pair(self, t=10):
        torch.manual_seed(0)
        target = torch.zeros(2, t, POSE_DIM)
        pred = torch.zeros(2, t, POSE_DIM)
        return pred, target

    def test_floor_one_equals_unweighted(self):
        torch.manual_seed(1)
        pred = torch.randn(2, 10, POSE_DIM)
        target = torch.randn(2, 10, POSE_DIM)
        plain = mpjpe_loss(pred, target)
        weighted = mpjpe_loss(pred, target, weights=horizon_weights(10, floor=1.0))
        self.assertAlmostEqual(plain.item(), weighted.item(), places=5)

    def test_late_error_downweighted(self):
        # error only in the LAST frame -> weighting should LOWER the loss
        pred, target = self._pair(t=10)
        pred[:, -1, :] = 1.0
        plain = mpjpe_loss(pred, target)
        weighted = mpjpe_loss(pred, target, weights=horizon_weights(10, floor=0.2))
        self.assertLess(weighted.item(), plain.item())

    def test_early_error_upweighted(self):
        # error only in the FIRST frame -> weighting should RAISE the loss
        pred, target = self._pair(t=10)
        pred[:, 0, :] = 1.0
        plain = mpjpe_loss(pred, target)
        weighted = mpjpe_loss(pred, target, weights=horizon_weights(10, floor=0.2))
        self.assertGreater(weighted.item(), plain.item())


if __name__ == "__main__":
    unittest.main()
