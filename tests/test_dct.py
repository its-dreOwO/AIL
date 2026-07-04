import numpy as np

from forecasting.dct import get_dct_matrix


def test_shapes():
    dct_m, idct_m = get_dct_matrix(250)
    assert dct_m.shape == (250, 250)
    assert idct_m.shape == (250, 250)


def test_roundtrip_reconstructs_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3))
    dct_m, idct_m = get_dct_matrix(50)
    x_rec = idct_m @ (dct_m @ x)
    assert np.allclose(x_rec, x, atol=1e-8)


def test_inverse_is_inverse():
    dct_m, idct_m = get_dct_matrix(32)
    assert np.allclose(idct_m @ dct_m, np.eye(32), atol=1e-8)
