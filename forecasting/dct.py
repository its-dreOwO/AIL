import numpy as np


def get_dct_matrix(N: int):
    """Return orthonormal DCT-II and inverse matrices over the time axis."""
    dct_m = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        w = np.sqrt(2.0 / N)
        if k == 0:
            w = np.sqrt(1.0 / N)
        for i in range(N):
            dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
