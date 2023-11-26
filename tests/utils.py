import numpy as np
from typing import List

def assert_equal(xs: List[np.ndarray], ys: List[np.ndarray]):
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        assert np.array_equal(x, y, equal_nan=True)

