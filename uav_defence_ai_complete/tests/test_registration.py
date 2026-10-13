import numpy as np
from src.registration.quality_check import normalized_correlation


def test_correlation_identical():
    x = np.zeros((20, 20), dtype=np.uint8)
    x[5:15, 5:15] = 255
    assert normalized_correlation(x, x) > 0.99
