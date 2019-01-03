import numpy as np


def l2_normalize(x, axis, epsilon=1e-12):
    out = x / np.sqrt(
        np.maximum(
            np.sum(np.square(x), axis=axis, keepdims=True),
            epsilon
        )
    )
    return out
