import numpy as np


def sem(arr, axis=None, ddof=1):
    """Standard error of the mean."""
    arr = np.asarray(arr)
    n = arr.shape[axis] if axis is not None else arr.size
    return arr.std(axis=axis, ddof=ddof) / np.sqrt(n)