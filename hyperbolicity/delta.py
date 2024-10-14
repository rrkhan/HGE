from typing import List
import numpy as np


def compute_delta(distance_matrix: np.ndarray,
                  delta_relative: bool = True) -> List[float]:
    """
    Computes the naive delta and diameter of a distance matrix.
    Adopted from the hypdelta python library (https://github.com/tnmtvv/hypdelta)

    Parameters:
    -----------
    dist_matrix : np.ndarray
        The distance matrix.

    Returns:
    --------
    List
        All computed deltas.
    """

    all_deltas = []    
    for p in range(distance_matrix.shape[0]):
        row = distance_matrix[p, :][np.newaxis, :]
        col = distance_matrix[:, p][:, np.newaxis]
        XY_p = 0.5 * (row + col - distance_matrix)
        maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
        deltas = maxmin - XY_p
        all_deltas.append(deltas.flatten())
    deltas = [item for sublist in all_deltas for item in sublist]
    if delta_relative:
        diam = np.max(distance_matrix)
        deltas = [(2 * e) / diam for e in deltas]
    return deltas
    