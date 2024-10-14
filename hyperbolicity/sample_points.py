import embedders
import numpy as np
from torchtyping import TensorType
from manifolds import Manifold
import torch
from typing import Tuple



def sample_points_on_manifold(curvature: float,
                              dim: int, 
                              n_points: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Samples points on a manifold based on the given curvatures.

    Args:
    - curvature: manifold curvature
    - dim: The dimension of the manifold.
    - n_points: The number of points to sample on each manifold.

    Returns:
    - the sampled points [n_points, dim].
    - the distance matrix between points [n_points, n_points]
    """

    m_h = Manifold(curvature, dim)
    if curvature != 0:
        points = m_h.sample([([1] + [0] * dim) * n_points])
    else:
        points = m_h.sample([([1] + [0] * (dim - 1)) * n_points]) 

    distance_mat = m_h.pdist(points)
    distance_mat = distance_mat.detach().numpy()
    return points, distance_mat
