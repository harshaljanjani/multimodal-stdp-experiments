import cupy as cp
import numpy as np

def generate_grid_positions(count: int):
    side = int(np.ceil(np.sqrt(count)))
    x = cp.linspace(0, 1, side)
    y = cp.linspace(0, 1, side)
    xv, yv = cp.meshgrid(x, y)
    positions = cp.vstack([xv.ravel(), yv.ravel()]).T
    return positions[:count].astype(cp.float32)

def generate_random_positions(count: int):
    return cp.random.rand(count, 2).astype(cp.float32)

def create_fixed_probability_connections(
    source_pop_size: int,
    target_pop_size: int,
    probability: float,
    allow_autapses: bool = False
):
    if probability <= 0:
        return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)
    num_synapses = int(source_pop_size * target_pop_size * probability)
    source_indices = cp.random.randint(0, source_pop_size, size=num_synapses, dtype=cp.int32)
    target_indices = cp.random.randint(0, target_pop_size, size=num_synapses, dtype=cp.int32)
    if not allow_autapses and source_pop_size == target_pop_size:
        autapses = source_indices == target_indices
        mask = ~autapses
        source_indices = source_indices[mask]
        target_indices = target_indices[mask]
    return source_indices, target_indices

def create_gaussian_distance_connections(
    source_pos: cp.ndarray,
    target_pos: cp.ndarray,
    p_max: float,
    sigma: float,
    allow_autapses: bool = False
):
    if p_max <= 0 or sigma <= 0:
        return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)
    n_source = source_pos.shape[0]
    n_target = target_pos.shape[0]
    diffs = source_pos[:, None, :] - target_pos[None, :, :]
    sq_dists = cp.sum(diffs**2, axis=-1)
    prob_matrix = p_max * cp.exp(-sq_dists / (2 * sigma**2))
    random_matrix = cp.random.rand(n_source, n_target, dtype=cp.float32)
    connections = random_matrix < prob_matrix
    if not allow_autapses and source_pos is target_pos:
        cp.fill_diagonal(connections, False)
    source_indices, target_indices = connections.nonzero()
    return source_indices.astype(cp.int32), target_indices.astype(cp.int32)
