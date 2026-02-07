"""Geometry utilities for swarm operations."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def pairwise_distances(positions: Array) -> Array:
    """Compute pairwise Euclidean distances.
    
    Args:
        positions: (N, D) array of positions
        
    Returns:
        (N, N) distance matrix
    """
    diff = positions[:, None, :] - positions[None, :, :]
    return jnp.linalg.norm(diff, axis=-1)


def nearest_neighbors(positions: Array, k: int) -> tuple[Array, Array]:
    """Find k nearest neighbors for each point.
    
    Args:
        positions: (N, D) array of positions
        k: Number of neighbors
        
    Returns:
        Tuple of (indices, distances) each of shape (N, k)
    """
    dists = pairwise_distances(positions)
    # Mask self-distances
    dists = dists + jnp.eye(positions.shape[0]) * 1e10
    
    indices = jnp.argsort(dists, axis=-1)[:, :k]
    neighbor_dists = jnp.take_along_axis(dists, indices, axis=-1)
    
    return indices, neighbor_dists


def in_radius(positions: Array, radius: float) -> Array:
    """Find neighbors within radius.
    
    Args:
        positions: (N, D) array of positions  
        radius: Communication radius
        
    Returns:
        (N, N) boolean adjacency matrix
    """
    dists = pairwise_distances(positions)
    # Exclude self
    mask = jnp.eye(positions.shape[0], dtype=bool)
    return (dists < radius) & ~mask


def quaternion_multiply(q1: Array, q2: Array) -> Array:
    """Multiply two quaternions (w, x, y, z format).
    
    Args:
        q1, q2: Quaternions of shape (..., 4)
        
    Returns:
        Product quaternion
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return jnp.stack([w, x, y, z], axis=-1)


def normalize_quaternion(q: Array) -> Array:
    """Normalize quaternion to unit length."""
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)
