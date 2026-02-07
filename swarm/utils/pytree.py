"""JAX pytree utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def tree_stack(trees: list) -> dict:
    """Stack a list of pytrees into a single pytree with batched leaves."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def tree_unstack(tree: dict) -> list:
    """Unstack a batched pytree into a list of pytrees."""
    leaves, treedef = jax.tree.flatten(tree)
    n = leaves[0].shape[0]
    return [
        jax.tree.unflatten(treedef, [leaf[i] for leaf in leaves])
        for i in range(n)
    ]


def tree_slice(tree: dict, start: int, end: int) -> dict:
    """Slice a batched pytree along the first axis."""
    return jax.tree.map(lambda x: x[start:end], tree)
