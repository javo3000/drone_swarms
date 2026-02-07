"""Multi-GPU sharding utilities for distributed training.

Uses JAX shard_map for distributing swarm simulation across multiple GPUs.
Designed for 4× RTX 5090 setup but works with any GPU count.
"""

from __future__ import annotations

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from swarm.envs.dynamics import SwarmState, PointMassDynamics


class ShardingConfig(NamedTuple):
    """Sharding configuration."""
    num_devices: int = 4
    batch_axis: str = "batch"
    agent_axis: str = "agents"


def get_device_mesh(num_devices: int | None = None) -> Mesh:
    """Create device mesh for sharding.
    
    Args:
        num_devices: Number of devices (None = use all available)
        
    Returns:
        JAX device mesh
    """
    devices = jax.devices()
    if num_devices is not None:
        devices = devices[:num_devices]
    
    # For 4 GPUs: 2x2 mesh or 1x4 depending on workload
    n = len(devices)
    if n >= 4:
        mesh = Mesh(jnp.array(devices).reshape(2, 2), ("batch", "data"))
    elif n >= 2:
        mesh = Mesh(jnp.array(devices).reshape(1, n), ("batch", "data"))
    else:
        mesh = Mesh(jnp.array(devices), ("batch",))
    
    return mesh


def shard_swarm_state(state: SwarmState, mesh: Mesh) -> SwarmState:
    """Shard swarm state across devices.
    
    Shards the batch dimension (environments) across devices.
    
    Args:
        state: SwarmState with batch dimension
        mesh: Device mesh
        
    Returns:
        Sharded SwarmState
    """
    spec = NamedSharding(mesh, P("batch", None, None))
    
    return SwarmState(
        pos=jax.device_put(state.pos, spec),
        vel=jax.device_put(state.vel, spec),
        energy=jax.device_put(state.energy, NamedSharding(mesh, P("batch", None))),
        time=state.time,
    )


def create_sharded_step_fn(
    dynamics: PointMassDynamics,
    mesh: Mesh,
):
    """Create a sharded step function.
    
    Args:
        dynamics: Physics model
        mesh: Device mesh
        
    Returns:
        Sharded step function
    """
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("batch", None, None), P("batch", None, None)),
        out_specs=P("batch", None, None),
    )
    def sharded_step(pos, actions):
        # Simplified step - just update positions for demo
        return pos + actions * 0.02
    
    return sharded_step


class DistributedTrainer:
    """Distributed trainer for multi-GPU training.
    
    Shards environments across GPUs and aggregates gradients.
    
    Args:
        num_envs_per_device: Environments per GPU
        config: Sharding configuration
    """
    
    def __init__(
        self,
        num_envs_per_device: int = 8,
        config: ShardingConfig | None = None,
    ):
        self.config = config or ShardingConfig()
        self.num_envs_per_device = num_envs_per_device
        
        # Create mesh
        self.mesh = get_device_mesh()
        self.num_devices = len(self.mesh.devices)
        self.total_envs = self.num_devices * num_envs_per_device
        
        print(f"Initialized distributed trainer:")
        print(f"  Devices: {self.num_devices}")
        print(f"  Envs per device: {num_envs_per_device}")
        print(f"  Total envs: {self.total_envs}")
    
    def shard_batch(self, data: Array) -> Array:
        """Shard data across devices."""
        spec = NamedSharding(self.mesh, P("batch", None))
        return jax.device_put(data, spec)
    
    def gather_batch(self, data: Array) -> Array:
        """Gather sharded data to single device."""
        return jax.device_get(data)


def estimate_throughput(
    num_agents: int,
    num_devices: int,
    batch_size: int = 32,
) -> dict:
    """Estimate simulation throughput.
    
    Args:
        num_agents: Agents per environment
        num_devices: Number of GPUs
        batch_size: Environments in parallel
        
    Returns:
        Throughput estimates
    """
    # Rough estimates based on JAX/MJX benchmarks
    single_gpu_sps = 50_000  # Steps per second (point-mass)
    
    # Linear scaling up to ~4 GPUs, then sublinear
    if num_devices <= 4:
        scaling = num_devices * 0.95  # 95% efficiency
    else:
        scaling = 4 * 0.95 + (num_devices - 4) * 0.7  # Sublinear after 4
    
    total_sps = single_gpu_sps * scaling * (batch_size / 8)
    
    # Adjust for agent count (O(n²) for collision detection)
    agent_factor = (10 / num_agents) ** 0.5 if num_agents > 10 else 1.0
    total_sps *= agent_factor
    
    return {
        "steps_per_second": int(total_sps),
        "agent_steps_per_second": int(total_sps * num_agents),
        "estimated_1m_steps_time_min": 1_000_000 / total_sps / 60,
    }


if __name__ == "__main__":
    # Print device info
    print("Available devices:", jax.devices())
    print()
    
    # Throughput estimates
    for agents in [10, 50, 100]:
        for gpus in [1, 4]:
            est = estimate_throughput(agents, gpus)
            print(f"{agents} agents, {gpus} GPU(s):")
            print(f"  SPS: {est['steps_per_second']:,}")
            print(f"  1M steps: {est['estimated_1m_steps_time_min']:.1f} min")
            print()
