"""Performance benchmarking for swarm simulation.

Measures steps/second at various agent counts and configurations.
"""

from __future__ import annotations

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp


class BenchmarkResult(NamedTuple):
    """Benchmark result."""
    num_agents: int
    num_steps: int
    total_time_sec: float
    steps_per_second: float
    agent_steps_per_second: float
    jit_compile_time_sec: float


def benchmark_dynamics(
    num_agents: int = 10,
    num_steps: int = 1000,
    warmup_steps: int = 100,
) -> BenchmarkResult:
    """Benchmark pure dynamics stepping.
    
    Args:
        num_agents: Number of agents
        num_steps: Steps to benchmark
        warmup_steps: JIT warmup steps
        
    Returns:
        BenchmarkResult
    """
    from swarm.envs.dynamics import PointMassDynamics, SwarmState
    
    # Create dynamics
    dynamics = PointMassDynamics()
    
    # Initial state
    key = jax.random.PRNGKey(0)
    pos = jax.random.uniform(key, (num_agents, 3)) * 100 - 50
    vel = jnp.zeros((num_agents, 3))
    energy = jnp.ones(num_agents)
    
    state = SwarmState(pos=pos, vel=vel, energy=energy, time=0.0)
    
    # JIT compile
    step_fn = jax.jit(dynamics.step_swarm)
    
    # Warmup (measure JIT time)
    jit_start = time.perf_counter()
    for _ in range(warmup_steps):
        actions = jax.random.normal(key, (num_agents, 3)) * 5
        state = step_fn(state, actions)
        state.pos.block_until_ready()
    jit_time = time.perf_counter() - jit_start
    
    # Benchmark
    start = time.perf_counter()
    for i in range(num_steps):
        actions = jax.random.normal(jax.random.PRNGKey(i), (num_agents, 3)) * 5
        state = step_fn(state, actions)
    state.pos.block_until_ready()  # Wait for async
    elapsed = time.perf_counter() - start
    
    sps = num_steps / elapsed
    
    return BenchmarkResult(
        num_agents=num_agents,
        num_steps=num_steps,
        total_time_sec=elapsed,
        steps_per_second=sps,
        agent_steps_per_second=sps * num_agents,
        jit_compile_time_sec=jit_time,
    )


def benchmark_environment(
    num_agents: int = 10,
    num_steps: int = 500,
) -> BenchmarkResult:
    """Benchmark full environment (dynamics + observations + rewards).
    
    Args:
        num_agents: Number of agents
        num_steps: Steps to benchmark
        
    Returns:
        BenchmarkResult
    """
    from swarm.envs.mjx_env import SwarmEnv, EnvConfig
    
    config = EnvConfig(num_agents=num_agents)
    env = SwarmEnv(config)
    
    # JIT compile step
    step_fn = jax.jit(env.step)
    
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key)
    
    # Warmup
    jit_start = time.perf_counter()
    for _ in range(10):
        actions = jax.random.normal(key, (num_agents, 3)) * 5
        result = step_fn(state, actions)
        result.state.pos.block_until_ready()
        state = result.state
    jit_time = time.perf_counter() - jit_start
    
    # Reset for benchmark
    state, _ = env.reset(key)
    
    # Benchmark
    start = time.perf_counter()
    for i in range(num_steps):
        actions = jax.random.normal(jax.random.PRNGKey(i), (num_agents, 3)) * 5
        result = step_fn(state, actions)
        state = result.state
    state.pos.block_until_ready()
    elapsed = time.perf_counter() - start
    
    sps = num_steps / elapsed
    
    return BenchmarkResult(
        num_agents=num_agents,
        num_steps=num_steps,
        total_time_sec=elapsed,
        steps_per_second=sps,
        agent_steps_per_second=sps * num_agents,
        jit_compile_time_sec=jit_time,
    )


def run_scaling_benchmark(max_agents: int = 100) -> list[BenchmarkResult]:
    """Run benchmark at multiple agent counts.
    
    Args:
        max_agents: Maximum agents to test
        
    Returns:
        List of benchmark results
    """
    agent_counts = [10, 20, 50, 100]
    agent_counts = [n for n in agent_counts if n <= max_agents]
    
    results = []
    
    print("=" * 60)
    print("🚀 Swarm Simulation Throughput Benchmark")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print()
    
    for num_agents in agent_counts:
        print(f"Benchmarking {num_agents} agents...")
        result = benchmark_environment(num_agents=num_agents)
        results.append(result)
        
        print(f"  Steps/sec: {result.steps_per_second:,.0f}")
        print(f"  Agent-steps/sec: {result.agent_steps_per_second:,.0f}")
        print(f"  JIT compile: {result.jit_compile_time_sec:.2f}s")
        print()
    
    # Summary table
    print("-" * 60)
    print(f"{'Agents':<10} {'SPS':>12} {'Agent-SPS':>15} {'1M Steps':>12}")
    print("-" * 60)
    for r in results:
        mins = 1_000_000 / r.steps_per_second / 60
        print(f"{r.num_agents:<10} {r.steps_per_second:>12,.0f} "
              f"{r.agent_steps_per_second:>15,.0f} {mins:>10.1f} min")
    print("-" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark swarm simulation")
    parser.add_argument("--max-agents", type=int, default=100, 
                        help="Maximum agent count to test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick benchmark (fewer steps)")
    
    args = parser.parse_args()
    
    run_scaling_benchmark(max_agents=args.max_agents)
