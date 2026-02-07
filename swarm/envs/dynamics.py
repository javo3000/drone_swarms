"""Point-mass dynamics for simplified drone simulation.

This module implements a 6-DOF point-mass model with thrust control,
suitable for swarm coordination research where aerodynamic fidelity
is less critical than scalability.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class DroneState(NamedTuple):
    """State of a single drone agent.
    
    Attributes:
        pos: Position (x, y, z) in meters
        vel: Velocity (vx, vy, vz) in m/s
        quat: Orientation quaternion (w, x, y, z)
        omega: Angular velocity (wx, wy, wz) in rad/s
        energy: Remaining battery energy (0-1)
    """
    pos: Array    # (3,)
    vel: Array    # (3,)
    quat: Array   # (4,)
    omega: Array  # (3,)
    energy: Array # ()


class SwarmState(NamedTuple):
    """State of the entire swarm.
    
    All arrays have shape (num_agents, ...).
    """
    pos: Array     # (N, 3)
    vel: Array     # (N, 3)
    quat: Array    # (N, 4)
    omega: Array   # (N, 3)
    energy: Array  # (N,)
    time: Array    # ()


class PointMassDynamics:
    """Point-mass dynamics with thrust and drag.
    
    Simplified 6-DOF model:
    - Linear acceleration from thrust
    - Quadratic drag proportional to velocity
    - Energy consumption proportional to thrust magnitude
    
    Args:
        mass: Drone mass in kg
        gravity: Gravitational acceleration (default: 9.81 m/s²)
        drag_coeff: Drag coefficient (default: 0.1)
        max_thrust: Maximum total thrust in N
        dt: Integration timestep in seconds
    """
    
    def __init__(
        self,
        mass: float = 1.0,
        gravity: float = 9.81,
        drag_coeff: float = 0.1,
        max_thrust: float = 20.0,
        dt: float = 0.02,
        energy_rate: float = 0.001,
    ):
        self.mass = mass
        self.gravity = jnp.array([0.0, 0.0, -gravity])
        self.drag_coeff = drag_coeff
        self.max_thrust = max_thrust
        self.dt = dt
        self.energy_rate = energy_rate
    
    @staticmethod
    def quat_rotate(quat: Array, vec: Array) -> Array:
        """Rotate vector by quaternion (w, x, y, z format)."""
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        
        # Quaternion rotation: q * v * q^-1
        t = 2.0 * jnp.cross(jnp.stack([x, y, z], axis=-1), vec)
        return vec + w[..., None] * t + jnp.cross(jnp.stack([x, y, z], axis=-1), t)
    
    def step_single(
        self,
        state: DroneState,
        thrust_cmd: Array,  # (4,) thrust per rotor or (3,) force vector
    ) -> DroneState:
        """Integrate dynamics for a single drone.
        
        Args:
            state: Current drone state
            thrust_cmd: Thrust command (3,) as force vector in body frame
            
        Returns:
            New drone state after dt
        """
        # Clip thrust and convert to world frame
        thrust_body = jnp.clip(thrust_cmd, -self.max_thrust, self.max_thrust)
        thrust_world = self.quat_rotate(state.quat, thrust_body)
        
        # Compute forces
        gravity_force = self.mass * self.gravity
        drag_force = -self.drag_coeff * state.vel * jnp.abs(state.vel)
        total_force = thrust_world + gravity_force + drag_force
        
        # Integrate (semi-implicit Euler)
        acc = total_force / self.mass
        new_vel = state.vel + acc * self.dt
        new_pos = state.pos + new_vel * self.dt
        
        # Energy consumption (proportional to thrust magnitude)
        thrust_mag = jnp.linalg.norm(thrust_body)
        new_energy = jnp.maximum(0.0, state.energy - self.energy_rate * thrust_mag * self.dt)
        
        return DroneState(
            pos=new_pos,
            vel=new_vel,
            quat=state.quat,  # Simplified: no rotation dynamics
            omega=state.omega,
            energy=new_energy,
        )
    
    def step_swarm(
        self,
        state: SwarmState,
        actions: Array,  # (N, 3)
    ) -> SwarmState:
        """Integrate dynamics for entire swarm (vmapped).
        
        Args:
            state: Current swarm state
            actions: Thrust commands for all agents (N, 3)
            
        Returns:
            New swarm state after dt
        """
        # Create per-agent states
        def step_agent(pos, vel, quat, omega, energy, action):
            agent_state = DroneState(pos, vel, quat, omega, energy)
            new_state = self.step_single(agent_state, action)
            return new_state.pos, new_state.vel, new_state.quat, new_state.omega, new_state.energy
        
        # Vectorize over agents
        new_pos, new_vel, new_quat, new_omega, new_energy = jax.vmap(step_agent)(
            state.pos, state.vel, state.quat, state.omega, state.energy, actions
        )
        
        return SwarmState(
            pos=new_pos,
            vel=new_vel,
            quat=new_quat,
            omega=new_omega,
            energy=new_energy,
            time=state.time + self.dt,
        )
    
    def reset_swarm(
        self,
        key: Array,
        num_agents: int,
        arena_size: float = 100.0,
        init_height: float = 10.0,
    ) -> SwarmState:
        """Initialize swarm with random positions in arena.
        
        Args:
            key: JAX PRNG key
            num_agents: Number of drones
            arena_size: Size of square arena in meters
            init_height: Initial flight height
            
        Returns:
            Initial swarm state
        """
        k1, k2 = jax.random.split(key)
        
        # Random XY positions, fixed height
        xy = jax.random.uniform(k1, (num_agents, 2), minval=-arena_size/2, maxval=arena_size/2)
        z = jnp.full((num_agents, 1), init_height)
        pos = jnp.concatenate([xy, z], axis=-1)
        
        # Zero initial velocity
        vel = jnp.zeros((num_agents, 3))
        
        # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
        quat = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (num_agents, 1))
        
        # Zero angular velocity
        omega = jnp.zeros((num_agents, 3))
        
        # Full battery
        energy = jnp.ones(num_agents)
        
        return SwarmState(
            pos=pos,
            vel=vel,
            quat=quat,
            omega=omega,
            energy=energy,
            time=jnp.array(0.0),
        )


# JIT-compiled versions for performance
@jax.jit
def dynamics_step(dynamics: PointMassDynamics, state: SwarmState, actions: Array) -> SwarmState:
    """JIT-compiled swarm dynamics step."""
    return dynamics.step_swarm(state, actions)
