"""Tests for swarm environment."""

import pytest
import jax
import jax.numpy as jnp

from swarm.envs.dynamics import PointMassDynamics, DroneState, SwarmState
from swarm.envs.mjx_env import SwarmEnv, EnvConfig
from swarm.envs.reset import reset_grid, reset_random_circle


class TestDynamics:
    """Test point-mass dynamics."""
    
    def test_init(self):
        """Test dynamics initialization."""
        dynamics = PointMassDynamics()
        assert dynamics.mass == 1.0
        assert dynamics.max_thrust == 20.0
    
    def test_reset_swarm(self):
        """Test swarm initialization."""
        dynamics = PointMassDynamics()
        key = jax.random.PRNGKey(0)
        state = dynamics.reset_swarm(key, num_agents=10)
        
        assert state.pos.shape == (10, 3)
        assert state.vel.shape == (10, 3)
        assert state.energy.shape == (10,)
        assert jnp.all(state.energy == 1.0)
    
    def test_step_swarm(self):
        """Test dynamics step."""
        dynamics = PointMassDynamics()
        key = jax.random.PRNGKey(0)
        state = dynamics.reset_swarm(key, num_agents=5)
        
        # Apply upward thrust to counteract gravity
        actions = jnp.zeros((5, 3))
        actions = actions.at[:, 2].set(dynamics.mass * 9.81)  # Hover thrust
        
        new_state = dynamics.step_swarm(state, actions)
        
        # Position should change slightly due to drag
        assert not jnp.allclose(new_state.pos, state.pos, atol=0.01)
        assert new_state.time > state.time
    
    def test_energy_consumption(self):
        """Test that thrust consumes energy."""
        dynamics = PointMassDynamics()
        key = jax.random.PRNGKey(0)
        state = dynamics.reset_swarm(key, num_agents=2)
        
        # High thrust
        actions = jnp.ones((2, 3)) * 10.0
        new_state = dynamics.step_swarm(state, actions)
        
        # Energy should decrease
        assert jnp.all(new_state.energy < state.energy)


class TestEnvironment:
    """Test swarm environment."""
    
    def test_init(self):
        """Test environment initialization."""
        env = SwarmEnv()
        assert env.config.num_agents == 10
    
    def test_reset(self):
        """Test environment reset."""
        env = SwarmEnv(EnvConfig(num_agents=5))
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)
        
        assert state.pos.shape == (5, 3)
        assert obs.own_state.shape == (5, 7)
    
    def test_step(self):
        """Test environment step."""
        env = SwarmEnv(EnvConfig(num_agents=5))
        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)
        
        actions = jax.random.normal(key, (5, 3))
        result = env.step(state, actions)
        
        assert result.state.pos.shape == (5, 3)
        assert result.reward.shape == (5,)
        assert isinstance(result.done, jax.Array)
    
    def test_observations(self):
        """Test observation structure."""
        env = SwarmEnv(EnvConfig(num_agents=10))
        key = jax.random.PRNGKey(0)
        state, obs = env.reset(key)
        
        # Check observation shapes
        assert obs.own_state.shape == (10, 7)  # pos(3) + vel(3) + energy(1)
        assert obs.relative_positions.shape[0] == 10
        assert obs.relative_velocities.shape[0] == 10


class TestResetFormations:
    """Test reset formations."""
    
    def test_grid(self):
        """Test grid formation."""
        state = reset_grid(num_agents=9, spacing=5.0, height=10.0)
        
        assert state.pos.shape == (9, 3)
        assert jnp.allclose(state.pos[:, 2], 10.0)  # All at same height
    
    def test_circle(self):
        """Test circle formation."""
        key = jax.random.PRNGKey(0)
        state = reset_random_circle(key, num_agents=10, radius=50.0)
        
        # All agents within radius
        xy_dist = jnp.linalg.norm(state.pos[:, :2], axis=-1)
        assert jnp.all(xy_dist <= 50.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
