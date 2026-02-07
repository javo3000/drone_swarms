"""Enhanced visualizer with 3D support and multiple scenarios.

Provides visualization for:
- 2D/3D swarm positions
- Threat zones
- GPS denied areas
- Communication links
- Agent trails
"""

from __future__ import annotations

from typing import NamedTuple
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from swarm.envs.dynamics import SwarmState


class VisualizerConfig(NamedTuple):
    """Visualization configuration."""
    mode: str = "2d"  # "2d" or "3d"
    show_trails: bool = True
    trail_length: int = 30
    show_energy: bool = True
    show_comm_links: bool = False
    show_voronoi: bool = False
    fps: int = 20
    figsize: tuple = (14, 6)


class SwarmVisualizer:
    """Enhanced swarm visualizer.
    
    Args:
        config: Visualization configuration
        arena_size: Arena size in meters
    """
    
    def __init__(
        self,
        config: VisualizerConfig | None = None,
        arena_size: float = 100.0,
    ):
        self.config = config or VisualizerConfig()
        self.arena_size = arena_size
        self.half = arena_size / 2
        
        # State
        self.fig = None
        self.axes = None
        self.scatter = None
        self.trail_lines = []
        self.history = {}
        
    def create_figure(self, num_agents: int):
        """Create matplotlib figure."""
        if self.config.mode == "3d":
            self.fig = plt.figure(figsize=self.config.figsize)
            self.ax_main = self.fig.add_subplot(121, projection='3d')
            self.ax_reward = self.fig.add_subplot(122)
        else:
            self.fig, axes = plt.subplots(1, 2, figsize=self.config.figsize)
            self.ax_main, self.ax_reward = axes
        
        self._setup_main_axis(num_agents)
        self._setup_reward_axis()
        
        # Initialize history
        self.history = {i: [] for i in range(num_agents)}
        
        return self.fig
    
    def _setup_main_axis(self, num_agents: int):
        """Setup main swarm axis."""
        ax = self.ax_main
        
        if self.config.mode == "3d":
            ax.set_xlim(-self.half, self.half)
            ax.set_ylim(-self.half, self.half)
            ax.set_zlim(0, 50)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        else:
            ax.set_xlim(-self.half * 1.2, self.half * 1.2)
            ax.set_ylim(-self.half * 1.2, self.half * 1.2)
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.3)
            
            # Arena boundary
            rect = patches.Rectangle(
                (-self.half, -self.half), self.arena_size, self.arena_size,
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
        
        ax.set_title('Swarm Positions')
        
        # Initialize trails
        if self.config.show_trails:
            for i in range(num_agents):
                if self.config.mode == "3d":
                    line, = ax.plot([], [], [], 'b-', alpha=0.3, linewidth=1)
                else:
                    line, = ax.plot([], [], 'b-', alpha=0.3, linewidth=1)
                self.trail_lines.append(line)
    
    def _setup_reward_axis(self):
        """Setup reward plot axis."""
        ax = self.ax_reward
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Reward Over Time')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        self.reward_line, = ax.plot([], [], 'b-', linewidth=2)
    
    def update(self, state: SwarmState, step: int, reward: float = 0.0):
        """Update visualization with new state."""
        pos = np.array(state.pos)
        energy = np.array(state.energy)
        
        # Update scatter
        if self.scatter is not None:
            self.scatter.remove()
        
        if self.config.mode == "3d":
            self.scatter = self.ax_main.scatter(
                pos[:, 0], pos[:, 1], pos[:, 2],
                c=energy if self.config.show_energy else 'blue',
                cmap='RdYlGn', s=100, vmin=0, vmax=1,
            )
        else:
            self.scatter = self.ax_main.scatter(
                pos[:, 0], pos[:, 1],
                c=energy if self.config.show_energy else 'blue',
                cmap='RdYlGn', s=120, edgecolors='black',
                linewidth=1.5, vmin=0, vmax=1,
            )
        
        # Update trails
        if self.config.show_trails:
            for i in range(pos.shape[0]):
                self.history[i].append(pos[i])
                if len(self.history[i]) > self.config.trail_length:
                    self.history[i] = self.history[i][-self.config.trail_length:]
                
                if len(self.history[i]) > 1:
                    trail = np.array(self.history[i])
                    if self.config.mode == "3d":
                        self.trail_lines[i].set_data(trail[:, 0], trail[:, 1])
                        self.trail_lines[i].set_3d_properties(trail[:, 2])
                    else:
                        self.trail_lines[i].set_data(trail[:, 0], trail[:, 1])
    
    def add_threat_zone(self, center: tuple, radius: float, color: str = 'red'):
        """Add threat zone circle."""
        if self.config.mode == "2d":
            circle = patches.Circle(
                center[:2], radius,
                linewidth=2, edgecolor=color, facecolor=color,
                alpha=0.2, linestyle='--'
            )
            self.ax_main.add_patch(circle)
    
    def add_gps_denied_zone(self, center: tuple, radius: float):
        """Add GPS denied zone."""
        if self.config.mode == "2d":
            circle = patches.Circle(
                center[:2], radius,
                linewidth=2, edgecolor='purple', facecolor='purple',
                alpha=0.15, linestyle=':'
            )
            self.ax_main.add_patch(circle)
            self.ax_main.annotate(
                'GPS\nDenied', center[:2],
                ha='center', va='center', fontsize=8, color='purple'
            )
    
    def add_jammer(self, center: tuple, radius: float):
        """Add jammer zone."""
        if self.config.mode == "2d":
            circle = patches.Circle(
                center[:2], radius,
                linewidth=3, edgecolor='orange', facecolor='orange',
                alpha=0.2, linestyle='-'
            )
            self.ax_main.add_patch(circle)
            self.ax_main.plot(center[0], center[1], 'X', 
                             color='orange', markersize=15, markeredgecolor='black')


def create_demo_animation(
    trajectory: list[SwarmState],
    rewards: list[float],
    arena_size: float = 100.0,
    save_path: str | None = None,
    title: str = "Swarm Demo",
    threats: list[tuple] | None = None,
    gps_denied: list[tuple] | None = None,
) -> FuncAnimation:
    """Create animated demo from trajectory.
    
    Args:
        trajectory: List of SwarmStates
        rewards: List of rewards at each step
        arena_size: Arena size
        save_path: Optional save path for GIF/MP4
        title: Animation title
        threats: List of (center, radius) for threat zones
        gps_denied: List of (center, radius) for GPS denied zones
        
    Returns:
        FuncAnimation object
    """
    config = VisualizerConfig(show_trails=True)
    viz = SwarmVisualizer(config, arena_size)
    
    num_agents = trajectory[0].pos.shape[0]
    fig = viz.create_figure(num_agents)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Add zones
    if threats:
        for center, radius in threats:
            viz.add_threat_zone(center, radius)
    
    if gps_denied:
        for center, radius in gps_denied:
            viz.add_gps_denied_zone(center, radius)
    
    # Time text
    time_text = viz.ax_main.text(
        0.02, 0.98, '', transform=viz.ax_main.transAxes,
        va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Setup reward axis limits
    if rewards:
        viz.ax_reward.set_xlim(0, len(trajectory))
        viz.ax_reward.set_ylim(min(rewards) - 0.5, max(max(rewards), 0.5) + 0.5)
    
    def update(frame):
        state = trajectory[frame]
        viz.update(state, frame, rewards[frame] if frame < len(rewards) else 0)
        
        time_text.set_text(f'Time: {float(state.time):.2f}s | Agents: {num_agents}')
        
        if frame > 0 and frame < len(rewards):
            viz.reward_line.set_data(range(frame), rewards[:frame])
        
        return [viz.scatter, time_text, viz.reward_line] + viz.trail_lines
    
    anim = FuncAnimation(
        fig, update,
        frames=len(trajectory),
        interval=1000 // config.fps,
        blit=False,  # 3D doesn't support blit
    )
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=config.fps)
        else:
            anim.save(save_path, writer='pillow', fps=config.fps)
        print("Done!")
    
    return anim
