"""Visualization utilities for swarm simulation.

Provides real-time 3D rendering and video recording capabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches

from swarm.envs.dynamics import SwarmState


def plot_swarm_2d(
    state: SwarmState,
    arena_size: float = 100.0,
    title: str = "Swarm State",
    ax: plt.Axes | None = None,
    show_velocity: bool = True,
    show_energy: bool = True,
) -> plt.Axes:
    """Plot 2D top-down view of swarm.
    
    Args:
        state: Current swarm state
        arena_size: Arena size for axis limits
        title: Plot title
        ax: Matplotlib axes (created if None)
        show_velocity: Whether to show velocity arrows
        show_energy: Whether to color by energy level
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    pos = np.asarray(state.pos)
    vel = np.asarray(state.vel)
    energy = np.asarray(state.energy)
    
    # Color by energy if requested
    if show_energy:
        colors = plt.cm.RdYlGn(energy)
    else:
        colors = 'blue'
    
    # Plot agents
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=100, edgecolors='black', zorder=3)
    
    # Plot velocity arrows
    if show_velocity:
        ax.quiver(
            pos[:, 0], pos[:, 1],
            vel[:, 0], vel[:, 1],
            color='gray', alpha=0.6, scale=50, zorder=2
        )
    
    # Arena boundary
    half = arena_size / 2
    rect = patches.Rectangle(
        (-half, -half), arena_size, arena_size,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    
    ax.set_xlim(-half * 1.1, half * 1.1)
    ax.set_ylim(-half * 1.1, half * 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_swarm_3d(
    state: SwarmState,
    arena_size: float = 100.0,
    title: str = "Swarm State (3D)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot 3D view of swarm.
    
    Args:
        state: Current swarm state
        arena_size: Arena size for axis limits
        title: Plot title
        ax: Matplotlib 3D axes (created if None)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    pos = np.asarray(state.pos)
    energy = np.asarray(state.energy)
    colors = plt.cm.RdYlGn(energy)
    
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=100, edgecolors='black')
    
    half = arena_size / 2
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(0, arena_size / 2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    return ax


def animate_trajectory(
    trajectory: Sequence[SwarmState],
    arena_size: float = 100.0,
    interval: int = 50,
    save_path: str | Path | None = None,
) -> FuncAnimation:
    """Create animation from state trajectory.
    
    Args:
        trajectory: List of swarm states over time
        arena_size: Arena size for axis limits
        interval: Milliseconds between frames
        save_path: Path to save video (optional)
        
    Returns:
        Matplotlib FuncAnimation
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize plot
    positions = np.asarray(trajectory[0].pos)
    scatter = ax.scatter(positions[:, 0], positions[:, 1], s=100, edgecolors='black')
    
    # Arena boundary
    half = arena_size / 2
    rect = patches.Rectangle(
        (-half, -half), arena_size, arena_size,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    
    ax.set_xlim(-half * 1.1, half * 1.1)
    ax.set_ylim(-half * 1.1, half * 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
    
    def update(frame):
        state = trajectory[frame]
        pos = np.asarray(state.pos)
        energy = np.asarray(state.energy)
        
        scatter.set_offsets(pos[:, :2])
        scatter.set_array(energy)
        scatter.set_cmap('RdYlGn')
        
        time_text.set_text(f'Time: {float(state.time):.2f}s')
        return scatter, time_text
    
    anim = FuncAnimation(
        fig, update,
        frames=len(trajectory),
        interval=interval,
        blit=True
    )
    
    if save_path is not None:
        writer = FFMpegWriter(fps=1000 // interval, metadata={'title': 'Swarm Simulation'})
        anim.save(str(save_path), writer=writer)
        print(f"Saved animation to {save_path}")
    
    return anim


def plot_metrics(
    metrics: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training metrics.
    
    Args:
        metrics: Dictionary of metric name -> values over time
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics), sharex=True)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Step')
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")
    
    return fig
