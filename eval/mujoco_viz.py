"""MuJoCo 3D visualization for swarm simulation.

Provides real-time 3D rendering using MuJoCo's native viewer.
"""

from __future__ import annotations

from typing import NamedTuple
from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer

from swarm.envs.dynamics import SwarmState


# Get path to scene XML
SCENE_XML = Path(__file__).parent.parent / "assets" / "mujoco" / "swarm_scene.xml"


class MuJoCoRendererConfig(NamedTuple):
    """MuJoCo renderer configuration."""
    width: int = 640   # Reduced for Windows framebuffer compatibility
    height: int = 480
    camera_distance: float = 150.0
    camera_azimuth: float = 45.0
    camera_elevation: float = -30.0
    drone_size: float = 1.5
    show_trails: bool = True
    trail_length: int = 20
    use_mujoco_renderer: bool = False  # Set True to try MuJoCo native (may fail on Windows)


def create_swarm_scene_xml(num_agents: int, arena_size: float = 100.0) -> str:
    """Generate MuJoCo XML for swarm scene with N drones.
    
    Args:
        num_agents: Number of drones
        arena_size: Arena size in meters
        
    Returns:
        XML string for MuJoCo model
    """
    half = arena_size / 2
    
    # Generate drone bodies
    drone_bodies = []
    for i in range(num_agents):
        # Random initial position
        x = np.random.uniform(-half * 0.8, half * 0.8)
        y = np.random.uniform(-half * 0.8, half * 0.8)
        z = np.random.uniform(5, 30)
        
        # Color based on agent index
        hue = i / num_agents
        r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
        
        drone_xml = f'''
    <body name="drone_{i}" pos="{x:.2f} {y:.2f} {z:.2f}">
      <freejoint name="drone_{i}_joint"/>
      <geom name="drone_{i}_body" type="sphere" size="1.5" rgba="{r:.2f} {g:.2f} {b:.2f} 1"/>
      <geom name="drone_{i}_rotor1" type="cylinder" pos="1.5 0 0.2" size="0.8 0.05" rgba="0.2 0.2 0.2 0.8"/>
      <geom name="drone_{i}_rotor2" type="cylinder" pos="-1.5 0 0.2" size="0.8 0.05" rgba="0.2 0.2 0.2 0.8"/>
      <geom name="drone_{i}_rotor3" type="cylinder" pos="0 1.5 0.2" size="0.8 0.05" rgba="0.2 0.2 0.2 0.8"/>
      <geom name="drone_{i}_rotor4" type="cylinder" pos="0 -1.5 0.2" size="0.8 0.05" rgba="0.2 0.2 0.2 0.8"/>
    </body>'''
        drone_bodies.append(drone_xml)
    
    xml = f'''<mujoco model="swarm_{num_agents}">
  <compiler angle="degree" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.02" integrator="implicit"/>
  
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker" rgb1="0.35 0.55 0.35" rgb2="0.25 0.45 0.25" width="256" height="256"/>
    <material name="ground_mat" texture="ground_tex" texrepeat="25 25" reflectance="0.1"/>
  </asset>
  
  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" size="{half * 1.5} {half * 1.5} 0.1" material="ground_mat"/>
    
    <!-- Arena boundary (visual only) -->
    <geom name="boundary_x+" type="box" pos="{half} 0 2" size="0.3 {half} 2" rgba="0.8 0.2 0.2 0.3" contype="0" conaffinity="0"/>
    <geom name="boundary_x-" type="box" pos="{-half} 0 2" size="0.3 {half} 2" rgba="0.8 0.2 0.2 0.3" contype="0" conaffinity="0"/>
    <geom name="boundary_y+" type="box" pos="0 {half} 2" size="{half} 0.3 2" rgba="0.8 0.2 0.2 0.3" contype="0" conaffinity="0"/>
    <geom name="boundary_y-" type="box" pos="0 {-half} 2" size="{half} 0.3 2" rgba="0.8 0.2 0.2 0.3" contype="0" conaffinity="0"/>
    
    <!-- Lighting -->
    <light name="sun" pos="0 0 100" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" pos="50 50 50" dir="-1 -1 -1" diffuse="0.3 0.3 0.4" specular="0.1 0.1 0.1"/>
    
    <!-- Drones -->
    {"".join(drone_bodies)}
  </worldbody>
</mujoco>'''
    
    return xml


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB."""
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


class MuJoCoRenderer:
    """MuJoCo-based 3D renderer for swarm visualization.
    
    Args:
        num_agents: Number of drones
        arena_size: Arena size
        config: Renderer configuration
    """
    
    def __init__(
        self,
        num_agents: int,
        arena_size: float = 100.0,
        config: MuJoCoRendererConfig | None = None,
    ):
        self.num_agents = num_agents
        self.arena_size = arena_size
        self.config = config or MuJoCoRendererConfig()
        
        # Generate XML
        xml_str = create_swarm_scene_xml(num_agents, arena_size)
        
        # Create model and data
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        
        # Persistent renderer for video export (created on demand)
        self._renderer = None
        
        # Viewer (created on demand)
        self.viewer = None
        
        print(f"MuJoCo renderer initialized: {num_agents} drones, {arena_size}m arena")
    
    def update(self, state: SwarmState):
        """Update drone positions from swarm state.
        
        Args:
            state: SwarmState with positions and velocities
        """
        pos = np.array(state.pos)
        vel = np.array(state.vel)
        
        for i in range(min(self.num_agents, pos.shape[0])):
            # Each drone has 7 qpos values (3 pos + 4 quat) for freejoint
            qpos_idx = i * 7
            
            # Set position
            self.data.qpos[qpos_idx:qpos_idx + 3] = pos[i]
            
            # Set quaternion (identity for now - could add orientation)
            self.data.qpos[qpos_idx + 3:qpos_idx + 7] = [1, 0, 0, 0]
            
            # Set velocity
            qvel_idx = i * 6
            self.data.qvel[qvel_idx:qvel_idx + 3] = vel[i]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def render_frame(self) -> np.ndarray:
        """Render a single frame.
        
        Returns:
            RGB image array
        """
        # Create persistent renderer on first call
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.config.height, self.config.width)
        
        # Update scene
        self._renderer.update_scene(self.data)
        
        # Render
        img = self._renderer.render()
        
        return img
    
    def close(self):
        """Clean up renderer resources."""
        if self._renderer is not None:
            try:
                self._renderer.close()
            except:
                pass  # Ignore close errors
            self._renderer = None
    
    def launch_viewer(self):
        """Launch interactive MuJoCo viewer."""
        print("Launching MuJoCo viewer...")
        print("  Close the viewer window to return.")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            
            # Set initial camera
            viewer.cam.distance = self.config.camera_distance
            viewer.cam.azimuth = self.config.camera_azimuth
            viewer.cam.elevation = self.config.camera_elevation
            viewer.cam.lookat[:] = [0, 0, 10]
            
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)
    
    def animate_trajectory(
        self,
        trajectory: list[SwarmState],
        fps: int = 30,
        save_path: str | None = None,
    ):
        """Animate a trajectory.
        
        Args:
            trajectory: List of SwarmStates
            fps: Frames per second
            save_path: Optional path to save video
        """
        if save_path:
            if self.config.use_mujoco_renderer:
                # Try MuJoCo native renderer (may fail on Windows)
                try:
                    self._save_mujoco_video(trajectory, save_path, fps)
                except Exception as e:
                    print(f"MuJoCo native renderer failed: {e}")
                    print("Falling back to matplotlib 3D...")
                    self._save_matplotlib_video(trajectory, save_path, fps)
            else:
                # Use matplotlib 3D (more compatible)
                self._save_matplotlib_video(trajectory, save_path, fps)
        else:
            # Interactive viewer
            print("Launching interactive animation...")
            
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Set camera
                viewer.cam.distance = self.config.camera_distance
                viewer.cam.azimuth = self.config.camera_azimuth
                viewer.cam.elevation = self.config.camera_elevation
                viewer.cam.lookat[:] = [0, 0, 10]
                
                frame_time = 1.0 / fps
                frame_idx = 0
                
                while viewer.is_running() and frame_idx < len(trajectory):
                    start = time.time()
                    
                    # Update state
                    self.update(trajectory[frame_idx])
                    viewer.sync()
                    
                    frame_idx += 1
                    
                    # Maintain framerate
                    elapsed = time.time() - start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                
                print("Animation complete!")
    
    def _save_mujoco_video(self, trajectory: list, save_path: str, fps: int):
        """Save video using MuJoCo's native renderer (may fail on Windows)."""
        import imageio
        
        print(f"Rendering {len(trajectory)} frames with MuJoCo...")
        
        # Create offscreen renderer
        renderer = mujoco.Renderer(
            self.model, 
            height=self.config.height, 
            width=self.config.width
        )
        
        frames = []
        for i, state in enumerate(trajectory):
            self.update(state)
            renderer.update_scene(self.data)
            frame = renderer.render()
            frames.append(frame.copy())
            
            if i % 50 == 0:
                print(f"  Frame {i}/{len(trajectory)}")
        
        # Don't call renderer.close() - causes OpenGL issues
        del renderer
        
        print(f"Saving video to {save_path}...")
        imageio.mimsave(save_path, frames, fps=fps)
        print("Done!")
    
    def _save_matplotlib_video(self, trajectory: list, save_path: str, fps: int):
        """Save video using matplotlib 3D (more compatible than MuJoCo offscreen)."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        
        print(f"Creating 3D animation with matplotlib...")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        half = self.arena_size / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(0, 50)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Swarm 3D Visualization')
        
        # Initialize scatter
        pos = np.array(trajectory[0].pos)
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_agents))
        scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=80)
        
        # Trail lines
        trail_length = 20
        history = {i: [] for i in range(self.num_agents)}
        trail_lines = []
        for i in range(self.num_agents):
            line, = ax.plot([], [], [], c=colors[i], alpha=0.4, linewidth=1)
            trail_lines.append(line)
        
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12)
        
        def update(frame):
            state = trajectory[frame]
            pos = np.array(state.pos)
            
            # Update scatter
            scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            
            # Update trails
            for i in range(self.num_agents):
                history[i].append(pos[i])
                if len(history[i]) > trail_length:
                    history[i] = history[i][-trail_length:]
                
                if len(history[i]) > 1:
                    trail = np.array(history[i])
                    trail_lines[i].set_data(trail[:, 0], trail[:, 1])
                    trail_lines[i].set_3d_properties(trail[:, 2])
            
            time_text.set_text(f'Time: {float(state.time):.2f}s')
            
            if frame % 50 == 0:
                print(f"  Frame {frame}/{len(trajectory)}")
            
            return [scatter, time_text] + trail_lines
        
        anim = FuncAnimation(
            fig, update,
            frames=len(trajectory),
            interval=1000 // fps,
            blit=False
        )
        
        print(f"Saving video to {save_path}...")
        if save_path.endswith('.mp4'):
            try:
                anim.save(save_path, writer='ffmpeg', fps=fps)
            except:
                gif_path = save_path.replace('.mp4', '.gif')
                print(f"ffmpeg not found, saving as GIF: {gif_path}")
                anim.save(gif_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='pillow', fps=fps)
        
        plt.close(fig)
        print("Done!")


def quick_3d_demo(num_agents: int = 10, num_steps: int = 200, save_path: str | None = None, use_mujoco: bool = False):
    """Quick demo of 3D visualization.
    
    Args:
        num_agents: Number of drones
        num_steps: Simulation steps
        save_path: Path to save video (optional)
        use_mujoco: Use MuJoCo native renderer (may fail on Windows)
    """
    import jax
    import jax.numpy as jnp
    from swarm.envs.mjx_env import SwarmEnv, EnvConfig
    from pathlib import Path
    
    print("=" * 50)
    print("🎬 MuJoCo 3D Demo")
    print("=" * 50)
    
    # Create environment
    config = EnvConfig(num_agents=num_agents, arena_size=100.0)
    env = SwarmEnv(config)
    
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)
    
    # Collect trajectory
    trajectory = [state]
    duration_sec = num_steps * 0.02  # dt = 0.02s
    print(f"Running {num_steps} steps ({duration_sec:.1f}s simulation)...")
    
    for step in range(num_steps):
        key, action_key = jax.random.split(key)
        actions = jax.random.normal(action_key, (num_agents, 3)) * 8.0
        result = env.step(state, actions)
        state = result.state
        trajectory.append(state)
        
        if step % 100 == 0:
            print(f"  Step {step}/{num_steps}")
    
    print("Creating 3D visualization...")
    
    # Create renderer with config
    config = MuJoCoRendererConfig(use_mujoco_renderer=use_mujoco)
    renderer = MuJoCoRenderer(num_agents, config=config)
    
    # Create output directory if saving
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Animate (save or interactive)
    renderer.animate_trajectory(trajectory, fps=30, save_path=save_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MuJoCo 3D demo")
    parser.add_argument("--agents", type=int, default=10, help="Number of drones")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps (50 steps = 1 second)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds (overrides --steps)")
    parser.add_argument("--save", type=str, default=None, help="Save video path (e.g., demos/video.mp4)")
    parser.add_argument("--mujoco", action="store_true", help="Use MuJoCo native renderer (experimental, may fail on Windows)")
    
    args = parser.parse_args()
    
    # Convert duration to steps if provided
    if args.duration:
        args.steps = int(args.duration / 0.02)  # dt = 0.02s
        print(f"Duration {args.duration}s -> {args.steps} steps")
    
    quick_3d_demo(args.agents, args.steps, args.save, args.mujoco)

