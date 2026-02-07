# Drone Swarms

JAX-accelerated multi-robot swarm simulation using MuJoCo/MJX for differentiable physics.

## Features

- **MJX Physics**: Differentiable simulation with point-mass drone dynamics
- **GNN Perception**: Graph Neural Networks for collaborative sensing
- **Hybrid RL-MPC**: PPO for high-level goals + differentiable MPC for safe tracking
- **RAPTOR Planning**: Perception-aware trajectory replanning
- **Multi-GPU Scaling**: JAX `shard_map` for 100+ agents on 4× RTX 5090

## Installation

```bash
# Create environment
conda create -n swarm python=3.11
conda activate swarm

# Install with CUDA support
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train 10-agent swarm on coverage task
python training/train.py --config configs/env/10_agents.yaml

# Evaluate with visualization
python eval/evaluate.py --checkpoint checkpoints/latest.pkl --render

# Run benchmarks
python benchmarks/throughput.py --agents 10,25,50,100
```

## Project Structure

```
drone-swarms/
├── swarm/              # Core simulation engine
│   ├── envs/           # MJX environments
│   ├── perception/     # GNN-based sensing
│   ├── control/        # PPO + MPC
│   ├── planning/       # RAPTOR replanning
│   └── comms/          # Communication modeling
├── training/           # Training infrastructure
├── eval/               # Evaluation & scenarios
├── benchmarks/         # Performance testing
└── configs/            # Hydra configs
```

## License

MIT
