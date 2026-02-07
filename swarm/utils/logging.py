"""Logging utilities for experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np


class MetricsLogger:
    """Simple metrics logger with optional W&B integration."""
    
    def __init__(
        self,
        log_dir: str | Path = "logs",
        experiment_name: str | None = None,
        use_wandb: bool = False,
        wandb_project: str = "drone-swarms",
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: dict[str, list[float]] = {}
        self.step = 0
        
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=self.experiment_name)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, falling back to local logging")
                self.use_wandb = False
    
    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics at current step.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (auto-increments if None)
        """
        if step is not None:
            self.step = step
        
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(float(value))
        
        if self.use_wandb:
            self.wandb.log(metrics, step=self.step)
        
        self.step += 1
    
    def save(self) -> None:
        """Save metrics to disk."""
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
    
    def get_metrics(self) -> dict[str, list[float]]:
        """Get all logged metrics."""
        return self.metrics


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to YAML file."""
    import yaml
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)
