"""Logging utilities for the RL project."""

import logging
import os
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger for training and evaluation metrics."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize logger.
        
        Args:
            log_dir: Directory for log files.
            use_tensorboard: Whether to use TensorBoard logging.
            use_wandb: Whether to use Weights & Biases logging.
            wandb_project: W&B project name.
            wandb_config: W&B configuration dictionary.
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup TensorBoard
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        else:
            self.tb_writer = None
            
        # Setup W&B
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "rl-combinatorial-optimization",
                    config=wandb_config,
                    dir=log_dir,
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None
            
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value.
        
        Args:
            name: Metric name.
            value: Metric value.
            step: Training step.
        """
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
            
        if self.wandb:
            self.wandb.log({name: value}, step=step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar values.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Training step.
        """
        if self.tb_writer:
            self.tb_writer.add_scalars("metrics", metrics, step)
            
        if self.wandb:
            self.wandb.log(metrics, step=step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram.
        
        Args:
            name: Histogram name.
            values: Values to create histogram from.
            step: Training step.
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
            
        if self.wandb:
            self.wandb.finish()
