"""Utility functions for device management and reproducibility."""

import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for PyTorch operations.
    
    Args:
        device: Device specification. If None or "auto", automatically select best available device.
        
    Returns:
        PyTorch device object.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> int:
    """Get the number of parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
