"""Neural network models for combinatorial optimization."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network for combinatorial optimization problems."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        """Initialize policy network.
        
        Args:
            input_size: Size of input state.
            output_size: Size of action space.
            hidden_sizes: List of hidden layer sizes.
            activation: Activation function name.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Action probabilities.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if i < len(self.layers) - 1:  # Don't apply activation to output layer
                    x = self.activation(x)
            else:
                x = layer(x)
        
        return F.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for baseline estimation."""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        """Initialize value network.
        
        Args:
            input_size: Size of input state.
            hidden_sizes: List of hidden layer sizes.
            activation: Activation function name.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.input_size = input_size
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            State value.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if i < len(self.layers) - 1:  # Don't apply activation to output layer
                    x = self.activation(x)
            else:
                x = layer(x)
        
        return x.squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network combining policy and value networks."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        shared_layers: int = 1,
    ):
        """Initialize actor-critic network.
        
        Args:
            input_size: Size of input state.
            output_size: Size of action space.
            hidden_sizes: List of hidden layer sizes.
            activation: Activation function name.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
            shared_layers: Number of shared layers between actor and critic.
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.shared_layers = shared_layers
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared layers
        shared_layers_list = []
        prev_size = input_size
        
        for i in range(shared_layers):
            shared_layers_list.append(nn.Linear(prev_size, hidden_sizes[i]))
            if use_batch_norm:
                shared_layers_list.append(nn.BatchNorm1d(hidden_sizes[i]))
            if dropout > 0:
                shared_layers_list.append(nn.Dropout(dropout))
            prev_size = hidden_sizes[i]
        
        self.shared_layers = nn.ModuleList(shared_layers_list)
        
        # Actor head
        actor_layers = []
        for i in range(shared_layers, len(hidden_sizes)):
            actor_layers.append(nn.Linear(prev_size, hidden_sizes[i]))
            if use_batch_norm:
                actor_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            if dropout > 0:
                actor_layers.append(nn.Dropout(dropout))
            prev_size = hidden_sizes[i]
        
        actor_layers.append(nn.Linear(prev_size, output_size))
        self.actor_layers = nn.ModuleList(actor_layers)
        
        # Critic head
        critic_layers = []
        prev_size = hidden_sizes[shared_layers - 1] if shared_layers > 0 else input_size
        
        for i in range(shared_layers, len(hidden_sizes)):
            critic_layers.append(nn.Linear(prev_size, hidden_sizes[i]))
            if use_batch_norm:
                critic_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            if dropout > 0:
                critic_layers.append(nn.Dropout(dropout))
            prev_size = hidden_sizes[i]
        
        critic_layers.append(nn.Linear(prev_size, 1))
        self.critic_layers = nn.ModuleList(critic_layers)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple of (action probabilities, state value).
        """
        # Shared layers
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.activation(x)
            else:
                x = layer(x)
        
        # Actor head
        actor_x = x
        for i, layer in enumerate(self.actor_layers):
            if isinstance(layer, nn.Linear):
                actor_x = layer(actor_x)
                if i < len(self.actor_layers) - 1:  # Don't apply activation to output layer
                    actor_x = self.activation(actor_x)
            else:
                actor_x = layer(actor_x)
        
        action_probs = F.softmax(actor_x, dim=-1)
        
        # Critic head
        critic_x = x
        for i, layer in enumerate(self.critic_layers):
            if isinstance(layer, nn.Linear):
                critic_x = layer(critic_x)
                if i < len(self.critic_layers) - 1:  # Don't apply activation to output layer
                    critic_x = self.activation(critic_x)
            else:
                critic_x = layer(critic_x)
        
        value = critic_x.squeeze(-1)
        
        return action_probs, value
