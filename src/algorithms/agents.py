"""Reinforcement Learning algorithms for combinatorial optimization."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ..models.networks import ActorCriticNetwork, PolicyNetwork, ValueNetwork
from ..utils.device import get_device
from ..utils.logging import Logger


class BaseAgent(ABC):
    """Base class for RL agents."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        """Initialize base agent.
        
        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            learning_rate: Learning rate.
            gamma: Discount factor.
            device: Device to use.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = get_device(device)
        
        self.training = True
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> Tuple[int, Optional[torch.Tensor]]:
        """Select action given state.
        
        Args:
            state: Current state.
            
        Returns:
            Selected action and optional log probability.
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent parameters.
        
        Args:
            batch: Training batch.
            
        Returns:
            Training metrics.
        """
        pass
    
    def set_training(self, training: bool) -> None:
        """Set training mode.
        
        Args:
            training: Whether to set training mode.
        """
        self.training = training
        if hasattr(self, 'policy'):
            self.policy.train(training)


class REINFORCEAgent(BaseAgent):
    """REINFORCE (Policy Gradient) agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [128, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        baseline_type: str = "moving_average",
        baseline_decay: float = 0.99,
        device: str = "auto",
    ):
        """Initialize REINFORCE agent.
        
        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_sizes: Hidden layer sizes.
            learning_rate: Learning rate.
            gamma: Discount factor.
            baseline_type: Type of baseline ("none", "moving_average", "neural_network").
            baseline_decay: Decay rate for moving average baseline.
            device: Device to use.
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma, device)
        
        self.baseline_type = baseline_type
        self.baseline_decay = baseline_decay
        
        # Policy network
        self.policy = PolicyNetwork(
            input_size=state_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Baseline
        if baseline_type == "neural_network":
            self.baseline_net = ValueNetwork(
                input_size=state_dim,
                hidden_sizes=hidden_sizes,
            ).to(self.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline_net.parameters(), lr=learning_rate
            )
        elif baseline_type == "moving_average":
            self.baseline_value = 0.0
        
        self.log_probs = []
        self.rewards = []
        self.states = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using policy.
        
        Args:
            state: Current state.
            
        Returns:
            Selected action and log probability.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            action_probs = self.policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        if self.training:
            self.log_probs.append(log_prob)
            self.states.append(state_tensor.squeeze(0))
        
        return action.item(), log_prob
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update policy using REINFORCE.
        
        Args:
            batch: Training batch.
            
        Returns:
            Training metrics.
        """
        if not self.log_probs:
            return {"policy_loss": 0.0, "baseline_loss": 0.0}
        
        # Calculate returns
        returns = self._calculate_returns(batch["rewards"])
        
        # Calculate baseline
        if self.baseline_type == "neural_network":
            states = torch.stack(self.states).to(self.device)
            baseline_values = self.baseline_net(states)
            advantages = returns - baseline_values.detach()
            
            # Update baseline network
            baseline_loss = nn.MSELoss()(baseline_values, returns)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        elif self.baseline_type == "moving_average":
            advantages = returns - self.baseline_value
            self.baseline_value = (
                self.baseline_decay * self.baseline_value + 
                (1 - self.baseline_decay) * returns.mean().item()
            )
        else:
            advantages = returns
        
        # Update policy
        log_probs = torch.stack(self.log_probs)
        policy_loss = -(log_probs * advantages).mean()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.states.clear()
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "baseline_loss": baseline_loss.item() if self.baseline_type == "neural_network" else 0.0,
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }
        
        return metrics
    
    def _calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """Calculate discounted returns.
        
        Args:
            rewards: List of rewards.
            
        Returns:
            Discounted returns.
        """
        returns = []
        discounted_reward = 0
        
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        return torch.FloatTensor(returns).to(self.device)


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [128, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: str = "auto",
    ):
        """Initialize PPO agent.
        
        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_sizes: Hidden layer sizes.
            learning_rate: Learning rate.
            gamma: Discount factor.
            clip_ratio: PPO clipping ratio.
            value_loss_coef: Value loss coefficient.
            entropy_coef: Entropy coefficient.
            max_grad_norm: Maximum gradient norm.
            ppo_epochs: Number of PPO epochs.
            device: Device to use.
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma, device)
        
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        
        # Actor-Critic network
        self.ac_net = ActorCriticNetwork(
            input_size=state_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        
        self.buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "advantages": [],
            "returns": [],
        }
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action using actor-critic network.
        
        Args:
            state: Current state.
            
        Returns:
            Selected action and log probability.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.ac_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        if self.training:
            self.buffer["states"].append(state_tensor.squeeze(0))
            self.buffer["actions"].append(action.item())
            self.buffer["values"].append(value.item())
            self.buffer["log_probs"].append(log_prob.item())
        
        return action.item(), log_prob
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent using PPO.
        
        Args:
            batch: Training batch.
            
        Returns:
            Training metrics.
        """
        if not self.buffer["states"]:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        
        # Calculate advantages and returns
        rewards = batch["rewards"]
        values = torch.FloatTensor(self.buffer["values"]).to(self.device)
        
        returns = self._calculate_returns(rewards)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.stack(self.buffer["states"]).to(self.device)
        actions = torch.LongTensor(self.buffer["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer["log_probs"]).to(self.device)
        
        # PPO updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Forward pass
            action_probs, values = self.ac_net(states)
            dist = Categorical(action_probs)
            
            # Calculate losses
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss (PPO)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Total loss
            total_loss = (
                policy_loss + 
                self.value_loss_coef * value_loss - 
                self.entropy_coef * entropy
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy.item()
        
        # Clear buffer
        for key in self.buffer:
            self.buffer[key].clear()
        
        metrics = {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy_loss": total_entropy_loss / self.ppo_epochs,
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }
        
        return metrics
    
    def _calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """Calculate discounted returns.
        
        Args:
            rewards: List of rewards.
            
        Returns:
            Discounted returns.
        """
        returns = []
        discounted_reward = 0
        
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        return torch.FloatTensor(returns).to(self.device)


def create_agent(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    **kwargs
) -> BaseAgent:
    """Create agent by algorithm name.
    
    Args:
        algorithm: Algorithm name.
        state_dim: State dimension.
        action_dim: Action dimension.
        **kwargs: Algorithm-specific arguments.
        
    Returns:
        Agent instance.
    """
    if algorithm == "reinforce":
        return REINFORCEAgent(state_dim, action_dim, **kwargs)
    elif algorithm == "ppo":
        return PPOAgent(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
