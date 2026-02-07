#!/usr/bin/env python3
"""
Project 626: RL for Combinatorial Optimization - Legacy Example

This file contains the original implementation for backward compatibility.
For the modern, research-ready implementation, see:
- train.py: Main training script
- demo.py: Interactive demo
- example.py: Simple usage example

The modern implementation includes:
- Gymnasium-compatible environments
- Modern RL algorithms (REINFORCE, PPO)
- Comprehensive evaluation framework
- Interactive visualization
- Type hints and documentation
- Reproducible research practices

WARNING: This is a research/educational demo. Do not use for production control of real systems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional


class CombinatorialOptimizationModel(nn.Module):
    """Legacy neural network model for RL in combinatorial optimization."""
    
    def __init__(self, input_size: int, output_size: int):
        super(CombinatorialOptimizationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class CombinatorialOptimizationAgent:
    """Legacy RL agent for combinatorial optimization."""
    
    def __init__(
        self, 
        model: nn.Module, 
        learning_rate: float = 0.001, 
        gamma: float = 0.99, 
        epsilon: float = 1.0, 
        epsilon_decay: float = 0.995
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.criterion = nn.MSELoss()
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state))  # Random action
        else:
            action_probs = self.model(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(action_probs).item()
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> float:
        """Update policy using policy gradient."""
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        log_prob = torch.log(action_probs[action])
        loss = -log_prob * reward  # Policy gradient
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if done:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


class KnapsackEnv:
    """Legacy Knapsack Problem environment."""
    
    def __init__(self, values: np.ndarray, weights: np.ndarray, capacity: int):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.num_items = len(values)
        self.state = np.zeros(self.num_items)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = np.zeros(self.num_items)
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Take a step in the environment."""
        if self.state[action] == 1:
            return self.state, 0, True  # Item already picked
        
        self.state[action] = 1  # Mark item as picked
        weight = np.sum(self.state * self.weights)
        
        if weight > self.capacity:
            self.state[action] = 0  # Undo pick if weight exceeds capacity
            return self.state, -1, False  # Penalty for exceeding capacity
        
        reward = np.sum(self.state * self.values)  # Reward is total value
        done = np.all(self.state)  # Done if all items are selected
        
        return self.state, reward, done


def main():
    """Run the legacy example."""
    print("Project 626: RL for Combinatorial Optimization - Legacy Example")
    print("=" * 60)
    print("WARNING: This is a research/educational demo.")
    print("Do not use for production control of real systems.")
    print("=" * 60)
    
    # Initialize the environment and RL agent
    values = np.array([10, 40, 30, 50])  # Item values
    weights = np.array([5, 4, 6, 3])    # Item weights
    capacity = 10                       # Knapsack capacity
    
    env = KnapsackEnv(values, weights, capacity)
    model = CombinatorialOptimizationModel(input_size=env.num_items, output_size=env.num_items)
    agent = CombinatorialOptimizationAgent(model)
    
    print(f"Environment: {env.num_items} items, capacity {env.capacity}")
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    # Calculate greedy optimal value
    items = list(zip(values, weights))
    items.sort(key=lambda x: x[0]/x[1], reverse=True)  # Sort by value/weight ratio
    greedy_value = 0
    greedy_weight = 0
    for value, weight in items:
        if greedy_weight + weight <= capacity:
            greedy_value += value
            greedy_weight += weight
    print(f"Optimal value (greedy): {greedy_value}")
    
    # Train the agent using policy gradient
    num_episodes = 1000
    print(f"\nTraining for {num_episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Update the agent using policy gradient
            loss = agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, Loss: {loss:.4f}")
    
    # Evaluate the agent after training
    print("\nEvaluating trained agent...")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 10:  # Prevent infinite loops
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
    
    print(f"Final evaluation reward: {total_reward:.2f}")
    print(f"Selected items: {env.state}")
    print(f"Total value: {np.sum(env.state * env.values):.2f}")
    print(f"Total weight: {np.sum(env.state * env.weights):.2f}")
    
    print("\n" + "=" * 60)
    print("Legacy example completed!")
    print("For the modern implementation, see:")
    print("  python train.py      # Full training pipeline")
    print("  python demo.py       # Interactive demo")
    print("  python example.py    # Simple usage example")
    print("=" * 60)


if __name__ == "__main__":
    main()
