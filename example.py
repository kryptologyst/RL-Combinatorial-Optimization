#!/usr/bin/env python3
"""Simple example script demonstrating the modernized RL framework."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.environments.envs import make_env
from src.algorithms.agents import create_agent
from src.utils.device import get_device, set_seed


def main():
    """Run a simple example."""
    print("RL Combinatorial Optimization - Simple Example")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42, deterministic=True)
    
    # Create environment
    print("Creating Knapsack environment...")
    env = make_env("knapsack", num_items=5, capacity=20)
    print(f"Environment created: {env.observation_space.shape[0]} states, {env.action_space.n} actions")
    
    # Create agent
    print("Creating PPO agent...")
    agent = create_agent(
        algorithm="ppo",
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=0.001,
        device=get_device("cpu"),  # Use CPU for simple example
    )
    print("Agent created successfully")
    
    # Run a few episodes
    print("\nRunning training episodes...")
    for episode in range(5):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(10):  # Limit steps for demo
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print("\nExample completed successfully!")
    print("\nTo run the full training pipeline:")
    print("  python train.py")
    print("\nTo launch the interactive demo:")
    print("  streamlit run demo.py")


if __name__ == "__main__":
    main()
