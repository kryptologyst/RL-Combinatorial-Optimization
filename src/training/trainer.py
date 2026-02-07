"""Training utilities for RL agents."""

import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from ..algorithms.agents import BaseAgent, create_agent
from ..environments.envs import make_env
from ..utils.device import get_device, set_seed
from ..utils.logging import Logger


class Trainer:
    """Trainer for RL agents."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger or Logger()
        
        # Set device and seed
        self.device = get_device(config.get("device", "auto"))
        set_seed(config.get("seed", 42), config.get("deterministic", True))
        
        # Create environment
        env_config = config["env"]
        self.env = make_env(env_config["name"], **env_config.get(env_config["name"], {}))
        
        # Create agent
        agent_config = config["training"]
        self.agent = create_agent(
            algorithm=agent_config["algorithm"],
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            **agent_config.get(agent_config["algorithm"], {}),
            device=self.device,
        )
        
        # Training parameters
        self.num_episodes = agent_config["num_episodes"]
        self.max_steps_per_episode = agent_config["max_steps_per_episode"]
        self.eval_frequency = config["evaluation"]["eval_frequency"]
        self.save_frequency = config["evaluation"]["save_frequency"]
        self.log_frequency = config["evaluation"]["log_frequency"]
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        
        # Create save directory
        self.save_dir = os.path.join("checkpoints", config.get("experiment_name", "default"))
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self) -> Dict[str, List[float]]:
        """Train the agent.
        
        Returns:
            Training metrics.
        """
        self.logger.info(f"Starting training for {self.num_episodes} episodes")
        
        best_reward = float('-inf')
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            # Training episode
            episode_reward, episode_length, metrics = self._train_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if metrics:
                self.training_metrics.append(metrics)
            
            # Logging
            if episode % self.log_frequency == 0:
                self._log_episode(episode, episode_reward, episode_length, metrics)
            
            # Evaluation
            if episode % self.eval_frequency == 0:
                eval_reward = self._evaluate()
                self.logger.info(f"Episode {episode}: Eval reward = {eval_reward:.2f}")
                
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self._save_checkpoint(episode, is_best=True)
            
            # Save checkpoint
            if episode % self.save_frequency == 0:
                self._save_checkpoint(episode, is_best=False)
        
        self.logger.info("Training completed")
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_metrics": self.training_metrics,
        }
    
    def _train_episode(self) -> Tuple[float, int, Optional[Dict[str, float]]]:
        """Train for one episode.
        
        Returns:
            Episode reward, length, and training metrics.
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Episode data
        states = []
        actions = []
        rewards = []
        
        for step in range(self.max_steps_per_episode):
            action, log_prob = self.agent.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update agent
        batch = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
        }
        
        metrics = self.agent.update(batch)
        
        return episode_reward, episode_length, metrics
    
    def _evaluate(self, num_episodes: int = 10) -> float:
        """Evaluate the agent.
        
        Args:
            num_episodes: Number of evaluation episodes.
            
        Returns:
            Average evaluation reward.
        """
        self.agent.set_training(False)
        
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps_per_episode):
                action, _ = self.agent.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        self.agent.set_training(True)
        
        return np.mean(eval_rewards)
    
    def _log_episode(
        self, 
        episode: int, 
        reward: float, 
        length: int, 
        metrics: Optional[Dict[str, float]]
    ) -> None:
        """Log episode information.
        
        Args:
            episode: Episode number.
            reward: Episode reward.
            length: Episode length.
            metrics: Training metrics.
        """
        # Log scalars
        self.logger.log_scalar("episode/reward", reward, episode)
        self.logger.log_scalar("episode/length", length, episode)
        
        if metrics:
            for key, value in metrics.items():
                self.logger.log_scalar(f"training/{key}", value, episode)
        
        # Log moving averages
        if len(self.episode_rewards) >= 100:
            recent_rewards = self.episode_rewards[-100:]
            self.logger.log_scalar("episode/reward_100", np.mean(recent_rewards), episode)
        
        # Print to console
        if metrics:
            self.logger.info(
                f"Episode {episode}: Reward = {reward:.2f}, Length = {length}, "
                f"Policy Loss = {metrics.get('policy_loss', 0):.4f}"
            )
        else:
            self.logger.info(f"Episode {episode}: Reward = {reward:.2f}, Length = {length}")
    
    def _save_checkpoint(self, episode: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            episode: Episode number.
            is_best: Whether this is the best model.
        """
        checkpoint = {
            "episode": episode,
            "agent_state_dict": self.agent.policy.state_dict() if hasattr(self.agent, 'policy') else self.agent.ac_net.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "config": self.config,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pt"))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_{episode}.pt"))
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if hasattr(self.agent, 'policy'):
            self.agent.policy.load_state_dict(checkpoint["agent_state_dict"])
        else:
            self.agent.ac_net.load_state_dict(checkpoint["agent_state_dict"])
        
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_lengths = checkpoint.get("episode_lengths", [])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
