"""Evaluation utilities for RL agents."""

import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from scipy import stats

from ..algorithms.agents import BaseAgent
from ..environments.envs import make_env
from ..utils.device import get_device


class Evaluator:
    """Evaluator for RL agents."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        agent: Optional[BaseAgent] = None,
    ):
        """Initialize evaluator.
        
        Args:
            config: Configuration dictionary.
            agent: Pre-trained agent (optional).
        """
        self.config = config
        self.device = get_device(config.get("device", "auto"))
        
        # Create environment
        env_config = config["env"]
        self.env = make_env(env_config["name"], **env_config.get(env_config["name"], {}))
        
        self.agent = agent
        self.num_eval_episodes = config["evaluation"]["num_eval_episodes"]
    
    def evaluate(
        self, 
        agent: Optional[BaseAgent] = None,
        num_episodes: Optional[int] = None,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate agent performance.
        
        Args:
            agent: Agent to evaluate (uses self.agent if None).
            num_episodes: Number of evaluation episodes.
            render: Whether to render episodes.
            
        Returns:
            Evaluation metrics.
        """
        if agent is None:
            agent = self.agent
        
        if agent is None:
            raise ValueError("No agent provided for evaluation")
        
        if num_episodes is None:
            num_episodes = self.num_eval_episodes
        
        agent.set_training(False)
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_info = []
        
        for episode in range(num_episodes):
            reward, length, success, info = self._evaluate_episode(agent, render)
            
            episode_rewards.append(reward)
            episode_lengths.append(length)
            episode_successes.append(success)
            episode_info.append(info)
        
        agent.set_training(True)
        
        # Calculate statistics
        metrics = self._calculate_metrics(
            episode_rewards, episode_lengths, episode_successes, episode_info
        )
        
        return metrics
    
    def _evaluate_episode(
        self, 
        agent: BaseAgent, 
        render: bool = False
    ) -> Tuple[float, int, bool, Dict[str, Any]]:
        """Evaluate one episode.
        
        Args:
            agent: Agent to evaluate.
            render: Whether to render.
            
        Returns:
            Episode reward, length, success flag, and info.
        """
        state, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config["training"]["max_steps_per_episode"]):
            if render:
                self.env.render()
            
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, step_info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Determine success based on environment type
        success = self._determine_success(episode_reward, info, step_info)
        
        return episode_reward, episode_length, success, step_info
    
    def _determine_success(
        self, 
        reward: float, 
        initial_info: Dict[str, Any], 
        final_info: Dict[str, Any]
    ) -> bool:
        """Determine if episode was successful.
        
        Args:
            reward: Episode reward.
            initial_info: Initial environment info.
            final_info: Final environment info.
            
        Returns:
            Whether episode was successful.
        """
        env_name = self.config["env"]["name"]
        
        if env_name == "knapsack":
            # Success if we achieved a reasonable fraction of optimal value
            optimal_value = initial_info.get("optimal_value", 0)
            if optimal_value > 0:
                return reward >= 0.8 * optimal_value
            else:
                return reward > 0
        
        elif env_name == "tsp":
            # Success if we completed the tour
            optimal_distance = initial_info.get("optimal_distance", float('inf'))
            if optimal_distance < float('inf'):
                return reward >= -1.2 * optimal_distance  # Within 20% of optimal
            else:
                return reward > -1000  # Reasonable threshold
        
        else:
            # Generic success criterion
            return reward > 0
    
    def _calculate_metrics(
        self,
        rewards: List[float],
        lengths: List[int],
        successes: List[bool],
        info_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics.
        
        Args:
            rewards: List of episode rewards.
            lengths: List of episode lengths.
            successes: List of success flags.
            info_list: List of episode info dictionaries.
            
        Returns:
            Evaluation metrics.
        """
        rewards = np.array(rewards)
        lengths = np.array(lengths)
        successes = np.array(successes)
        
        # Basic statistics
        metrics = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "median_reward": np.median(rewards),
            
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            
            "success_rate": np.mean(successes),
            "num_episodes": len(rewards),
        }
        
        # Confidence intervals
        if len(rewards) > 1:
            ci_lower, ci_upper = stats.t.interval(
                0.95, len(rewards) - 1, 
                loc=np.mean(rewards), 
                scale=stats.sem(rewards)
            )
            metrics["reward_ci_95"] = (ci_lower, ci_upper)
        
        # Environment-specific metrics
        env_name = self.config["env"]["name"]
        
        if env_name == "knapsack":
            optimal_values = [info.get("optimal_value", 0) for info in info_list]
            if any(ov > 0 for ov in optimal_values):
                relative_rewards = [r / ov if ov > 0 else 0 for r, ov in zip(rewards, optimal_values)]
                metrics["mean_relative_reward"] = np.mean(relative_rewards)
                metrics["std_relative_reward"] = np.std(relative_rewards)
        
        elif env_name == "tsp":
            optimal_distances = [info.get("optimal_distance", float('inf')) for info in info_list]
            if any(od < float('inf') for od in optimal_distances):
                relative_distances = [abs(r) / od if od > 0 else 0 for r, od in zip(rewards, optimal_distances)]
                metrics["mean_relative_distance"] = np.mean(relative_distances)
                metrics["std_relative_distance"] = np.std(relative_distances)
        
        return metrics
    
    def compare_agents(
        self, 
        agents: Dict[str, BaseAgent],
        num_episodes: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple agents.
        
        Args:
            agents: Dictionary of agent names and instances.
            num_episodes: Number of evaluation episodes per agent.
            
        Returns:
            Comparison results.
        """
        results = {}
        
        for name, agent in agents.items():
            print(f"Evaluating agent: {name}")
            metrics = self.evaluate(agent, num_episodes)
            results[name] = metrics
        
        return results
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate evaluation report.
        
        Args:
            metrics: Evaluation metrics.
            
        Returns:
            Formatted report string.
        """
        report = f"""
Evaluation Report
================

Basic Statistics:
- Episodes: {metrics['num_episodes']}
- Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}
- Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]
- Median Reward: {metrics['median_reward']:.2f}

Episode Length:
- Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}

Success Rate:
- Success Rate: {metrics['success_rate']:.1%}

"""
        
        if "reward_ci_95" in metrics:
            ci_lower, ci_upper = metrics["reward_ci_95"]
            report += f"Confidence Intervals:\n- 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n\n"
        
        if "mean_relative_reward" in metrics:
            report += f"Relative Performance:\n- Mean Relative Reward: {metrics['mean_relative_reward']:.2f} ± {metrics['std_relative_reward']:.2f}\n\n"
        
        if "mean_relative_distance" in metrics:
            report += f"Distance Performance:\n- Mean Relative Distance: {metrics['mean_relative_distance']:.2f} ± {metrics['std_relative_distance']:.2f}\n\n"
        
        return report
