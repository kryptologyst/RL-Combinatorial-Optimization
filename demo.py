"""Interactive demo for RL Combinatorial Optimization."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.algorithms.agents import create_agent
from src.environments.envs import make_env, KnapsackEnv, TSPEnv
from src.utils.device import get_device


class RLCombinatorialOptimizationDemo:
    """Interactive demo for RL Combinatorial Optimization."""
    
    def __init__(self):
        """Initialize the demo."""
        st.set_page_config(
            page_title="RL Combinatorial Optimization Demo",
            page_icon="🧮",
            layout="wide"
        )
        
        self.device = get_device("cpu")  # Use CPU for demo
        
        # Initialize session state
        if "env" not in st.session_state:
            st.session_state.env = None
        if "agent" not in st.session_state:
            st.session_state.agent = None
        if "episode_data" not in st.session_state:
            st.session_state.episode_data = []
    
    def run(self):
        """Run the demo."""
        st.title("🧮 RL Combinatorial Optimization Demo")
        
        st.markdown("""
        This demo showcases Reinforcement Learning algorithms applied to combinatorial optimization problems.
        
        **WARNING**: This is a research/educational demo. Do not use for production control of real systems.
        """)
        
        # Sidebar for configuration
        self._create_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Environment", "Training", "Evaluation", "Visualization"])
        
        with tab1:
            self._environment_tab()
        
        with tab2:
            self._training_tab()
        
        with tab3:
            self._evaluation_tab()
        
        with tab4:
            self._visualization_tab()
    
    def _create_sidebar(self):
        """Create sidebar configuration."""
        st.sidebar.header("Configuration")
        
        # Environment selection
        env_name = st.sidebar.selectbox(
            "Environment",
            ["knapsack", "tsp"],
            help="Select the combinatorial optimization problem"
        )
        
        if env_name == "knapsack":
            num_items = st.sidebar.slider("Number of Items", 5, 20, 10)
            capacity = st.sidebar.slider("Knapsack Capacity", 20, 100, 50)
            
            env_config = {
                "name": "knapsack",
                "knapsack": {
                    "num_items": num_items,
                    "capacity": capacity,
                    "value_range": [1, 100],
                    "weight_range": [1, 20],
                }
            }
        
        elif env_name == "tsp":
            num_cities = st.sidebar.slider("Number of Cities", 5, 15, 10)
            
            env_config = {
                "name": "tsp",
                "tsp": {
                    "num_cities": num_cities,
                    "city_range": [0, 100],
                }
            }
        
        # Algorithm selection
        algorithm = st.sidebar.selectbox(
            "Algorithm",
            ["reinforce", "ppo"],
            help="Select the RL algorithm"
        )
        
        # Training parameters
        st.sidebar.subheader("Training Parameters")
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
        num_episodes = st.sidebar.slider("Number of Episodes", 100, 5000, 1000)
        
        # Store configuration
        st.session_state.config = {
            "env": env_config,
            "training": {
                "algorithm": algorithm,
                "learning_rate": learning_rate,
                "num_episodes": num_episodes,
                "max_steps_per_episode": 100,
                "gamma": 0.99,
            },
            "model": {
                "hidden_sizes": [128, 128],
                "activation": "relu",
            },
            "device": "cpu",
        }
    
    def _environment_tab(self):
        """Environment configuration and visualization tab."""
        st.header("Environment Configuration")
        
        config = st.session_state.config
        env_name = config["env"]["name"]
        
        # Create environment
        if st.button("Create Environment") or st.session_state.env is None:
            env_config = config["env"]
            env = make_env(env_name, **env_config.get(env_name, {}))
            st.session_state.env = env
            
            st.success(f"Created {env_name.upper()} environment!")
        
        if st.session_state.env is not None:
            env = st.session_state.env
            
            # Environment info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Environment Properties")
                st.write(f"**Type**: {env_name.upper()}")
                st.write(f"**State Space**: {env.observation_space.shape}")
                st.write(f"**Action Space**: {env.action_space.n}")
                
                if env_name == "knapsack":
                    st.write(f"**Items**: {env.num_items}")
                    st.write(f"**Capacity**: {env.capacity}")
                    st.write(f"**Values**: {env.values}")
                    st.write(f"**Weights**: {env.weights}")
                
                elif env_name == "tsp":
                    st.write(f"**Cities**: {env.num_cities}")
                    st.write(f"**City Coordinates**: {env.cities}")
            
            with col2:
                st.subheader("Environment Visualization")
                
                if env_name == "knapsack":
                    self._plot_knapsack_problem(env)
                elif env_name == "tsp":
                    self._plot_tsp_problem(env)
    
    def _training_tab(self):
        """Training tab."""
        st.header("Agent Training")
        
        if st.session_state.env is None:
            st.warning("Please create an environment first!")
            return
        
        config = st.session_state.config
        env = st.session_state.env
        
        # Create agent
        if st.button("Create Agent") or st.session_state.agent is None:
            agent = create_agent(
                algorithm=config["training"]["algorithm"],
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                **config["training"].get(config["training"]["algorithm"], {}),
                device=self.device,
            )
            st.session_state.agent = agent
            
            st.success(f"Created {config['training']['algorithm'].upper()} agent!")
        
        if st.session_state.agent is not None:
            agent = st.session_state.agent
            
            # Training controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Controls")
                
                if st.button("Start Training"):
                    self._train_agent(agent, env, config)
            
            with col2:
                st.subheader("Agent Info")
                st.write(f"**Algorithm**: {config['training']['algorithm'].upper()}")
                st.write(f"**Learning Rate**: {config['training']['learning_rate']}")
                st.write(f"**Episodes**: {config['training']['num_episodes']}")
                
                if hasattr(agent, 'policy'):
                    model_size = sum(p.numel() for p in agent.policy.parameters())
                else:
                    model_size = sum(p.numel() for p in agent.ac_net.parameters())
                st.write(f"**Model Size**: {model_size:,} parameters")
    
    def _evaluation_tab(self):
        """Evaluation tab."""
        st.header("Agent Evaluation")
        
        if st.session_state.agent is None:
            st.warning("Please create and train an agent first!")
            return
        
        agent = st.session_state.agent
        env = st.session_state.env
        config = st.session_state.config
        
        # Evaluation controls
        col1, col2 = st.columns(2)
        
        with col1:
            num_eval_episodes = st.slider("Evaluation Episodes", 1, 100, 10)
            
            if st.button("Run Evaluation"):
                self._evaluate_agent(agent, env, num_eval_episodes)
        
        with col2:
            if st.button("Run Single Episode"):
                self._run_single_episode(agent, env)
    
    def _visualization_tab(self):
        """Visualization tab."""
        st.header("Training Visualization")
        
        if not st.session_state.episode_data:
            st.info("No training data available. Please train an agent first.")
            return
        
        # Plot training curves
        self._plot_training_curves()
    
    def _plot_knapsack_problem(self, env: KnapsackEnv):
        """Plot knapsack problem visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar chart of items
        x = np.arange(env.num_items)
        width = 0.35
        
        ax.bar(x - width/2, env.values, width, label='Values', alpha=0.8)
        ax.bar(x + width/2, env.weights, width, label='Weights', alpha=0.8)
        
        ax.set_xlabel('Items')
        ax.set_ylabel('Value/Weight')
        ax.set_title('Knapsack Problem: Item Values and Weights')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Item {i}' for i in range(env.num_items)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add capacity line
        ax.axhline(y=env.capacity, color='red', linestyle='--', 
                  label=f'Capacity: {env.capacity}')
        ax.legend()
        
        st.pyplot(fig)
    
    def _plot_tsp_problem(self, env: TSPEnv):
        """Plot TSP problem visualization."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot cities
        ax.scatter(env.cities[:, 0], env.cities[:, 1], 
                  c='red', s=100, alpha=0.8, zorder=5)
        
        # Add city labels
        for i, (x, y) in enumerate(env.cities):
            ax.annotate(f'C{i}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        # Plot optimal tour (nearest neighbor heuristic)
        optimal_distance = env._get_optimal_distance()
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'TSP Problem: {env.num_cities} Cities\n'
                    f'Optimal Distance: {optimal_distance:.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        st.pyplot(fig)
    
    def _train_agent(self, agent, env, config):
        """Train the agent."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        episode_rewards = []
        
        for episode in range(config["training"]["num_episodes"]):
            # Training episode
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(config["training"]["max_steps_per_episode"]):
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Update agent
            batch = {
                "states": [state],
                "actions": [action],
                "rewards": [reward],
            }
            agent.update(batch)
            
            episode_rewards.append(episode_reward)
            
            # Update progress
            progress = (episode + 1) / config["training"]["num_episodes"]
            progress_bar.progress(progress)
            status_text.text(f"Episode {episode + 1}/{config['training']['num_episodes']}: "
                           f"Reward = {episode_reward:.2f}")
        
        # Store training data
        st.session_state.episode_data = episode_rewards
        
        st.success("Training completed!")
        
        # Plot training curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(episode_rewards, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    def _evaluate_agent(self, agent, env, num_episodes):
        """Evaluate the agent."""
        agent.set_training(False)
        
        eval_rewards = []
        progress_bar = st.progress(0)
        config = st.session_state.config
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(config["training"]["max_steps_per_episode"]):
                action, _ = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            progress_bar.progress((episode + 1) / num_episodes)
        
        agent.set_training(True)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Evaluation Results")
            st.write(f"**Mean Reward**: {np.mean(eval_rewards):.2f}")
            st.write(f"**Std Reward**: {np.std(eval_rewards):.2f}")
            st.write(f"**Min Reward**: {np.min(eval_rewards):.2f}")
            st.write(f"**Max Reward**: {np.max(eval_rewards):.2f}")
        
        with col2:
            # Plot evaluation results
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(eval_rewards, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Episode Reward')
            ax.set_ylabel('Frequency')
            ax.set_title('Evaluation Reward Distribution')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    def _run_single_episode(self, agent, env):
        """Run a single episode with visualization."""
        agent.set_training(False)
        config = st.session_state.config
        
        state, info = env.reset()
        episode_reward = 0
        episode_steps = []
        
        for step in range(config["training"]["max_steps_per_episode"]):
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            episode_steps.append({
                "step": step,
                "action": action,
                "reward": reward,
                "state": state.copy(),
                "info": step_info,
            })
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        agent.set_training(True)
        
        # Display episode details
        st.subheader("Episode Details")
        st.write(f"**Total Reward**: {episode_reward:.2f}")
        st.write(f"**Steps**: {len(episode_steps)}")
        
        # Show step-by-step details
        if st.checkbox("Show Step Details"):
            for step_data in episode_steps:
                st.write(f"Step {step_data['step']}: Action={step_data['action']}, "
                        f"Reward={step_data['reward']:.2f}")
    
    def _plot_training_curves(self):
        """Plot training curves."""
        episode_data = st.session_state.episode_data
        
        if not episode_data:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Moving Average (100)', 
                          'Reward Distribution', 'Cumulative Rewards'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Episode rewards
        fig.add_trace(
            go.Scatter(y=episode_data, mode='lines', name='Episode Rewards'),
            row=1, col=1
        )
        
        # Moving average
        if len(episode_data) >= 100:
            moving_avg = np.convolve(episode_data, np.ones(100)/100, mode='valid')
            fig.add_trace(
                go.Scatter(y=moving_avg, mode='lines', name='Moving Average'),
                row=1, col=2
            )
        
        # Reward distribution
        fig.add_trace(
            go.Histogram(x=episode_data, name='Reward Distribution'),
            row=2, col=1
        )
        
        # Cumulative rewards
        cumulative_rewards = np.cumsum(episode_data)
        fig.add_trace(
            go.Scatter(y=cumulative_rewards, mode='lines', name='Cumulative Rewards'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main demo function."""
    demo = RLCombinatorialOptimizationDemo()
    demo.run()


if __name__ == "__main__":
    main()
