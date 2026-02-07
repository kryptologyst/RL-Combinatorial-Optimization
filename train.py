#!/usr/bin/env python3
"""Main training script for RL Combinatorial Optimization."""

import argparse
import os
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.algorithms.agents import create_agent
from src.environments.envs import make_env
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.device import get_device, set_seed
from src.utils.logging import Logger


def load_config(config_path: str) -> dict:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent for combinatorial optimization")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    
    # Set device and seed
    device = get_device(config.get("device", "auto"))
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    
    # Create logger
    logger = Logger(
        log_dir=config["logging"]["log_dir"],
        use_tensorboard=config["logging"]["use_tensorboard"],
        use_wandb=config["logging"]["use_wandb"],
        wandb_project=config["logging"]["wandb_project"],
        wandb_config=config,
    )
    
    try:
        if args.eval_only:
            # Evaluation only mode
            logger.info("Starting evaluation mode")
            
            # Create environment and agent
            env_config = config["env"]
            env = make_env(env_config["name"], **env_config.get(env_config["name"], {}))
            
            agent = create_agent(
                algorithm=config["training"]["algorithm"],
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                **config["training"].get(config["training"]["algorithm"], {}),
                device=device,
            )
            
            # Load checkpoint if provided
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location=device)
                if hasattr(agent, 'policy'):
                    agent.policy.load_state_dict(checkpoint["agent_state_dict"])
                else:
                    agent.ac_net.load_state_dict(checkpoint["agent_state_dict"])
                logger.info(f"Loaded checkpoint from {args.checkpoint}")
            
            # Create evaluator and evaluate
            evaluator = Evaluator(config, agent)
            metrics = evaluator.evaluate(render=args.render)
            
            # Print results
            report = evaluator.generate_report(metrics)
            print(report)
            
            # Save results
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            with open(results_dir / "evaluation_results.txt", "w") as f:
                f.write(report)
            
            logger.info("Evaluation completed")
            
        else:
            # Training mode
            logger.info("Starting training mode")
            
            # Create trainer
            trainer = Trainer(config, logger)
            
            # Train agent
            training_metrics = trainer.train()
            
            # Final evaluation
            logger.info("Running final evaluation")
            evaluator = Evaluator(config, trainer.agent)
            final_metrics = evaluator.evaluate()
            
            # Print final results
            report = evaluator.generate_report(final_metrics)
            print("\n" + "="*50)
            print("FINAL EVALUATION RESULTS")
            print("="*50)
            print(report)
            
            # Save final results
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            with open(results_dir / "final_results.txt", "w") as f:
                f.write(report)
            
            logger.info("Training completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
