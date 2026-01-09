"""Train GEMTextAgent on multi-episode GEM environment with custom metrics.

This script uses MultiEpisodeEnv wrapper with a custom execution engine that
extracts multi-episode metrics. The multi-episode logic is encapsulated in the
environment, allowing simpler integration with the standard training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.gem_text_agent import GEMTextAgent  # noqa: E402
from data.prepare_gem_data import prepare_gem_data  # noqa: E402
from envs.multi_episode_env import MultiEpisodeEnv  # noqa: E402
from rllm.data import DatasetRegistry  # type: ignore  # noqa: E402
from trainers.train_multi_episode import run_ppo_agent  # noqa: E402


def _default_multi_episode_prompt() -> str:
    """System prompt that reminds the policy about multi-episode control."""
    return (
        "You are solving the same task across multiple episodes with a fixed total step budget. "
        "Each episode resets the environment but keeps the task identical. "
        "Leverage information gathered from earlier episodes to succeed faster. "
        "Respond with actions inside \\boxed{} each turn."
    )


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(cfg) -> None:  # type: ignore
    """Entry point for multi-episode GEM training using environment wrapper.

    This uses a custom MultiEpisodeAgentPPOTrainer that extracts multi-episode
    metrics from environments with get_metrics() method.
    """
    # Extract environment configuration
    env_args = OmegaConf.to_container(cfg.rllm.env.env_args, resolve=True)  # type: ignore

    # Extract inner env configuration
    inner_env_kwargs = env_args.get("inner_env_kwargs", {})
    env_id = inner_env_kwargs.get("env_id", "game:GuessTheNumber-v0-hard")

    # Register and load GEM task datasets (train/val)
    train_dataset, val_dataset = prepare_gem_data(env_id=env_id)

    # Set dataset paths in config
    if train_dataset is not None and hasattr(cfg, "data"):
        cfg.data.train_files = train_dataset.get_verl_data_path()
    if val_dataset is not None and hasattr(cfg, "data"):
        cfg.data.val_files = val_dataset.get_verl_data_path()

    # Configure agent with multi-episode-aware system prompt
    agent_args = OmegaConf.to_container(
        cfg.rllm.agent.get("agent_args", {}), resolve=True
    )  # type: ignore
    agent_args = dict(agent_args or {})
    agent_args.setdefault("system_prompt", _default_multi_episode_prompt())

    # Use our custom training function that uses MultiEpisodeAgentPPOTrainer
    run_ppo_agent(
        cfg,
        env_class=MultiEpisodeEnv,
        agent_class=GEMTextAgent,
        agent_args=agent_args,
    )


if __name__ == "__main__":
    main()

