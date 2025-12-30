"""Train GEMTextAgent on a single GEM environment via AgentTrainer."""

from __future__ import annotations

from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.gem_text_agent import GEMTextAgent  # noqa: E402
from data.prepare_gem_data import prepare_gem_data  # noqa: E402
from envs.gem_env_adapter import GEMEnvAdapter  # noqa: E402
from rllm.trainer.agent_trainer import AgentTrainer  # type: ignore
from rllm.data import DatasetRegistry  # type: ignore


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(cfg):  # type: ignore
    # Register and load GEM task datasets (train/val).
    env_id = cfg.rllm.env.env_args.get("env_id", "game:GuessTheNumber-v0-hard")
    prepare_gem_data(env_id=env_id)
    train_dataset = DatasetRegistry.load_dataset("gem_tasks", "train")
    val_dataset = DatasetRegistry.load_dataset("gem_tasks", "test")

    trainer = AgentTrainer(
        agent_class=GEMTextAgent,
        env_class=GEMEnvAdapter,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()

