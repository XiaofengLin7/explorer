"""Train GEMTextAgent in a multi-episode workflow using SDK to prevent retokenization.

This script uses the SDK pattern (AgentTrainer with agent_run_func) instead of
the workflow pattern. The SDK captures token IDs directly from vLLM through
the LiteLLM proxy, avoiding retokenization mismatch during training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.prepare_gem_data import prepare_gem_data  # noqa: E402
from envs.gem_env_adapter import GEMEnvAdapter  # noqa: E402
from rllm.data import DatasetRegistry  # type: ignore  # noqa: E402
from rllm.trainer.agent_trainer import AgentTrainer  # type: ignore  # noqa: E402
from workflows.multi_episode_sdk import create_multi_episode_rollout_func  # noqa: E402


def _default_multi_episode_prompt() -> str:
    """System prompt that reminds the policy about multi-episode control."""
    return (
        "You are solving the same task across multiple episodes with a fixed total step budget. "
        "Each episode resets the environment but keeps the task identical. "
        "Leverage information gathered from earlier episodes to succeed faster. "
        "Respond with actions inside \\boxed{} each turn."
    )


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(cfg):  # type: ignore
    """Entry point for multi-episode GEM training using SDK pattern."""
    # Prepare dataset
    env_id = cfg.rllm.env.env_args.get("env_id", "game:GuessTheNumber-v0-hard")
    prepare_gem_data(env_id=env_id)
    train_dataset = DatasetRegistry.load_dataset("gem_tasks", "train")
    val_dataset = DatasetRegistry.load_dataset("gem_tasks", "test")

    # Extract environment configuration
    env_args = OmegaConf.to_container(cfg.rllm.env.env_args, resolve=True)  # type: ignore
    agent_args = OmegaConf.to_container(cfg.rllm.agent.get("agent_args", {}), resolve=True)  # type: ignore

    # Provide multi-episode-aware system prompt by default
    agent_args = dict(agent_args or {})
    system_prompt = agent_args.get("system_prompt", _default_multi_episode_prompt())

    # Get step cap from config
    env_kwargs = env_args.get("env_kwargs", {})
    total_step_cap = env_kwargs.get("total_step_cap") or env_args.get("total_step_cap")

    # Get model name from SDK proxy config, falling back to actor model path
    # This must match the model name configured in the LiteLLM proxy
    sdk_cfg = cfg.rllm.get("sdk", {})
    proxy_cfg = sdk_cfg.get("proxy", {})
    model_name = proxy_cfg.get("model_name") or cfg.actor_rollout_ref.model.path

    # Get rollout temperature from config
    rollout_cfg = cfg.actor_rollout_ref.rollout
    temperature = rollout_cfg.get("temperature", 0.6)
    max_response_length = cfg.data.get("max_response_length", 8192)

    # Create the configured rollout function
    rollout_func = create_multi_episode_rollout_func(
        env_cls=GEMEnvAdapter,
        env_args=env_args,
        system_prompt=system_prompt,
        total_step_cap=total_step_cap,
        min_episodes=3,
        success_reward=1.0,
        episode_header="New episode starts; reuse prior knowledge.",
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_response_length,
    )

    # Use AgentTrainer with agent_run_func (SDK pattern)
    # This uses AgentSdkEngine instead of AgentWorkflowEngine
    trainer = AgentTrainer(
        agent_run_func=rollout_func,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()

