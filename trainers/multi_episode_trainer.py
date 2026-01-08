"""Custom trainer that extracts multi-episode metrics.

This trainer extends AgentPPOTrainer to extract additional metrics from
environments that have a get_metrics() method, such as MultiEpisodeEnv.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


class MultiEpisodeAsyncAgentExecutionEngine(AsyncAgentExecutionEngine):
    """Execution engine that extracts metrics from environments with get_metrics()."""

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run trajectory and include environment metrics if available."""
        # Store reference to env before it might be modified
        env = self.envs[idx]

        # Call parent method
        result = await super().run_agent_trajectory_async(
            idx, application_id, seed=seed, mode=mode, **kwargs
        )

        # If Token mode, extract additional metrics from environment
        if mode == "Token" and isinstance(result, dict) and "metrics" in result:
            if hasattr(env, "get_metrics") and callable(env.get_metrics):
                try:
                    env_metrics = env.get_metrics()
                    if isinstance(env_metrics, dict):
                        # Flatten nested keys and add to metrics
                        for key, value in env_metrics.items():
                            flat_key = key.replace("/", "_")
                            result["metrics"][flat_key] = value
                except Exception:
                    pass  # Silently ignore if metrics extraction fails

        return result


class MultiEpisodeAgentPPOTrainer(AgentPPOTrainer):
    """Trainer that uses MultiEpisodeAsyncAgentExecutionEngine for metrics extraction."""

    def init_workers(self):
        """Initialize workers with custom execution engine."""
        # Call grandparent's init_workers (skip AgentPPOTrainer's)
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        RayPPOTrainer.init_workers(self)

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = (
            engine_args.pop("n_parallel_agents", None)
            or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        )
        print(f"n_parallel_agents: {n_parallel_agents}")

        # Use our custom execution engine instead
        self.agent_execution_engine = MultiEpisodeAsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

