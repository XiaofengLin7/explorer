"""Multi-episode workflow that reuses the same task over several episodes.

This workflow keeps the agent and environment outside the vendored rLLM codebase
and drives repeated episodes until a global step budget is exhausted. Rewards
are shaped so the trajectory reward equals the number of successful episodes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.timing_mixin import TimingTrackingMixin
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class MultiEpisodeWorkflow(TimingTrackingMixin, Workflow):
    """Run multiple episodes for a single task under a global step cap."""

    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args: Dict[str, Any] | None = None,
        env_args: Dict[str, Any] | None = None,
        total_step_cap: int | None = None,
        min_episodes: int = 3,
        success_reward: float = 1.0,
        episode_header: str = "New episode begins.",
        **kwargs: Any,
    ):
        """
        Args:
            agent_cls: Agent class or registry key.
            env_cls: Environment class or registry key.
            agent_args: Constructor kwargs forwarded to the agent.
            env_args: Constructor kwargs forwarded to the environment.
            total_step_cap: Maximum steps across all episodes; defaults to
                3 * per-episode max_turns when available.
            min_episodes: Soft target for how many episodes to attempt.
            success_reward: Reward assigned when an episode succeeds.
            episode_header: Text prepended to the observation on each reset.
            **kwargs: Passed to the parent workflow.
        """
        from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING

        super().__init__(**kwargs)

        agent_args = dict(agent_args or {})
        env_args = dict(env_args or {})

        agent_cls = AGENT_CLASS_MAPPING[agent_cls] if isinstance(agent_cls, str) else agent_cls
        env_cls = ENV_CLASS_MAPPING[env_cls] if isinstance(env_cls, str) else env_cls

        self.agent = agent_cls(**agent_args)

        # Allow both wrapped env_kwargs and direct kwargs for flexibility in tests.
        env_init_args = dict(env_args)
        nested_env_kwargs = env_init_args.pop("env_kwargs", None)
        try:
            self.env = env_cls(**env_init_args)
        except TypeError:
            if nested_env_kwargs is not None:
                self.env = env_cls(**nested_env_kwargs)
            else:
                raise

        self.success_reward = float(success_reward)
        self.min_episodes = max(1, int(min_episodes))
        self.episode_header = episode_header
        self._configured_step_cap = total_step_cap
        self._episode_turn_cap = self._infer_episode_turn_cap(env_args)

        self._episode_successes: List[bool] = []
        self._total_steps: int = 0
        self._episode_lengths: List[int] = []

    async def run(self, task: dict, uid: str, **kwargs: Any) -> Episode:
        """Execute the workflow until the global step budget is exhausted."""
        self.reset(task=task, uid=uid)
        self.start_timing()

        step_cap = self._configured_step_cap or self._default_step_cap()

        observation, info = await self.timed_env_call(self._reset_env)
        episode_index = 0
        total_steps = 0
        episode_successes: List[bool] = []
        episode_lengths: List[int] = []

        while total_steps < step_cap:
            episode_step = 0
            observation_with_header = self._format_observation(observation, episode_index)
            priming_state = self.agent.get_current_state()
            priming_reward = priming_state.reward if priming_state is not None else 0.0
            priming_done = priming_state.done if priming_state is not None else False
            pre_step_info = self._augment_info(
                base_info=info,
                episode_index=episode_index,
                episode_step=episode_step,
                total_step=total_steps,
                episode_start=True,
                episode_done=False,
                total_step_cap=step_cap,
            )
            # Preserve the previous step's reward/done when priming a new episode.
            self.agent.update_from_env(observation_with_header, priming_reward, priming_done, pre_step_info)

            while total_steps < step_cap:
                model_output: ModelOutput = await self.timed_llm_call(self.agent.chat_completions, application_id=uid, **kwargs)
                response_text = model_output.text or ""
                if model_output.finish_reason == "length":
                    raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

                action = self.agent.update_from_model(response_text)
                next_observation, env_reward, done, step_info = await self.timed_env_call(self.env.step, action)

                total_steps += 1
                success = self._is_episode_success(done=done, info=step_info, reward=env_reward)
                episode_done = bool(done or total_steps >= step_cap)
                shaped_reward = self.success_reward if success else 0.0

                merged_info = self._augment_info(
                    base_info=step_info,
                    episode_index=episode_index,
                    episode_step=episode_step,
                    total_step=total_steps,
                    episode_start=False,
                    episode_done=done,
                    success=success,
                    raw_reward=env_reward,
                    total_step_cap=step_cap,
                )

                self.agent.update_from_env(next_observation, shaped_reward, episode_done, merged_info)
                current_step = self.agent.get_current_state()
                if current_step is not None:
                    current_step.reward = shaped_reward
                    current_step.done = episode_done
                    current_step.info.update(merged_info)

                episode_step += 1
                if done or total_steps >= step_cap:
                    episode_successes.append(success)
                    episode_lengths.append(episode_step)
                    break

            if total_steps >= step_cap:
                break

            # Start the next episode with the same task/seed.
            observation, info = await self.timed_env_call(self._reset_env)
            episode_index += 1

            # Soft target to attempt at least min_episodes; continue while steps allow.
            if episode_index + 1 >= self.min_episodes and total_steps >= step_cap:
                break

        self._episode_successes = episode_successes
        self._episode_lengths = episode_lengths
        self._total_steps = total_steps

        trajectory = self.agent.trajectory
        trajectory.name = trajectory.name or "multi_episode_agent"
        trajectory.info.update(
            {
                "multi_episode": True,
                "episode_successes": episode_successes,
                "success_count": sum(episode_successes),
                "num_episodes": len(episode_successes),
                "total_steps": total_steps,
                "step_cap": step_cap,
                "episode_lengths": episode_lengths,
            }
        )

        # Optional: log chat completions similar to the non-workflow path.
        self._log_chat_completions(uid, trajectory)

        termination_reason = TerminationReason.MAX_TURNS_EXCEEDED if total_steps >= step_cap else TerminationReason.ENV_DONE
        return self.postprocess_episode(self.collect_trajectories(), termination_reason=termination_reason)

    def collect_metrics(self, episode: Episode):
        """Attach episode- and per-episode success metrics."""
        super().collect_metrics(episode)
        successes = self._episode_successes
        any_success = 1.0 if any(successes) else 0.0
        metrics = {
            "episode/success_rate": any_success,
            "episode/num_episodes": len(successes),
            "episode/success_count": sum(successes),
            "episode/total_steps": self._total_steps,
        }
        for idx, flag in enumerate(successes[:3], start=1):
            metrics[f"episode_{idx}/success_rate"] = 1.0 if flag else 0.0
        for idx, length in enumerate(self._episode_lengths[:3], start=1):
            metrics[f"episode_{idx}/steps"] = length
        episode.metrics.update(metrics)

    def assign_episode_correctness(self, episode: Episode):
        """Mark an episode correct if any sub-episode succeeds."""
        episode.is_correct = sum(self._episode_successes) > 0

    def _default_step_cap(self) -> int:
        """Fallback global step budget."""
        if self._configured_step_cap is not None:
            return self._configured_step_cap
        if self._episode_turn_cap is not None:
            return max(self._episode_turn_cap * 3, self._episode_turn_cap)
        return 50

    def _infer_episode_turn_cap(self, env_args: Dict[str, Any]) -> int | None:
        """Infer a per-episode turn cap from env kwargs when available."""
        env_kwargs = env_args.get("env_kwargs", {})
        turn_cap = env_kwargs.get("max_turns")
        if isinstance(turn_cap, int) and turn_cap > 0:
            return turn_cap
        return None

    def _reset_env(self) -> tuple[Any, dict]:
        """Reset the environment with a fixed seed when available."""
        try:
            return self.env.reset()
        except TypeError:
            # Backwards compatibility with reset signatures that accept task only.
            return self.env.reset(task=self.task)

    def _format_observation(self, observation: Any, episode_index: int) -> Any:
        """Prepend an episode marker to the observation for policy awareness."""
        header = f"[Episode {episode_index + 1}] {self.episode_header}".strip()
        return f"{header}\n{observation}"

    @staticmethod
    def _is_episode_success(done: bool, info: Dict[str, Any], reward: float) -> bool:
        """Determine whether the just-finished episode succeeded."""
        if not done:
            return False
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if truncated:
            return False
        return terminated and reward > 0

    @staticmethod
    def _augment_info(
        base_info: dict | None,
        episode_index: int,
        episode_step: int,
        total_step: int,
        episode_start: bool,
        episode_done: bool,
        total_step_cap: int,
        success: bool | None = None,
        raw_reward: float | None = None,
    ) -> dict:
        """Inject multi-episode metadata into the info dict."""
        info = dict(base_info or {})
        info.update(
            {
                "episode_index": episode_index,
                "episode_step": episode_step,
                "total_step": total_step,
                "episode_start": episode_start,
                "episode_done": episode_done,
                "multi_episode": True,
                "step_cap": total_step_cap,
            }
        )
        if success is not None:
            info["episode_success"] = success
        if raw_reward is not None:
            info["raw_reward"] = raw_reward
        return info

    def _log_chat_completions(self, uid: str, trajectory) -> None:
        """Persist chat completions, mirroring single-agent logging."""
        if not trajectory.steps:
            return
        chat_completions = trajectory.steps[-1].chat_completions
        if not chat_completions:
            return

        # Prefer trainer.default_local_dir when provided via AgentWorkflowEngine.
        base_dir = getattr(self, "default_local_dir", None) or os.getcwd()
        out_dir = Path(base_dir) / "chat_completions"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use current training step if engine provided it; fall back to uid.
        step_getter = getattr(self, "training_step_getter", None)
        step = step_getter() if callable(step_getter) else None
        filename = f"{step}.jsonl" if step is not None else f"{uid}.jsonl"

        out_file = out_dir / filename
        # Append so multiple trajectories in the same step do not overwrite.
        with out_file.open("a") as f:
            f.write(json.dumps(chat_completions) + "\n")

