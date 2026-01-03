from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Tuple

import pytest

from workflows.multi_episode_workflow import MultiEpisodeWorkflow
from rllm.workflows.workflow import TerminationReason  # type: ignore


class FixedTaskEnv:
    """Env that returns a deterministic task payload and tracks resets."""

    def __init__(self, env_id: str = "game:GuessTheNumber-v0-hard", seed: int | None = None):
        self.env_id = env_id
        self._seed = seed
        self._reset_calls: list[Tuple[int | None, Dict[str, Any]]] = []
        self._task_payload = {"env_id": env_id, "seed": seed, "target": 42}

    def reset(self, seed: int | None = None, task: dict | None = None):
        # Use provided seed when available; otherwise reuse internal seed.
        effective_seed = seed if seed is not None else self._seed
        # Capture the incoming seed/task to ensure consistency.
        self._reset_calls.append((effective_seed, task or {}))
        # If task carries a seed, mirror it into the task payload so downstream logic can verify.
        task_seed = (task or {}).get("seed", effective_seed)
        payload = dict(self._task_payload)
        payload["seed"] = task_seed
        return str(payload), {"task": payload, "seed": task_seed}

    def step(self, action: Any):
        # Single-step terminate to trigger resets quickly.
        return "obs", 0.0, True, {"task": self._task_payload, "terminated": True, "truncated": False, "seed": self._seed}

    @staticmethod
    def from_dict(info: dict) -> "FixedTaskEnv":
        return FixedTaskEnv(env_id=info.get("env_id", "game:GuessTheNumber-v0-hard"), seed=info.get("seed"))

    @staticmethod
    def is_multithread_safe() -> bool:
        return True


class DummyRolloutEngine:
    """Trivial rollout engine that returns a fixed response."""

    async def get_model_response(self, messages, **kwargs):
        class Out:
            text = "guess"
            finish_reason = None
        return Out()

    async def wake_up(self):
        return

    async def sleep(self):
        return


class MinimalAgent:
    """Minimal agent compatible with MultiEpisodeWorkflow for testing resets."""

    def __init__(self):
        self._messages = [{"role": "system", "content": "test"}]
        self._trajectory = type("Traj", (), {"steps": [], "name": "", "info": {}})()

    @property
    def chat_completions(self):
        return list(self._messages)

    @property
    def trajectory(self):
        return self._trajectory

    def reset(self):
        self._messages = [{"role": "system", "content": "test"}]
        self._trajectory = type("Traj", (), {"steps": [], "name": "", "info": {}})()

    def update_from_env(self, observation, reward, done, info, **kwargs):
        self._messages.append({"role": "user", "content": str(observation)})
        if self._trajectory.steps:
            step = self._trajectory.steps[-1]
            step.reward = reward
            step.done = done
            step.info.update(info or {})

    def update_from_model(self, response, **kwargs):
        self._messages.append({"role": "assistant", "content": response})
        step = type("Step", (), {"reward": 0.0, "done": False, "info": {}, "chat_completions": list(self._messages)})()
        self._trajectory.steps.append(step)
        return type("Action", (), {"action": response})

    def get_current_state(self):
        return self._trajectory.steps[-1] if self._trajectory.steps else None


@pytest.mark.asyncio
async def test_env_reset_reuses_same_task_across_episodes():
    """Ensure repeated resets in multi-episode workflow reuse the same task/seed."""
    env = FixedTaskEnv(env_id="game:GuessTheNumber-v0-hard", seed=123)
    workflow = MultiEpisodeWorkflow(
        agent_cls=MinimalAgent,
        env_cls=lambda **_: env,  # inject prebuilt env
        agent_args={},
        env_args={},
        total_step_cap=4,  # force multiple episodes
        min_episodes=2,
        success_reward=1.0,
        rollout_engine=DummyRolloutEngine(),
        executor=ThreadPoolExecutor(max_workers=1),
        timeout=5.0,
    )

    episode = await workflow.run(task={"env_id": env.env_id, "seed": env._seed}, uid="test")
    # Expect at least two resets (initial + after first done)
    assert len(env._reset_calls) >= 2
    # All resets should carry the same seed/task payload
    seeds_seen = {call[0] for call in env._reset_calls}
    tasks_seen = {tuple(sorted((call[1] or {}).items())) for call in env._reset_calls}
    assert seeds_seen == {123}
    assert tasks_seen == {tuple(sorted({"env_id": env.env_id, "seed": 123}.items()))}
    # Episode termination should be MAX_TURNS_EXCEEDED or ENV_DONE, not due to task change.
    assert episode.termination_reason in {TerminationReason.MAX_TURNS_EXCEEDED, TerminationReason.ENV_DONE}

