from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, List

import pytest

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.environments.base.base_env import BaseEnv
from workflows.multi_episode_workflow import MultiEpisodeWorkflow


class DummyRolloutEngine(RolloutEngine):
    """Deterministic rollout engine for tests."""

    def __init__(self, responses: Iterable[str]):
        super().__init__()
        self._responses = list(responses)
        self._cursor = 0

    async def get_model_response(self, messages: List[dict], **kwargs) -> ModelOutput:  # type: ignore[override]
        if self._cursor >= len(self._responses):
            raise IndexError("Ran out of scripted responses")
        text = self._responses[self._cursor]
        self._cursor += 1
        return ModelOutput(text=text, finish_reason=None)


class DummyAgent(BaseAgent):
    """Minimal agent that echoes responses as actions."""

    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []
        self._trajectory = Trajectory()
        self.reset()

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return list(self._messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": "test"}]
        self._trajectory = Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs) -> None:
        self._messages.append({"role": "user", "content": str(observation)})
        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = float(reward)
            last_step.done = bool(done)
            last_step.info.update(info or {})

    def update_from_model(self, response: str, **kwargs) -> Action:
        self._messages.append({"role": "assistant", "content": response})
        step = Step(chat_completions=list(self._messages), action=Action(action=response), model_response=response, info={})
        self._trajectory.steps.append(step)
        return Action(action=response)

    def get_current_state(self) -> Step:
        if not self._trajectory.steps:
            return Step(chat_completions=list(self._messages))
        return self._trajectory.steps[-1]


class DummyEnv(BaseEnv):
    """Environment that succeeds when the action matches 'solve'."""

    def __init__(self, max_turns: int = 2):
        self.max_turns = max_turns
        self._turn = 0
        self._reset_count = 0

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[str, dict]:
        self._turn = 0
        self._reset_count += 1
        return f"obs-{self._reset_count}", {"reset_count": self._reset_count, "max_turns": self.max_turns, "seed": seed}

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        self._turn += 1
        raw_action = action.action if isinstance(action, Action) else action
        success = raw_action == "solve"
        terminated = success or self._turn >= self.max_turns
        truncated = not success and self._turn >= self.max_turns
        reward = 1.0 if success else 0.0
        info = {"terminated": terminated, "truncated": truncated, "raw_reward": reward}
        return f"obs-{self._reset_count}-{self._turn}", reward, terminated or truncated, info

    @staticmethod
    def from_dict(info: dict) -> "DummyEnv":
        return DummyEnv(max_turns=info.get("max_turns", 2))


async def _run_workflow(responses: list[str], total_step_cap: int = 5) -> tuple[Trajectory, dict]:
    """Helper to drive the workflow asynchronously."""
    rollout_engine = DummyRolloutEngine(responses=responses)
    executor = ThreadPoolExecutor(max_workers=2)
    workflow = MultiEpisodeWorkflow(
        agent_cls=DummyAgent,
        env_cls=DummyEnv,
        agent_args={},
        env_args={"env_kwargs": {"max_turns": 2}},
        total_step_cap=total_step_cap,
        min_episodes=1,
        success_reward=1.0,
        rollout_engine=rollout_engine,
        executor=executor,
        timeout=10.0,
    )
    try:
        episode = await workflow.run(task={}, uid="test")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    assert episode.trajectories, "Workflow must produce at least one trajectory"
    return episode.trajectories[0], episode.metrics


@pytest.mark.asyncio
async def test_success_count_matches_reward():
    trajectory, metrics = await _run_workflow(responses=["solve", "noop", "solve", "noop", "noop"])
    assert len(trajectory.steps) == 5
    assert trajectory.reward == 2.0
    assert trajectory.info["success_count"] == 2
    assert metrics["episode/success_rate"] == 1.0
    assert metrics["episode_1/success_rate"] == 1.0
    assert metrics["episode_2/success_rate"] == 1.0
    assert metrics["episode_3/success_rate"] == 0.0


@pytest.mark.asyncio
async def test_step_info_has_episode_markers():
    trajectory, _ = await _run_workflow(responses=["noop", "solve", "noop", "noop", "solve"], total_step_cap=4)
    first_step = trajectory.steps[0]
    last_step = trajectory.steps[-1]
    assert first_step.info["episode_index"] == 0
    assert first_step.info["episode_step"] == 0
    assert first_step.info["total_step"] == 1
    assert first_step.info["multi_episode"] is True
    assert last_step.done is True
    assert last_step.info["total_step"] == 4

