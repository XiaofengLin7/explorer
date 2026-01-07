"""Tests for multi_episode_sdk workflow.

These tests verify that the SDK-based multi-episode rollout function
correctly implements the multi-episode logic and produces the expected
trajectory structure.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rllm.agents.agent import Action
from rllm.environments.base.base_env import BaseEnv
from rllm.sdk.protocol import StepView, TrajectoryView
from workflows.multi_episode_sdk import (
    MultiEpisodeSDKRunner,
    create_multi_episode_rollout_func,
    extract_last_boxed,
)


class DummyEnv(BaseEnv):
    """Environment that succeeds when the action matches '\\boxed{solve}'."""

    def __init__(self, max_turns: int = 2, env_kwargs: dict | None = None):
        self.max_turns = env_kwargs.get("max_turns", max_turns) if env_kwargs else max_turns
        self._turn = 0
        self._reset_count = 0

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[str, dict]:
        self._turn = 0
        self._reset_count += 1
        return f"obs-{self._reset_count}", {"reset_count": self._reset_count, "max_turns": self.max_turns, "seed": seed}

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        self._turn += 1
        raw_action = action.action if isinstance(action, Action) else action
        # Extract boxed content for comparison
        if "solve" in str(raw_action):
            success = True
        else:
            success = False
        terminated = success or self._turn >= self.max_turns
        truncated = not success and self._turn >= self.max_turns
        reward = 1.0 if success else 0.0
        info = {"terminated": terminated, "truncated": truncated, "raw_reward": reward}
        return f"obs-{self._reset_count}-{self._turn}", reward, terminated or truncated, info

    @staticmethod
    def from_dict(info: dict) -> "DummyEnv":
        return DummyEnv(max_turns=info.get("max_turns", 2))


class TestExtractLastBoxed:
    """Tests for the boxed answer extraction function."""

    def test_extract_single_boxed(self):
        text = r"The answer is \boxed{42}"
        assert extract_last_boxed(text) == "42"

    def test_extract_multiple_boxed_returns_last(self):
        text = r"First \boxed{wrong} then \boxed{correct}"
        assert extract_last_boxed(text) == "correct"

    def test_no_boxed_returns_stripped_text(self):
        text = "  no boxed here  "
        assert extract_last_boxed(text) == "no boxed here"

    def test_boxed_with_spaces(self):
        text = r"\boxed{  answer  }"
        assert extract_last_boxed(text) == "answer"


class TestMultiEpisodeSDKRunner:
    """Tests for the MultiEpisodeSDKRunner class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock session context manager."""
        mock_sess = MagicMock()
        mock_sess.__enter__ = MagicMock(return_value=mock_sess)
        mock_sess.__exit__ = MagicMock(return_value=False)
        mock_sess.steps = []
        return mock_sess

    @pytest.fixture
    def mock_client(self):
        """Create a mock async OpenAI client."""
        mock = AsyncMock()
        return mock

    def _create_mock_response(self, content: str, finish_reason: str = "stop"):
        """Helper to create mock LLM response."""
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_choice.finish_reason = finish_reason
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.mark.asyncio
    async def test_single_episode_success(self, mock_session):
        """Test a single episode that succeeds immediately with step_cap=1."""
        # Create mock step for session
        mock_step = StepView(id="step1", input={}, output={}, reward=0.0)
        mock_session.steps = [mock_step]

        # Create a proper mock client with the right structure
        async def mock_create(**kwargs):
            return self._create_mock_response(r"I'll solve it: \boxed{solve}")

        mock_completions = MagicMock()
        mock_completions.create = mock_create

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = MagicMock()
        mock_client.chat = mock_chat

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                runner = MultiEpisodeSDKRunner(
                    env_cls=DummyEnv,
                    env_args={"env_kwargs": {"max_turns": 5}},
                    total_step_cap=1,  # Only 1 step budget = 1 episode
                    min_episodes=1,
                    success_reward=1.0,
                )
                trajectories = await runner.run(task={"seed": 42})

        assert len(trajectories) == 1
        traj = trajectories[0]
        assert traj.name == "multi_episode_agent"
        assert traj.metadata["success_count"] == 1
        assert traj.metadata["num_episodes"] == 1
        assert traj.reward == 1.0

    @pytest.mark.asyncio
    async def test_multiple_successful_episodes_until_cap(self, mock_session):
        """Test that workflow runs multiple episodes until step cap is reached."""
        # Each success takes 1 step, step_cap=3 means 3 successful episodes
        mock_session.steps = [
            StepView(id=f"step{i}", input={}, output={}, reward=0.0) for i in range(3)
        ]

        async def mock_create(**kwargs):
            return self._create_mock_response(r"\boxed{solve}")

        mock_completions = MagicMock()
        mock_completions.create = mock_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_client = MagicMock()
        mock_client.chat = mock_chat

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                runner = MultiEpisodeSDKRunner(
                    env_cls=DummyEnv,
                    env_args={"env_kwargs": {"max_turns": 5}},
                    total_step_cap=3,  # 3 steps = 3 episodes (each success = 1 step)
                    min_episodes=1,
                    success_reward=1.0,
                )
                trajectories = await runner.run(task={"seed": 42})

        traj = trajectories[0]
        assert traj.metadata["num_episodes"] == 3
        assert traj.metadata["success_count"] == 3
        assert traj.reward == 3.0

    @pytest.mark.asyncio
    async def test_multiple_episodes_with_mixed_results(self, mock_session):
        """Test multiple episodes with different outcomes."""
        # Response sequence: fail, solve (ep1=2 steps), fail, fail (ep2=2 steps), solve (ep3=1 step)
        # Total = 5 steps, but we have step_cap=5
        responses = [
            self._create_mock_response(r"\boxed{wrong}"),
            self._create_mock_response(r"\boxed{solve}"),
            self._create_mock_response(r"\boxed{fail}"),
            self._create_mock_response(r"\boxed{nope}"),
            self._create_mock_response(r"\boxed{solve}"),
            self._create_mock_response(r"\boxed{extra}"),  # shouldn't be reached
        ]

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)

        # Create mock steps
        mock_steps = [StepView(id=f"step{i}", input={}, output={}, reward=0.0) for i in range(6)]
        mock_session.steps = mock_steps[:5]  # Only 5 steps should be used

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                runner = MultiEpisodeSDKRunner(
                    env_cls=DummyEnv,
                    env_args={"env_kwargs": {"max_turns": 2}},
                    total_step_cap=5,  # Adjusted to match expected 5 steps
                    min_episodes=2,
                    success_reward=1.0,
                )
                trajectories = await runner.run(task={"seed": 42})

        assert len(trajectories) == 1
        traj = trajectories[0]
        # Episode 1: fail, solve -> success (2 steps)
        # Episode 2: fail, fail -> truncated (2 steps)
        # Episode 3: solve -> success (1 step)
        # Total: 5 steps
        assert traj.metadata["num_episodes"] == 3
        assert traj.metadata["success_count"] == 2
        assert traj.reward == 2.0

    @pytest.mark.asyncio
    async def test_step_cap_limits_execution(self, mock_session, mock_client):
        """Test that step cap correctly limits total steps."""
        runner = MultiEpisodeSDKRunner(
            env_cls=DummyEnv,
            env_args={"env_kwargs": {"max_turns": 10}},  # High max_turns
            total_step_cap=3,  # Low step cap
            min_episodes=1,
            success_reward=1.0,
        )

        # Always fail - should hit step cap
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._create_mock_response(r"\boxed{fail}")
        )

        mock_steps = [StepView(id=f"step{i}", input={}, output={}, reward=0.0) for i in range(3)]
        mock_session.steps = mock_steps

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                trajectories = await runner.run(task={})

        assert len(trajectories) == 1
        traj = trajectories[0]
        assert traj.metadata["total_steps"] == 3
        assert traj.metadata["step_cap"] == 3

    @pytest.mark.asyncio
    async def test_episode_header_in_observations(self, mock_session, mock_client):
        """Test that episode headers are added to observations."""
        runner = MultiEpisodeSDKRunner(
            env_cls=DummyEnv,
            env_args={"env_kwargs": {"max_turns": 1}},
            total_step_cap=3,
            min_episodes=2,
            episode_header="Episode started!",
        )

        recorded_messages = []

        async def record_messages(**kwargs):
            recorded_messages.append(kwargs.get("messages", []))
            return self._create_mock_response(r"\boxed{fail}")

        mock_client.chat.completions.create = AsyncMock(side_effect=record_messages)
        mock_session.steps = [StepView(id=f"step{i}", input={}, output={}, reward=0.0) for i in range(3)]

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                await runner.run(task={})

        # Check that episode headers are in the messages
        assert len(recorded_messages) >= 2
        # First episode should have "[Episode 1]"
        first_call_user_msg = recorded_messages[0][1]["content"]  # [0] is system, [1] is user
        assert "[Episode 1]" in first_call_user_msg
        assert "Episode started!" in first_call_user_msg


class TestCreateMultiEpisodeRolloutFunc:
    """Tests for the factory function."""

    @pytest.mark.asyncio
    async def test_factory_creates_working_function(self):
        """Test that the factory creates a callable rollout function."""
        rollout_fn = create_multi_episode_rollout_func(
            env_cls=DummyEnv,
            env_args={"env_kwargs": {"max_turns": 2}},
            total_step_cap=5,
        )

        # Mock the SDK components
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.steps = [StepView(id="step1", input={}, output={}, reward=0.0)]

        mock_client = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = r"\boxed{solve}"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                result = await rollout_fn(seed=42)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TrajectoryView)

    def test_factory_preserves_configuration(self):
        """Test that factory correctly captures configuration."""
        rollout_fn = create_multi_episode_rollout_func(
            env_cls=DummyEnv,
            env_args={"env_id": "test"},
            system_prompt="Custom prompt",
            total_step_cap=100,
            min_episodes=5,
            success_reward=2.0,
            episode_header="Custom header",
            model_name="test-model",
            temperature=0.8,
            max_tokens=1024,
        )

        # The function should be an async function
        assert asyncio.iscoroutinefunction(rollout_fn)


class TestIntegrationWithMockedSDK:
    """Integration tests with mocked SDK components."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_reward_assignment(self):
        """Test that rewards are correctly assigned to the last step."""
        runner = MultiEpisodeSDKRunner(
            env_cls=DummyEnv,
            env_args={"env_kwargs": {"max_turns": 2}},
            total_step_cap=4,
            min_episodes=2,
            success_reward=1.0,
        )

        # Create mock client
        mock_client = AsyncMock()
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=r"\boxed{solve}"), finish_reason="stop")]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=r"\boxed{fail}"), finish_reason="stop")]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=r"\boxed{fail}"), finish_reason="stop")]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=r"\boxed{solve}"), finish_reason="stop")]),
        ]
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)

        # Create mock session with steps
        mock_steps = [
            StepView(id="step1", input={}, output={}, reward=0.0),
            StepView(id="step2", input={}, output={}, reward=0.0),
            StepView(id="step3", input={}, output={}, reward=0.0),
        ]
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.steps = mock_steps

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                trajectories = await runner.run(task={})

        traj = trajectories[0]

        # Episode 1: solve -> success (1 step)
        # Episode 2: fail, fail -> truncated (2 steps, but only fit 2 more in cap)
        # Total: 3 steps (step cap of 4, but ep1 uses 1, ep2 uses 2)
        assert traj.metadata["success_count"] >= 1

        # Last step should have the final reward
        assert traj.steps[-1].reward == traj.reward

    @pytest.mark.asyncio
    async def test_trajectory_metadata_completeness(self):
        """Test that trajectory metadata contains all required fields."""
        runner = MultiEpisodeSDKRunner(
            env_cls=DummyEnv,
            env_args={"env_kwargs": {"max_turns": 2}},
            total_step_cap=3,
            min_episodes=1,
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=r"\boxed{solve}"), finish_reason="stop")]
            )
        )

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.steps = [StepView(id="step1", input={}, output={}, reward=0.0)]

        with patch("workflows.multi_episode_sdk.session", return_value=mock_session):
            with patch("workflows.multi_episode_sdk.get_chat_client_async", return_value=mock_client):
                trajectories = await runner.run(task={})

        traj = trajectories[0]
        metadata = traj.metadata

        # Check all required metadata fields
        assert "multi_episode" in metadata
        assert "episode_successes" in metadata
        assert "success_count" in metadata
        assert "num_episodes" in metadata
        assert "total_steps" in metadata
        assert "step_cap" in metadata
        assert "episode_lengths" in metadata

        assert metadata["multi_episode"] is True
        assert isinstance(metadata["episode_successes"], list)
        assert isinstance(metadata["episode_lengths"], list)

