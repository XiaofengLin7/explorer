"""Tests for the GEM environment adapter located outside vendor submodules."""

import pytest

from envs.gem_env_adapter import GEMEnvAdapter
from rllm.agents.agent import Action  # type: ignore


@pytest.fixture()
def gem_env() -> GEMEnvAdapter:
    """Provide a deterministic GEM environment instance for tests."""
    return GEMEnvAdapter(env_id="game:GuessTheNumber-v0-easy", env_kwargs={"max_turns": 1})


def test_reset_returns_observation_and_info(gem_env: GEMEnvAdapter) -> None:
    """Adapter.reset should forward observation and info."""
    observation, info = gem_env.reset()

    assert observation
    assert isinstance(info, dict)


def test_step_maps_done_signal(gem_env: GEMEnvAdapter) -> None:
    """`done` should be True when either terminated or truncated is True."""
    gem_env.reset()
    observation, reward, done, info = gem_env.step("not a boxed number")

    assert isinstance(observation, str)
    assert isinstance(reward, float)
    assert done is True
    assert isinstance(info, dict)


def test_accepts_rllm_action_wrapper(gem_env: GEMEnvAdapter) -> None:
    """Adapter should unwrap rLLM Action payloads."""
    gem_env.reset()
    action = Action(action="\\boxed{1}")
    _, _, done, _ = gem_env.step(action)

    assert isinstance(done, bool)


def test_factory_from_dict() -> None:
    """Factory helper should construct the environment."""
    env = GEMEnvAdapter.from_dict({"env_id": "game:GuessTheNumber-v0-easy", "env_kwargs": {"max_turns": 1}})
    obs, _ = env.reset()

    assert obs

