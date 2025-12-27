"""GEM environment adapter for rLLM placed outside vendor submodules.

This adapter keeps third-party code untouched while exposing GEM's Gym-style
interface through the rLLM `BaseEnv` contract used by the training stack.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore

try:
    import gem  # type: ignore
    import gem.envs  # type: ignore  # noqa: F401  # populate registries
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'gem' package is required for GEMEnvAdapter. "
        "Install it via `pip install gem-llm` or ensure it is on PYTHONPATH."
    ) from exc


class GEMEnvAdapter(BaseEnv):
    """Adapter that wraps a GEM environment and normalizes the rLLM API.

    The adapter only touches the `done` signal (terminated or truncated) and
    otherwise passes observations, rewards, and info through unchanged.
    """

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the GEM environment adapter.

        Args:
            env_id: Registered GEM environment identifier.
            env_kwargs: Optional keyword arguments forwarded to ``gem.make``.
        """
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        self._env = gem.make(env_id, **self.env_kwargs)

    def reset(self) -> tuple[Any, dict]:
        """Reset the underlying GEM environment.

        Returns:
            observation: The initial observation returned by GEM.
            info: Auxiliary information from GEM.
        """
        observation, info = self._env.reset()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute one step in the GEM environment.

        Args:
            action: Action to execute. If an rLLM ``Action`` is provided, its
                ``action`` payload is forwarded to GEM.

        Returns:
            observation: Observation after the step.
            reward: Reward returned by GEM, cast to float.
            done: ``True`` if terminated or truncated.
            info: Diagnostic information from GEM.
        """
        raw_action = action.action if isinstance(action, Action) else action
        observation, reward, terminated, truncated, info = self._env.step(raw_action)
        done = bool(terminated or truncated)
        return observation, float(reward), done, info

    def close(self) -> None:
        """Close the underlying GEM environment."""
        if hasattr(self._env, "close"):
            self._env.close()

    @staticmethod
    def from_dict(info: dict) -> "GEMEnvAdapter":
        """Factory helper to create the adapter from a dictionary.

        Args:
            info: Configuration dictionary containing ``env_id`` and optional
                ``env_kwargs``.

        Returns:
            An initialized ``GEMEnvAdapter`` instance.
        """
        env_id = info["env_id"]
        env_kwargs = info.get("env_kwargs", {})
        return GEMEnvAdapter(env_id=env_id, env_kwargs=env_kwargs)

