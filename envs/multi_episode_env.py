"""Multi-episode environment wrapper for rLLM.

This wrapper encapsulates multi-episode logic, allowing an agent to interact
with the same task across multiple episodes within a single trajectory. The
inner environment is reset when an episode ends, but the outer trajectory
continues until the total step cap is reached.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Type

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore


class MultiEpisodeEnv(BaseEnv):
    """Wrapper environment that runs multiple episodes of an inner environment.

    This environment wraps another BaseEnv and implements multi-episode logic:
    - When the inner environment returns done=True, it resets internally
    - The outer trajectory only ends when total_steps reaches total_step_cap
    - Rewards are shaped: success_reward on episode success, 0.0 otherwise

    Attributes:
        inner_env: The wrapped environment instance.
        total_step_cap: Maximum steps across all episodes.
        success_reward: Reward given when an episode succeeds.
        episode_header: Text prepended to observations on episode start.
    """

    def __init__(
        self,
        inner_env_class: Type[BaseEnv] | str,
        inner_env_kwargs: Optional[Dict[str, Any]] = None,
        total_step_cap: int = 30,
        success_reward: float = 1.0,
        episode_header: str = "New episode begins.",
        **kwargs: Any,
    ):
        """Initialize the multi-episode environment wrapper.

        Args:
            inner_env_class: The inner environment class or its fully qualified
                string name (e.g., "envs.gem_env_adapter.GEMEnvAdapter").
            inner_env_kwargs: Keyword arguments to pass to the inner environment
                constructor.
            total_step_cap: Maximum number of steps across all episodes.
            success_reward: Reward assigned when an episode succeeds.
            episode_header: Text prepended to observations at episode start.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        super().__init__()

        # Resolve inner environment class from string if needed
        if isinstance(inner_env_class, str):
            inner_env_class = self._resolve_class(inner_env_class)

        self.inner_env_class = inner_env_class
        self.inner_env_kwargs = dict(inner_env_kwargs or {})
        self.total_step_cap = max(1, int(total_step_cap))
        self.success_reward = float(success_reward)
        self.episode_header = episode_header

        # Create the inner environment
        self.inner_env: BaseEnv = self.inner_env_class(**self.inner_env_kwargs)

        # Episode tracking state (initialized in reset)
        self._total_steps: int = 0
        self._episode_index: int = 0
        self._episode_step: int = 0
        self._episode_successes: List[bool] = []
        self._episode_lengths: List[int] = []
        self._task: Optional[dict] = None
        self._seed: Optional[int] = None

    @staticmethod
    def _resolve_class(class_path: str) -> Type[BaseEnv]:
        """Resolve a class from its fully qualified string path.

        Args:
            class_path: Fully qualified class name (e.g., "envs.gem_env_adapter.GEMEnvAdapter").

        Returns:
            The resolved class.

        Raises:
            ImportError: If the module or class cannot be found.
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def reset(
        self, seed: Optional[int] = None, task: Optional[dict] = None
    ) -> tuple[Any, dict]:
        """Reset the multi-episode environment.

        Initializes tracking state and resets the inner environment.

        Args:
            seed: Optional random seed for reproducibility.
            task: Optional task dictionary to pass to inner environment.

        Returns:
            observation: Initial observation with episode header.
            info: Initial info dictionary with multi-episode metadata.
        """
        self._total_steps = 0
        self._episode_index = 0
        self._episode_step = 0
        self._episode_successes = []
        self._episode_lengths = []
        self._task = task
        self._seed = seed if seed is not None else self._seed

        # Reset inner environment
        observation, info = self._reset_inner_env()

        # Format observation with episode header
        observation = self._format_observation(observation, self._episode_index)

        # Augment info with multi-episode metadata
        augmented_info = self._augment_info(
            base_info=info,
            episode_index=self._episode_index,
            episode_step=self._episode_step,
            total_step=self._total_steps,
            episode_start=True,
            episode_done=False,
        )

        return observation, augmented_info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute one step in the multi-episode environment.

        If the inner environment returns done=True, it is reset and the
        trajectory continues. The outer trajectory only ends when total_steps
        reaches total_step_cap.

        Args:
            action: Action to execute in the inner environment.

        Returns:
            observation: Next observation (with episode header on new episode).
            reward: Shaped reward (success_reward on episode success, else 0.0).
            done: True only when total_steps >= total_step_cap.
            info: Augmented info dictionary with multi-episode metadata.
        """
        # Unwrap Action if needed
        raw_action = action.action if isinstance(action, Action) else action

        # Execute step in inner environment
        observation, env_reward, inner_done, info = self.inner_env.step(raw_action)

        self._total_steps += 1
        self._episode_step += 1

        # Determine if this episode succeeded
        success = self._is_episode_success(
            done=inner_done, info=info, reward=env_reward
        )

        # Shape reward: success_reward on episode success, 0.0 otherwise
        shaped_reward = self.success_reward if success else 0.0

        # Determine if outer trajectory should end
        outer_done = self._total_steps >= self.total_step_cap

        # Handle episode completion
        if inner_done:
            self._episode_successes.append(success)
            self._episode_lengths.append(self._episode_step)

            if not outer_done:
                # Reset for next episode
                observation, reset_info = self._reset_inner_env()
                self._episode_index += 1
                self._episode_step = 0
                observation = self._format_observation(
                    observation, self._episode_index
                )
                # Merge reset info into info
                info = {**info, **reset_info}

        # Augment info with multi-episode metadata
        augmented_info = self._augment_info(
            base_info=info,
            episode_index=self._episode_index,
            episode_step=self._episode_step,
            total_step=self._total_steps,
            episode_start=(self._episode_step == 0),
            episode_done=inner_done,
            success=success if inner_done else None,
            raw_reward=env_reward,
            trajectory_done=outer_done,
        )

        return observation, shaped_reward, outer_done, augmented_info

    def close(self) -> None:
        """Close the inner environment."""
        if hasattr(self.inner_env, "close"):
            self.inner_env.close()

    def compute_final_reward(self) -> float:
        """Compute the final trajectory reward.

        Returns:
            The sum of success rewards (number of successful episodes * success_reward).
        """
        return sum(self._episode_successes) * self.success_reward

    def get_metrics(self) -> Dict[str, Any]:
        """Collect metrics matching MultiEpisodeWorkflow format.

        Returns a dictionary with the same metric keys as MultiEpisodeWorkflow's
        collect_metrics method for consistent logging across both approaches.

        Note: Always returns metrics for 3 episodes with -1 for episodes that
        didn't occur. The -1 values are filtered out during aggregation (the
        trainer filters values < 0).

        Returns:
            Dictionary containing:
                - episode/success_rate: 1.0 if any episode succeeded, else 0.0
                - episode/num_episodes: Total number of episodes attempted
                - episode/success_count: Number of successful episodes
                - episode/total_steps: Total steps across all episodes
                - episode_N/success_rate: Success rate for episode N (first 3)
                - episode_N/steps: Steps taken in episode N (first 3)
        """
        successes = self._episode_successes
        any_success = 1.0 if any(successes) else 0.0

        metrics: Dict[str, Any] = {
            "episode/success_rate": any_success,
            "episode/num_episodes": len(successes),
            "episode/success_count": sum(successes),
            "episode/total_steps": self._total_steps,
        }

        # Per-episode metrics for first 3 episodes
        # Always include all 3 episodes with -1 for missing ones (filtered during aggregation)
        for idx in range(1, 4):
            if idx <= len(successes):
                metrics[f"episode_{idx}/success_rate"] = 1.0 if successes[idx - 1] else 0.0
            else:
                metrics[f"episode_{idx}/success_rate"] = -1.0  # Will be filtered

        for idx in range(1, 4):
            if idx <= len(self._episode_lengths):
                metrics[f"episode_{idx}/steps"] = self._episode_lengths[idx - 1]
            else:
                metrics[f"episode_{idx}/steps"] = -1  # Will be filtered

        return metrics

    def get_trajectory_info(self) -> Dict[str, Any]:
        """Get trajectory-level info matching MultiEpisodeWorkflow format.

        Returns a dictionary with the same keys as trajectory.info.update()
        in MultiEpisodeWorkflow for consistent metadata across both approaches.

        Returns:
            Dictionary containing:
                - multi_episode: Always True
                - episode_successes: List of booleans for each episode
                - success_count: Number of successful episodes
                - num_episodes: Total episodes attempted
                - total_steps: Total steps across all episodes
                - step_cap: The configured step cap
                - episode_lengths: List of step counts per episode
        """
        return {
            "multi_episode": True,
            "episode_successes": list(self._episode_successes),
            "success_count": sum(self._episode_successes),
            "num_episodes": len(self._episode_successes),
            "total_steps": self._total_steps,
            "step_cap": self.total_step_cap,
            "episode_lengths": list(self._episode_lengths),
        }

    @property
    def is_correct(self) -> bool:
        """Whether the trajectory is considered correct.

        A trajectory is correct if at least one episode succeeded,
        matching the logic in MultiEpisodeWorkflow.assign_episode_correctness.

        Returns:
            True if any episode succeeded, False otherwise.
        """
        return sum(self._episode_successes) > 0

    def _reset_inner_env(self) -> tuple[Any, dict]:
        """Reset the inner environment with the stored seed and task.

        Returns:
            observation: Initial observation from inner environment.
            info: Info dictionary from inner environment.
        """
        try:
            return self.inner_env.reset(seed=self._seed, task=self._task)
        except TypeError:
            # Fallback for envs without task/seed args
            try:
                return self.inner_env.reset(seed=self._seed)
            except TypeError:
                try:
                    return self.inner_env.reset(task=self._task)
                except TypeError:
                    return self.inner_env.reset()

    def _format_observation(self, observation: Any, episode_index: int) -> Any:
        """Prepend episode marker to observation.

        Args:
            observation: Raw observation from inner environment.
            episode_index: Current episode index (0-based).

        Returns:
            Observation with episode header prepended.
        """
        header = f"[Episode {episode_index + 1}] {self.episode_header}".strip()
        if isinstance(observation, str):
            return f"{header}\n{observation}"
        return observation

    @staticmethod
    def _is_episode_success(done: bool, info: Dict[str, Any], reward: float) -> bool:
        """Determine whether the just-finished episode succeeded.

        An episode is considered successful if it terminated (not truncated)
        with a positive reward.

        Args:
            done: Whether the episode ended.
            info: Info dictionary from the environment.
            reward: Reward from the final step.

        Returns:
            True if the episode succeeded, False otherwise.
        """
        if not done:
            return False
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if truncated:
            return False
        return terminated and reward > 0

    def _augment_info(
        self,
        base_info: Optional[dict],
        episode_index: int,
        episode_step: int,
        total_step: int,
        episode_start: bool,
        episode_done: bool,
        success: Optional[bool] = None,
        raw_reward: Optional[float] = None,
        trajectory_done: bool = False,
    ) -> dict:
        """Inject multi-episode metadata into the info dict.

        Args:
            base_info: Original info from inner environment.
            episode_index: Current episode index (0-based).
            episode_step: Step within current episode.
            total_step: Total steps across all episodes.
            episode_start: Whether this is the start of an episode.
            episode_done: Whether the current episode just ended.
            success: Whether the episode succeeded (if episode_done).
            raw_reward: Raw reward from inner environment.
            trajectory_done: Whether the outer trajectory is complete.

        Returns:
            Augmented info dictionary.
        """
        info = dict(base_info or {})
        info.update(
            {
                "episode_index": episode_index,
                "episode_step": episode_step,
                "total_step": total_step,
                "episode_start": episode_start,
                "episode_done": episode_done,
                "multi_episode": True,
                "step_cap": self.total_step_cap,
                "episode_successes": list(self._episode_successes),
                "episode_lengths": list(self._episode_lengths),
            }
        )
        if success is not None:
            info["episode_success"] = success
        if raw_reward is not None:
            info["raw_reward"] = raw_reward

        # Include full metrics and trajectory info when trajectory completes
        if trajectory_done:
            info["metrics"] = self.get_metrics()
            info["trajectory_info"] = self.get_trajectory_info()
            info["is_correct"] = self.is_correct

        return info

    @staticmethod
    def from_dict(info: dict) -> "MultiEpisodeEnv":
        """Factory method to create MultiEpisodeEnv from a dictionary.

        Expected keys:
            - inner_env_class: Class or string path of inner environment.
            - inner_env_kwargs: Dict of kwargs for inner environment.
            - total_step_cap: Maximum steps across all episodes.
            - success_reward: Reward for successful episodes.
            - episode_header: Text to prepend at episode start.

        Args:
            info: Configuration dictionary.

        Returns:
            A new MultiEpisodeEnv instance.
        """
        inner_env_class = info.get("inner_env_class")
        inner_env_kwargs = info.get("inner_env_kwargs", {})
        total_step_cap = info.get("total_step_cap", 30)
        success_reward = info.get("success_reward", 1.0)
        episode_header = info.get("episode_header", "New episode begins.")

        if inner_env_class is None:
            raise ValueError("inner_env_class must be provided in env_args")

        return MultiEpisodeEnv(
            inner_env_class=inner_env_class,
            inner_env_kwargs=inner_env_kwargs,
            total_step_cap=total_step_cap,
            success_reward=success_reward,
            episode_header=episode_header,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        """Indicate whether this environment is safe for multi-threaded use.

        Returns:
            True, as the wrapper itself is thread-safe. The inner environment's
            thread safety should be verified separately.
        """
        return True

