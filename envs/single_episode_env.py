"""Single-episode environment wrapper for rLLM.

This wrapper provides a simple single-episode interface with metrics logging,
analogous to MultiEpisodeEnv but without multi-episode logic. The trajectory
ends when the inner environment returns done=True.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Type

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore


class SingleEpisodeEnv(BaseEnv):
    """Wrapper environment that runs a single episode of an inner environment.

    This environment wraps another BaseEnv and provides:
    - Episode header prepended to observations
    - Shaped rewards (success_reward on success, 0.0 otherwise)
    - Metrics logging compatible with MultiEpisodeAgentPPOTrainer

    Attributes:
        inner_env: The wrapped environment instance.
        success_reward: Reward given when the episode succeeds.
        episode_header: Text prepended to observations.
    """

    def __init__(
        self,
        inner_env_class: Type[BaseEnv] | str,
        inner_env_kwargs: Optional[Dict[str, Any]] = None,
        success_reward: float = 1.0,
        episode_header: str = "",
        **kwargs: Any,
    ):
        """Initialize the single-episode environment wrapper.

        Args:
            inner_env_class: The inner environment class or its fully qualified
                string name (e.g., "envs.gem_env_adapter.GEMEnvAdapter").
            inner_env_kwargs: Keyword arguments to pass to the inner environment
                constructor.
            success_reward: Reward assigned when the episode succeeds.
            episode_header: Text prepended to observations.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        super().__init__()

        # Resolve inner environment class from string if needed
        if isinstance(inner_env_class, str):
            inner_env_class = self._resolve_class(inner_env_class)

        self.inner_env_class = inner_env_class
        self.inner_env_kwargs = dict(inner_env_kwargs or {})
        self.success_reward = float(success_reward)
        self.episode_header = episode_header

        # Create the inner environment
        self.inner_env: BaseEnv = self.inner_env_class(**self.inner_env_kwargs)

        # Episode tracking state (initialized in reset)
        self._episode_step: int = 0
        self._success: bool = False
        self._truncated: bool = False
        self._done: bool = False
        self._task: Optional[dict] = None
        self._seed: Optional[int] = None
        # Store the initial task from dataset (extracted in from_dict)
        self._initial_task: Optional[dict] = None

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
        """Reset the single-episode environment.

        Initializes tracking state and resets the inner environment.

        Args:
            seed: Optional random seed for reproducibility.
            task: Optional task dictionary to pass to inner environment.
                If None, uses the initial task from dataset (set in from_dict).

        Returns:
            observation: Initial observation with episode header.
            info: Initial info dictionary with metadata.
        """
        self._episode_step = 0
        self._success = False
        self._truncated = False
        self._done = False
        # Use provided task, or fall back to initial task from dataset
        self._task = task if task is not None else self._initial_task
        self._seed = seed if seed is not None else self._seed

        # Reset inner environment
        observation, info = self._reset_inner_env()

        # Format observation with episode header
        observation = self._format_observation(observation)

        # Augment info with metadata
        augmented_info = self._augment_info(
            base_info=info,
            episode_step=self._episode_step,
            episode_done=False,
        )

        return observation, augmented_info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute one step in the single-episode environment.

        The trajectory ends when the inner environment returns done=True.

        Args:
            action: Action to execute in the inner environment.

        Returns:
            observation: Next observation.
            reward: Shaped reward (success_reward on success, else 0.0).
            done: True when the inner environment returns done=True.
            info: Augmented info dictionary with metadata.
        """
        # Unwrap Action if needed
        raw_action = action.action if isinstance(action, Action) else action

        # Execute step in inner environment
        observation, env_reward, inner_done, info = self.inner_env.step(raw_action)

        self._episode_step += 1

        # Determine if episode succeeded
        if inner_done:
            self._done = True
            self._success = self._is_episode_success(
                done=inner_done, info=info, reward=env_reward
            )
            # Note: self._truncated is NOT set here based on inner env's truncated flag.
            # It is only set in close() when the execution engine terminates early.

        # Shape reward: success_reward on episode success, 0.0 otherwise
        shaped_reward = self.success_reward if self._success else 0.0

        # Augment info with metadata
        augmented_info = self._augment_info(
            base_info=info,
            episode_step=self._episode_step,
            episode_done=inner_done,
            success=self._success if inner_done else None,
            raw_reward=env_reward,
        )

        # Include full metrics and trajectory info when trajectory completes
        if inner_done:
            augmented_info["metrics"] = self.get_metrics()
            augmented_info["trajectory_info"] = self.get_trajectory_info()
            augmented_info["is_correct"] = self.is_correct

        return observation, shaped_reward, inner_done, augmented_info

    def close(self) -> None:
        """Close the inner environment.

        Also detects if the trajectory was terminated early by the execution
        engine (e.g., TRUNCATION, TIMEOUT, MAX_STEPS) before the inner env
        returned done=True. This is tracked in self._truncated for metrics.
        """
        # If episode didn't complete naturally (inner env didn't return done=True),
        # mark as truncated. This happens when execution engine terminates early.
        if not self._done and self._episode_step > 0:
            self._truncated = True

        if hasattr(self.inner_env, "close"):
            self.inner_env.close()

    def get_metrics(self) -> Dict[str, Any]:
        """Collect metrics for logging.

        Returns metrics compatible with MultiEpisodeAgentPPOTrainer:
        - episode/success_rate: 1.0 if episode succeeded, 0.0 otherwise
        - episode/episode_length: Number of steps taken
        - episode/truncated: 1.0 if trajectory was terminated early by execution
          engine (e.g., max_response_length exceeded, TRUNCATION, TIMEOUT),
          0.0 if episode completed normally (inner env returned done=True)

        Returns:
            Dictionary containing episode metrics.
        """
        return {
            "episode/success_rate": 1.0 if self._success else 0.0,
            "episode/episode_length": self._episode_step,
            "episode/truncated": 1.0 if self._truncated else 0.0,
        }

    def get_trajectory_info(self) -> Dict[str, Any]:
        """Get trajectory-level info for metadata.

        Returns:
            Dictionary containing:
                - single_episode: Always True
                - success: Whether the episode succeeded
                - episode_length: Number of steps taken
                - truncated: Whether trajectory was terminated early by execution engine
        """
        return {
            "single_episode": True,
            "success": self._success,
            "episode_length": self._episode_step,
            "truncated": self._truncated,
        }

    @property
    def is_correct(self) -> bool:
        """Whether the trajectory is considered correct.

        A trajectory is correct if the episode succeeded.

        Returns:
            True if episode succeeded, False otherwise.
        """
        return self._success

    def _reset_inner_env(self) -> tuple[Any, dict]:
        """Reset the inner environment with the stored seed and task.

        Returns:
            observation: Initial observation from inner environment.
            info: Info dictionary from inner environment.
        """
        # Extract seed from task dict if available, otherwise use stored seed
        reset_seed = None
        if self._task is not None and isinstance(self._task, dict):
            reset_seed = self._task.get("seed")
        if reset_seed is None:
            reset_seed = self._seed

        # Try to pass task first to preserve task configuration
        if self._task is not None:
            try:
                return self.inner_env.reset(seed=reset_seed, task=self._task)
            except TypeError:
                try:
                    return self.inner_env.reset(task=self._task)
                except TypeError:
                    if reset_seed is not None:
                        try:
                            return self.inner_env.reset(seed=reset_seed)
                        except TypeError:
                            pass
                    return self.inner_env.reset()
        else:
            if reset_seed is not None:
                try:
                    return self.inner_env.reset(seed=reset_seed)
                except TypeError:
                    return self.inner_env.reset()
            else:
                return self.inner_env.reset()

    def _format_observation(self, observation: Any) -> Any:
        """Prepend episode header to observation.

        Args:
            observation: Raw observation from inner environment.

        Returns:
            Observation with episode header prepended (if string).
        """
        if self.episode_header and isinstance(observation, str):
            return f"{self.episode_header}\n{observation}"
        return observation

    @staticmethod
    def _is_episode_success(done: bool, info: Dict[str, Any], reward: float) -> bool:
        """Determine whether the episode succeeded.

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
        episode_step: int,
        episode_done: bool,
        success: Optional[bool] = None,
        raw_reward: Optional[float] = None,
    ) -> dict:
        """Inject metadata into the info dict.

        Args:
            base_info: Original info from inner environment.
            episode_step: Current step in episode.
            episode_done: Whether the episode just ended.
            success: Whether the episode succeeded (if episode_done).
            raw_reward: Raw reward from inner environment.

        Returns:
            Augmented info dictionary.
        """
        info = dict(base_info or {})
        info.update(
            {
                "episode_step": episode_step,
                "episode_done": episode_done,
                "single_episode": True,
            }
        )
        if success is not None:
            info["episode_success"] = success
        if raw_reward is not None:
            info["raw_reward"] = raw_reward

        return info

    @staticmethod
    def from_dict(info: dict) -> "SingleEpisodeEnv":
        """Factory method to create SingleEpisodeEnv from a dictionary.

        Expected keys:
            - inner_env_class: Class or string path of inner environment.
            - inner_env_kwargs: Dict of kwargs for inner environment.
            - success_reward: Reward for successful episodes.
            - episode_header: Text to prepend at episode start.
            - task: Optional task dictionary from dataset.

        Per-task configuration (from dataset task dict):
            - max_turns_per_episode: Overrides inner_env_kwargs.env_kwargs.max_turns.

        Args:
            info: Configuration dictionary. May contain per-sample task data.

        Returns:
            A new SingleEpisodeEnv instance.
        """
        import logging

        logger = logging.getLogger(__name__)

        inner_env_class = info.get("inner_env_class")
        inner_env_kwargs = info.get("inner_env_kwargs", {})
        success_reward = info.get("success_reward", 1.0)
        episode_header = info.get("episode_header", "")

        if inner_env_class is None:
            raise ValueError("inner_env_class must be provided in env_args")

        # Extract task-related keys from info (these come from dataset extra_info)
        config_keys = {
            "inner_env_class",
            "inner_env_kwargs",
            "success_reward",
            "episode_header",
        }
        task_dict = {k: v for k, v in info.items() if k not in config_keys and v is not None}

        # Extract per-task configuration from task dict
        per_task_env_id = task_dict.get("env_id")
        per_task_max_turns = task_dict.get("max_turns_per_episode")

        # Override inner_env_kwargs with per-task configuration
        inner_env_kwargs = dict(inner_env_kwargs)  # Make a copy

        # Extract env_id from task dict and set it in inner_env_kwargs
        if per_task_env_id is not None:
            inner_env_kwargs["env_id"] = per_task_env_id
            logger.debug(f"Using per-task env_id: {per_task_env_id}")

        if per_task_max_turns is not None:
            # Ensure env_kwargs exists
            if "env_kwargs" not in inner_env_kwargs:
                inner_env_kwargs["env_kwargs"] = {}
            inner_env_kwargs["env_kwargs"] = dict(inner_env_kwargs["env_kwargs"])
            inner_env_kwargs["env_kwargs"]["max_turns"] = int(per_task_max_turns)
            # Add key value pairs to inner_env_kwargs from task_dict
            for k, v in task_dict.items():
                if k not in ("env_id", "max_turns_per_episode", "total_step_cap", "data_source", "seed", "uid"):
                    inner_env_kwargs["env_kwargs"][k] = v
            logger.debug(f"Using per-task max_turns_per_episode: {per_task_max_turns}")

        # Clean task dict for storage
        task_dict_clean = {
            k: v
            for k, v in task_dict.items()
            if k not in ("max_turns_per_episode", "total_step_cap")
        }
        initial_task = task_dict_clean if task_dict_clean else None

        logger.debug(
            f"SingleEpisodeEnv.from_dict: info keys={sorted(info.keys())}, "
            f"task_dict={task_dict}, initial_task={initial_task}"
        )

        env = SingleEpisodeEnv(
            inner_env_class=inner_env_class,
            inner_env_kwargs=inner_env_kwargs,
            success_reward=success_reward,
            episode_header=episode_header,
        )
        # Store the initial task for reset
        env._initial_task = initial_task
        env._task = initial_task

        return env

    @staticmethod
    def is_multithread_safe() -> bool:
        """Indicate whether this environment is safe for multi-threaded use.

        Returns:
            True, as the wrapper itself is thread-safe.
        """
        return True
