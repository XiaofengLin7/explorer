"""Multi-episode rollout function using rLLM SDK to prevent retokenization.

This module implements the same multi-episode logic as MultiEpisodeWorkflow,
but uses the SDK pattern (get_chat_client_async + session) to ensure that
token IDs are captured directly from vLLM and stored in SQLite, avoiding
retokenization mismatch during training.

Usage:
    The `multi_episode_rollout` function is passed to `AgentTrainer` as
    `agent_run_func`. It receives task kwargs and returns a list of
    TrajectoryView objects with step-level rewards.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from rllm.sdk import TrajectoryView, get_chat_client_async, session
from rllm.sdk.protocol import StepView


# Default proxy URL - will be overridden by AgentSdkEngine during training
DEFAULT_PROXY_URL = "http://localhost:4000/v1"

# Pattern to extract boxed answers
BOXED_PATTERN = re.compile(r"\\boxed{([^}]+)}")


def extract_last_boxed(text: str) -> str:
    """Extract the last ``\\boxed{...}`` substring; return raw text if none found."""
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return text.strip()
    return matches[-1].group(1).strip()


class MultiEpisodeSDKRunner:
    """Encapsulates multi-episode rollout logic for SDK-based training.

    This class mirrors the logic in MultiEpisodeWorkflow but is designed
    to work with the SDK pattern. It:
    - Creates an environment instance
    - Maintains chat history like GEMTextAgent
    - Makes LLM calls through the SDK client (tracked by session)
    - Runs multiple episodes until step budget is exhausted
    - Returns TrajectoryView with all steps and success count as reward
    """

    def __init__(
        self,
        env_cls: type,
        env_args: dict[str, Any],
        system_prompt: str | None = None,
        total_step_cap: int | None = None,
        min_episodes: int = 3,
        success_reward: float = 1.0,
        episode_header: str = "New episode begins.",
        model_name: str = "default",
        proxy_url: str = DEFAULT_PROXY_URL,
        temperature: float = 0.6,
        max_tokens: int = 8192,
    ):
        """Initialize the multi-episode SDK runner.

        Args:
            env_cls: Environment class (e.g., GEMEnvAdapter).
            env_args: Arguments passed to environment constructor.
            system_prompt: System prompt for the agent.
            total_step_cap: Maximum steps across all episodes.
            min_episodes: Soft target for how many episodes to attempt.
            success_reward: Reward assigned when an episode succeeds.
            episode_header: Text prepended to observation on each reset.
            model_name: Model name to use for LLM calls.
            proxy_url: URL of the LiteLLM proxy.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens for LLM response.
        """
        self.env_cls = env_cls
        self.env_args = env_args or {}
        self.system_prompt = system_prompt or (
            "You are solving a task. Return your final answer inside \\boxed{}."
        )
        self.total_step_cap = total_step_cap
        self.min_episodes = max(1, int(min_episodes))
        self.success_reward = float(success_reward)
        self.episode_header = episode_header
        self.model_name = model_name
        self.proxy_url = proxy_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Will be set during run
        self._messages: list[dict[str, str]] = []
        self._env = None
        self._client = None
        self._task = None

    def _reset_messages(self) -> None:
        """Reset chat history with system prompt."""
        self._messages = [{"role": "system", "content": self.system_prompt}]

    def _add_user_message(self, content: str) -> None:
        """Add user message (observation) to chat history."""
        self._messages.append({"role": "user", "content": content})

    def _add_assistant_message(self, content: str) -> None:
        """Add assistant message (model response) to chat history."""
        self._messages.append({"role": "assistant", "content": content})

    def _format_observation(self, observation: Any, episode_index: int) -> str:
        """Prepend an episode marker to the observation."""
        header = f"[Episode {episode_index + 1}] {self.episode_header}".strip()
        return f"{header}\n{observation}"

    def _infer_step_cap(self) -> int:
        """Infer total step cap from config or use default."""
        if self.total_step_cap is not None:
            return self.total_step_cap
        # Try to get max_turns from env_kwargs
        env_kwargs = self.env_args.get("env_kwargs", {})
        max_turns = env_kwargs.get("max_turns")
        if isinstance(max_turns, int) and max_turns > 0:
            return max_turns * 3
        return 50

    def _reset_env(self) -> tuple[Any, dict]:
        """Reset the environment with a fixed seed when available."""
        seed = None
        try:
            seed = self._task.get("seed") if isinstance(self._task, dict) else None
        except Exception:
            seed = None

        try:
            return self._env.reset(seed=seed, task=self._task)
        except TypeError:
            try:
                return self._env.reset(seed=seed)
            except TypeError:
                return self._env.reset()

    @staticmethod
    def _is_episode_success(done: bool, info: dict[str, Any], reward: float) -> bool:
        """Determine whether the just-finished episode succeeded."""
        if not done:
            return False
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if truncated:
            return False
        return terminated and reward > 0

    async def run(self, task: dict[str, Any]) -> list[TrajectoryView]:
        """Execute multi-episode rollout and return trajectories.

        Args:
            task: Task dictionary containing task specification.

        Returns:
            List containing a single TrajectoryView with all steps
            and reward = success count.
        """
        self._task = task

        # Initialize environment
        env_kwargs = dict(self.env_args.get("env_kwargs", {}))
        env_init_args = {
            k: v for k, v in self.env_args.items() if k not in {"env_kwargs", "total_step_cap"}
        }
        try:
            self._env = self.env_cls(env_kwargs=env_kwargs, **env_init_args)
        except TypeError:
            try:
                self._env = self.env_cls(**env_init_args)
            except TypeError:
                self._env = self.env_cls(**env_kwargs)

        # Initialize SDK client
        self._client = get_chat_client_async(base_url=self.proxy_url, api_key="EMPTY")

        # Reset chat history
        self._reset_messages()

        step_cap = self._infer_step_cap()
        episode_successes: list[bool] = []
        episode_lengths: list[int] = []
        total_steps = 0
        episode_index = 0

        # Use session to track all LLM calls
        with session(agent="multi_episode") as sess:
            observation, info = self._reset_env()

            while total_steps < step_cap:
                episode_step = 0
                observation_with_header = self._format_observation(observation, episode_index)

                # Add initial observation as user message
                self._add_user_message(observation_with_header)

                while total_steps < step_cap:
                    # Make LLM call through SDK client (tracked by session)
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=copy.deepcopy(self._messages),
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                    response_text = response.choices[0].message.content or ""
                    finish_reason = response.choices[0].finish_reason

                    # Check for length-based termination
                    if finish_reason == "length":
                        # Model hit max tokens - treat as failed step but continue
                        pass

                    # Add assistant response to history
                    self._add_assistant_message(response_text)

                    # Parse action and step environment
                    parsed_action = extract_last_boxed(response_text)
                    boxed_action = f"\\boxed{{{parsed_action}}}"

                    next_observation, env_reward, done, step_info = self._env.step(boxed_action)

                    total_steps += 1
                    episode_step += 1

                    success = self._is_episode_success(
                        done=done, info=step_info, reward=env_reward
                    )

                    if done or total_steps >= step_cap:
                        episode_successes.append(success)
                        episode_lengths.append(episode_step)
                        break
                    else:
                        # Add next observation as user message for continued conversation
                        self._add_user_message(str(next_observation))

                if total_steps >= step_cap:
                    break

                # Start next episode
                observation, info = self._reset_env()
                episode_index += 1

                if episode_index + 1 >= self.min_episodes and total_steps >= step_cap:
                    break

            # Collect steps from session
            steps: list[StepView] = sess.steps

        # Calculate final reward = success count
        success_count = sum(episode_successes)
        final_reward = float(success_count) * self.success_reward

        # Assign reward to last step only (GRPO pattern)
        if steps:
            steps[-1].reward = final_reward

        # Create trajectory view
        trajectory = TrajectoryView(
            name="multi_episode_agent",
            steps=steps,
            reward=final_reward,
            metadata={
                "multi_episode": True,
                "episode_successes": episode_successes,
                "success_count": success_count,
                "num_episodes": len(episode_successes),
                "total_steps": total_steps,
                "step_cap": step_cap,
                "episode_lengths": episode_lengths,
            },
        )

        return [trajectory]


async def multi_episode_rollout(
    # Task fields (passed as kwargs from dataset)
    seed: int | None = None,
    # Configuration fields (passed via config)
    env_cls: type | None = None,
    env_args: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    total_step_cap: int | None = None,
    min_episodes: int = 3,
    success_reward: float = 1.0,
    episode_header: str = "New episode begins.",
    model: str = "default",
    proxy_url: str = DEFAULT_PROXY_URL,
    temperature: float = 0.6,
    max_tokens: int = 8192,
    **kwargs,
) -> list[TrajectoryView]:
    """Multi-episode rollout function for SDK-based training.

    This function is passed to AgentTrainer as `agent_run_func`.
    It implements the same logic as MultiEpisodeWorkflow but uses
    the SDK pattern to prevent retokenization.

    Args:
        seed: Task seed for reproducibility.
        env_cls: Environment class to instantiate.
        env_args: Arguments for environment constructor.
        system_prompt: System prompt for the agent.
        total_step_cap: Maximum steps across all episodes.
        min_episodes: Target number of episodes to attempt.
        success_reward: Reward per successful episode.
        episode_header: Header text for each episode.
        model: Model name for LLM calls.
        proxy_url: LiteLLM proxy URL.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        **kwargs: Additional task fields.

    Returns:
        List of TrajectoryView objects (single trajectory with all steps).
    """
    # Merge task dict from kwargs
    task = {"seed": seed, **kwargs}

    runner = MultiEpisodeSDKRunner(
        env_cls=env_cls,
        env_args=env_args or {},
        system_prompt=system_prompt,
        total_step_cap=total_step_cap,
        min_episodes=min_episodes,
        success_reward=success_reward,
        episode_header=episode_header,
        model_name=model,
        proxy_url=proxy_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return await runner.run(task)


def create_multi_episode_rollout_func(
    env_cls: type,
    env_args: dict[str, Any],
    system_prompt: str | None = None,
    total_step_cap: int | None = None,
    min_episodes: int = 3,
    success_reward: float = 1.0,
    episode_header: str = "New episode begins.",
    model_name: str = "default",
    temperature: float = 0.6,
    max_tokens: int = 8192,
):
    """Factory function to create a configured multi-episode rollout function.

    This is a convenience function that pre-configures the rollout function
    with environment and agent settings, returning a function that only
    needs the task kwargs.

    Args:
        env_cls: Environment class to instantiate.
        env_args: Arguments for environment constructor.
        system_prompt: System prompt for the agent.
        total_step_cap: Maximum steps across all episodes.
        min_episodes: Target number of episodes to attempt.
        success_reward: Reward per successful episode.
        episode_header: Header text for each episode.
        model_name: Model name for LLM calls (placeholder, will be set by proxy).
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        Async function that takes task kwargs and returns TrajectoryView list.

    Example:
        >>> from envs.gem_env_adapter import GEMEnvAdapter
        >>> rollout_fn = create_multi_episode_rollout_func(
        ...     env_cls=GEMEnvAdapter,
        ...     env_args={"env_id": "game:GuessTheNumber-v0-hard"},
        ...     total_step_cap=21,
        ... )
        >>> # Pass to AgentTrainer
        >>> trainer = AgentTrainer(agent_run_func=rollout_fn, ...)
    """
    async def configured_rollout(**task_kwargs) -> list[TrajectoryView]:
        runner = MultiEpisodeSDKRunner(
            env_cls=env_cls,
            env_args=env_args,
            system_prompt=system_prompt,
            total_step_cap=total_step_cap,
            min_episodes=min_episodes,
            success_reward=success_reward,
            episode_header=episode_header,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await runner.run(task_kwargs)

    return configured_rollout



