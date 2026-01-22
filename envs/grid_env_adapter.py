"""Grid navigation environment and adapter for rLLM.

This module defines a simple grid world with three special tiles:
H (agent), F (flag), S (goal). All other tiles are "*".
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore
from third_party.gem.gem.utils.parsing import extract_last_boxed_answer  # type: ignore


Position = Tuple[int, int]


class GridEnv:
    """Minimal grid environment with a Gym-like API."""

    def __init__(
        self,
        n: int = 4,
        m: int = 4,
        max_turns: Optional[int] = 100,
        seed: Optional[int] = None,
    ) -> None:
        if n <= 0 or m <= 0:
            raise ValueError("n and m must be positive integers.")
        self.n = n
        self.m = m
        self.max_turns = max_turns
        self._rng = random.Random(seed)
        self._seed = seed

        self.h_pos: Position = (0, 0)
        self.f_pos: Optional[Position] = None
        self.s_pos: Position = (0, 0)
        self.f_collected: bool = False
        self.turn: int = 0
        self._done: bool = False

    def reset(self, seed: Optional[int] = None, task: Optional[dict] = None) -> tuple[str, dict]:
        if seed is None and task is not None:
            seed = task.get("seed")
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            self._rng.seed(self._seed)

        positions = self._rng.sample(range(self.n * self.m), 3)
        self.h_pos = self._index_to_pos(positions[0])
        self.f_pos = self._index_to_pos(positions[1])
        self.s_pos = self._index_to_pos(positions[2])
        self.f_collected = False
        self.turn = 0
        self._done = False

        observation = self._format_obs()
        info = self._build_info(terminated=False, truncated=False, reward=0.0)
        return observation, info

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        normalized_action = self._normalize_action(action)
        self.turn += 1

        if normalized_action["action"] in {"up", "right", "down", "left"}:
            self.h_pos = self._move(self.h_pos, normalized_action["action"])

        if self.f_pos is not None and self.h_pos == self.f_pos:
            self.f_collected = True
            self.f_pos = None

        terminated = self.h_pos == self.s_pos and self.f_collected
        reward = 1.0 if terminated else 0.0

        truncated = False
        if self.max_turns is not None and self.turn >= self.max_turns:
            truncated = True

        done = bool(terminated or truncated)
        self._done = done

        observation = self._format_obs()
        info = self._build_info(terminated=terminated, truncated=truncated, reward=reward)
        info["last_action"] = normalized_action
        info["success"] = bool(terminated)
        return observation, reward, terminated, truncated, info

    def _normalize_action(self, action: Any) -> Dict[str, str]:
        if isinstance(action, Action):
            action = action.action

        if isinstance(action, str):
            extracted = extract_last_boxed_answer(action)
            if extracted is not None:
                action = extracted.strip()
            action = action.strip().lower().strip(".! ")
            action = action.split()[0] if action else ""
            if action in {"up", "right", "down", "left"}:
                return {"action": action}
            return {"action": "noop"}

        if isinstance(action, dict):
            action_value = str(action.get("action", "")).lower().strip()
            if action_value in {"up", "right", "down", "left"}:
                return {"action": action_value}
            return {"action": "noop"}

        return {"action": "noop"}

    def _move(self, pos: Position, action: str) -> Position:
        row, col = pos
        if action == "up":
            row = max(0, row - 1)
        elif action == "down":
            row = min(self.n - 1, row + 1)
        elif action == "left":
            col = max(0, col - 1)
        elif action == "right":
            col = min(self.m - 1, col + 1)
        return row, col

    def _format_obs(self) -> str:
        grid = [["*" for _ in range(self.m)] for _ in range(self.n)]
        if self.f_pos is not None:
            f_row, f_col = self.f_pos
            grid[f_row][f_col] = "F"
        s_row, s_col = self.s_pos
        grid[s_row][s_col] = "S"

        h_row, h_col = self.h_pos
        grid[h_row][h_col] = "H"

        return "\n".join(" ".join(row) for row in grid)

    def _build_info(self, terminated: bool, truncated: bool, reward: float) -> dict:
        return {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "raw_reward": float(reward),
            "turn": self.turn,
            "max_turns": self.max_turns,
            "n": self.n,
            "m": self.m,
            "h_pos": self.h_pos,
            "f_pos": self.f_pos,
            "s_pos": self.s_pos,
            "f_collected": self.f_collected,
        }

    def _index_to_pos(self, idx: int) -> Position:
        return divmod(idx, self.m)


class GridEnvAdapter(BaseEnv):
    """Adapter exposing GridEnv via the rLLM BaseEnv contract."""

    def __init__(
        self,
        env_id: Optional[str] = None,  # Ignored; accepted for compatibility with MultiEpisodeEnv
        env_kwargs: Optional[dict] = None,
        n: int = 5,
        m: int = 5,
        max_turns: Optional[int] = 100,
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        if env_kwargs:
            n = int(env_kwargs.get("n", env_kwargs.get("rows", n)))
            m = int(env_kwargs.get("m", env_kwargs.get("cols", m)))
            max_turns = env_kwargs.get("max_turns", max_turns)
            seed = env_kwargs.get("seed", seed)

        self.n = n
        self.m = m
        self.max_turns = max_turns
        self._seed = seed
        self._env = GridEnv(n=n, m=m, max_turns=max_turns, seed=seed)

    def reset(self, seed: Optional[int] = None, task: Optional[dict] = None) -> tuple[Any, dict]:
        reset_seed = seed if seed is not None else self._seed
        self._seed = reset_seed
        obs, info = self._env.reset(seed=reset_seed, task=task)
        return self._rules_prompt_with_obs(obs), info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        enriched_info = dict(info)
        enriched_info.update(
            {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "raw_reward": float(reward),
                "max_turns": self.max_turns,
            }
        )

        if terminated:
            obs_text = f"Current observation:\n{obs}\nYou succeed!"
            return obs_text, float(reward), True, enriched_info

        obs_text = f"Current observation:\n{obs}\nPlease select your action."
        return obs_text, float(reward), done, enriched_info

    def close(self) -> None:
        return None

    @staticmethod
    def from_dict(info: dict) -> "GridEnvAdapter":
        return GridEnvAdapter(
            n=info.get("n", 5),
            m=info.get("m", 5),
            max_turns=info.get("max_turns", 100),
            seed=info.get("seed"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    @staticmethod
    def _rules_prompt_with_obs(obs_text: str) -> str:
        prompt = (
            "You are playing a game.\n"
            "Available actions are up, right, down, left.\n"
            "There are no instructions. Play the game to discover controls, rules, and goal.\n"
            f"Current observation:\n{obs_text}\n"
            "Output your action using \\box{action} and nothing else."
        )
        return prompt
