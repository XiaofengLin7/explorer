from __future__ import annotations

from typing import Any, Dict, Optional, List
import re
import numpy as np

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore


class _PolarizedAdversaryDistribution:
    """
    Hidden adversary distribution over {rock, paper, scissors} with a dominant action.

    Parameters
    ----------
    min_dom : float
        Minimum probability for the dominant action, in (0, 1). The sampler guarantees
        P(dominant) >= min_dom.

    Internal
    --------
    Uses t ~ Beta(2, 2) to spread dominant probability within [min_dom, 1),
    then allocates the remaining mass to the other two actions via Dirichlet([1, 1]).
    """

    LABELS = ("rock", "paper", "scissors")

    def __init__(self, *, min_dom: float):
        if not (0.0 < min_dom < 1.0):
            raise ValueError("min_dom must be in (0, 1)")
        self.min_dom = float(min_dom)
        self._shape = 2.0  # fixed mild concentration

    def sample(self, seed: int) -> Dict[str, float]:
        rng = np.random.default_rng(seed)

        # Choose dominant index uniformly
        dom_idx = int(rng.integers(0, 3))

        # Dominant prob in [min_dom, 1)
        t = float(rng.beta(self._shape, self._shape))
        dom_prob = self.min_dom + (1.0 - self.min_dom) * t

        # Split the remaining mass over the two non-dominant actions
        remainder = 1.0 - dom_prob
        rest = rng.dirichlet([1.0, 1.0]) * remainder

        probs = np.zeros(3, dtype=float)
        others = [i for i in (0, 1, 2) if i != dom_idx]
        probs[dom_idx] = dom_prob
        probs[others[0]] = float(rest[0])
        probs[others[1]] = float(rest[1])

        return {label: float(p) for label, p in zip(self.LABELS, probs)}


class RockPaperScissorsEnvAdapter(BaseEnv):
    """
    Multi-turn Rock-Paper-Scissors environment with a hidden per-episode polarized distribution.

    Episode mechanics
    -----------------
    - reset(seed) creates a polarized hidden distribution over {rock, paper, scissors}
      with one action having probability >= min_dom (configurable).
    - For each step, the adversary action is sampled i.i.d. from that fixed distribution
      using an unseeded RNG (true random; not controlled by reset seed).

    Observations
    ------------
    - Non-final turns: "You win" | "You lose" | "You draw"
    - Final turn:      "You win" | "You lose" | "You draw. Now play the next episode."

    Rewards
    -------
    - Only the final turn can produce a non-zero reward.
    - Let wins be the number of turns where the agent beats the sampled adversary action.
      Final reward = 1 iff (wins / max_turns) >= 1/2, else 0.

    Action formatting
    -----------------
    - The agent should output its action within \\boxed{...}, e.g., \\boxed{rock}.
    """

    ACTIONS = {"rock", "paper", "scissors"}

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        self.env_id = env_id
        if env_kwargs is None:
            raise ValueError("env_kwargs must include: max_turns (int > 0), min_dom (0 < float < 1)")

        # Required: only these two keys remain
        self.max_turns = env_kwargs.get("max_turns", None)
        self.min_dom = env_kwargs.get("min_dom", None)

        if not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("env_kwargs['max_turns'] must be a positive int")
        if not isinstance(self.min_dom, (int, float)) or not (0.0 < float(self.min_dom) < 1.0):
            raise ValueError("env_kwargs['min_dom'] must be a float in (0, 1)")

        # Distribution generator (polarized)
        self._dist_gen = _PolarizedAdversaryDistribution(min_dom=float(self.min_dom))

        # Episode state
        self.turn: int = 0
        self.terminated: bool = False

        # Hidden distribution and per-turn RNG
        self._probs: Optional[Dict[str, float]] = None
        self._rng_turn: Optional[np.random.Generator] = None

        # Outcome accumulation for final reward
        self._wins: int = 0
        self._losses: int = 0
        self._draws: int = 0

    # ---------- Small utils ---------- #
    @staticmethod
    def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
        s = float(sum(dist.values()))
        return {k: float(v) / s for k, v in dist.items()}

    def _parse_action(self, raw_text: str) -> str:
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        m = re.search(boxed_pattern, str(raw_text), re.IGNORECASE)
        return (m.group(1) if m else str(raw_text)).strip().lower()

    @staticmethod
    def _outcome(agent: str, adv: str) -> str:
        if agent not in {"rock", "paper", "scissors"}:
            return "lose"
        if agent == adv:
            return "draw"
        if (agent, adv) in {("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")}:
            return "win"
        return "lose"

    @staticmethod
    def _format_outcome_obs(outcome: str, is_last_turn: bool) -> str:
        base = f"You {outcome}"
        return base + ". Now play the next episode." if is_last_turn else base

    # ---------- BaseEnv API ---------- #
    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[str, Dict[str, Any]]:
        if seed is None:
            raise ValueError("Seed must be provided.")

        self.turn = 0
        self.terminated = False
        self._wins = self._losses = self._draws = 0

        # Build (and fix) the polarized hidden distribution for this episode
        self._probs = self._dist_gen.sample(seed)

        # Per-turn RNG: unseeded (true random, decoupled from reset seed)
        self._rng_turn = np.random.default_rng()

        # Initial instruction (no thinking tag, still enforce boxed output)
        initial = (
            "You are playing a multi-turn Rock-Paper-Scissors game against an adversary. "
            "In each episode, at every turn, the adversary's action is sampled from a fixed (hidden) distribution determined by the seed. "
            "Your objective is to choose the action that maximizes the probability of winning against this hidden distribution each turn. "
            "Now let's start the game. Output your action within \\boxed{...}."
        )
        info = {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": False,
            "truncated": False,
        }
        return initial, info

    def step(self, action: Any) -> tuple[str, float, bool, Dict[str, Any]]:

        if self._probs is None or self._rng_turn is None:
            raise ValueError("Environment not initialized. Please call reset() first.")

        # Sample adversary action (true random) from the fixed hidden distribution
        labels = np.array(["rock", "paper", "scissors"])
        p = np.array([self._probs["rock"], self._probs["paper"], self._probs["scissors"]], dtype=float)
        p = p / p.sum()
        adv_action = str(self._rng_turn.choice(labels, p=p))

        # Parse agent action
        agent_action = self._parse_action(action.action if isinstance(action, Action) else action)

        # Outcome for observation and accumulation
        outcome = self._outcome(agent_action, adv_action)
        if outcome == "win":
            self._wins += 1
        elif outcome == "lose":
            self._losses += 1
        else:
            self._draws += 1

        # Advance turn and check termination
        self.turn += 1
        is_last = self.turn >= self.max_turns
        if is_last:
            self.terminated = True

        # Final-only reward
        if is_last:
            win_rate = self._wins / float(self.max_turns)
            reward = 1.0 if win_rate >= 0.5 else 0.0
        else:
            reward = 0.0

        # Observation formatting
        obs = self._format_outcome_obs(outcome, is_last_turn=is_last)


        info = {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": is_last,
            "truncated": False,
            "raw_reward": float(reward),
            "hidden": {
                "adversary_distribution": self._probs,
                "wins": self._wins,
                "losses": self._losses,
                "draws": self._draws,
                "final_win_rate_threshold": 0.5,
                "min_dom": float(self.min_dom),
            } if is_last else None,
        }
        return obs, float(reward), is_last, info

    def close(self) -> None:
        pass

    @staticmethod
    def from_dict(info: dict) -> "RockPaperScissorsEnvAdapter":
        env_id = info["env_id"]
        env_kwargs = info.get("env_kwargs", None)
        return RockPaperScissorsEnvAdapter(env_id=env_id, env_kwargs=env_kwargs)
