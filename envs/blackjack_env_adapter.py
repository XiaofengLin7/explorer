"""Lightweight Blackjack environment and adapter for rLLM.

This module defines a small, self-contained Blackjack environment with a
Gym-like API (reset/step) and an adapter that exposes it through the rLLM
``BaseEnv`` interface. It avoids external dependencies so it can be used
without installing Gym.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore
from third_party.gem.gem.utils.parsing import extract_last_boxed_answer  # type: ignore


Card = int


def _hand_value(hand: List[Card]) -> Tuple[int, bool]:
    total = sum(hand)
    usable_ace = 1 in hand and total + 10 <= 21
    if usable_ace:
        total += 10
    return total, usable_ace


class BlackjackEnv:
    """Minimal Blackjack environment with a Gym-like API and explicit deck state."""

    def __init__(
        self,
        max_turns: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        self.max_turns = max_turns
        self._rng = random.Random(seed)
        self._seed = seed

        self.player: List[Card] = []
        self.dealer: List[Card] = []
        self.deck: List[Card] = []
        self._taken_indices: set[int] = set()
        self._available_indices: List[int] = []
        self.turn: int = 0
        self._done: bool = False

    def reset(self, seed: Optional[int] = None, task: Optional[dict] = None) -> tuple[Dict[str, Any], dict]:
        if seed is None and task is not None:
            seed = task.get("seed")
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            self._rng.seed(self._seed)

        # Build a deck and ensure the dealer's initial two-card total is at least 17.
        # If not, reshuffle and retry (bounded attempts).
        max_attempts = 50
        attempts = 0
        while True:
            self.deck = self._build_deck()
            self.dealer = [self.deck.pop(0), self.deck.pop(0)]
            dealer_total, _ = _hand_value(self.dealer)
            if dealer_total >= 17 or attempts >= max_attempts:
                break
            attempts += 1
        # Player draws after dealer is fixed.
        self.player = [self.deck.pop(0), self.deck.pop(0)]

        # Remaining deck is 52 - 4 = 48 cards; keep length fixed for the episode.
        self._taken_indices = set()
        self._available_indices = list(range(len(self.deck)))
        self.turn = 0
        self._done = False

        player_total, _ = _hand_value(self.player)
        dealer_total, _ = _hand_value(self.dealer)

        terminated = False
        reward = 0.0
        natural = False

        observation = self._format_obs(reveal_dealer=terminated)
        info = self._build_info(
            terminated=terminated,
            truncated=False,
            reward=reward,
            natural=natural,
            player_total=player_total,
            dealer_total=dealer_total if terminated else None,
        )

        self._done = terminated
        return observation, info

    def step(self, action: Any) -> tuple[Dict[str, Any], float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        normalized_action = self._normalize_action(action)
        # If the requested hit index is no longer available, fall back to stand.
        if normalized_action.get("action") == "hit":
            idx = normalized_action.get("card_index")
            if (
                idx is None
                or not self._available_indices
                or idx not in self._available_indices
            ):
                normalized_action = {"action": "stand"}

        self.turn += 1
        truncated = self.turn >= self.max_turns

        if normalized_action["action"] == "hit":
            card = self._take_card(normalized_action["card_index"])
            self.player.append(card)
            player_total, _ = _hand_value(self.player)
            if player_total > 21:
                terminated = True
                reward = 0.0
            else:
                terminated = False
                reward = 0.0
        else:  # stand
            player_total, _ = _hand_value(self.player)
            terminated = True
            reward = 0.0
            # Dealer does not draw; compare against current two-card hand.
            dealer_total, _ = _hand_value(self.dealer)

            if player_total <= 21 and dealer_total <= 21 and player_total > dealer_total:
                reward = 1.0

        done = terminated or truncated
        self._done = done

        observation = self._format_obs(reveal_dealer=done)
        info = self._build_info(
            terminated=terminated,
            truncated=truncated,
            reward=reward,
            natural=False,
            player_total=_hand_value(self.player)[0],
            dealer_total=_hand_value(self.dealer)[0] if done else None,
            last_drawn_card=card if normalized_action["action"] == "hit" else None,
        )
        info["last_action"] = normalized_action

        return observation, reward, terminated, truncated, info

    def _normalize_action(self, action: Any) -> Dict[str, Union[str, int]]:
        """Normalize actions, accepting boxed strings and forgiving formats."""
        if isinstance(action, Action):
            action = action.action

        # Strip last boxed content using GEM helper
        if isinstance(action, str):
            extracted = extract_last_boxed_answer(action)
            if extracted is not None:
                action = extracted.strip()
            else:
                action = action.strip()

        # Treat None or empty as stand to avoid crashes
        if action is None:
            return {"action": "stand"}

        if isinstance(action, str):
            cleaned = action.strip().strip(".!").lower()
            if not cleaned:
                return {"action": "stand"}
            parts = cleaned.split()
            if parts[0] in {"stand", "s", "stick", "stay"}:
                return {"action": "stand"}
            if parts[0] in {"hit", "h"}:
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                        return {"action": "hit", "card_index": idx}
                    except ValueError:
                        pass  # fall back below
                # Default to first available index if provided
                if self._available_indices:
                    return {"action": "hit", "card_index": self._available_indices[0]}
                return {"action": "stand"}

        if isinstance(action, dict):
            act = str(action.get("action", "")).lower()
            if act in {"stand", "s", "stick", "stay"}:
                return {"action": "stand"}
            if act in {"hit", "h"}:
                if "card_index" in action:
                    try:
                        return {"action": "hit", "card_index": int(action["card_index"])}
                    except Exception:
                        return {"action": "stand"}
                if self._available_indices:
                    return {"action": "hit", "card_index": self._available_indices[0]}
                return {"action": "stand"}

        # Fallback: treat any unsupported/empty action as stand.
        return {"action": "stand"}

    def _format_obs(self, reveal_dealer: bool) -> Dict[str, Any]:
        player_total, player_soft = _hand_value(self.player)
        dealer_total, dealer_soft = _hand_value(self.dealer)

        deck_view = [
            self._card_to_str(self.deck[idx]) if idx in self._taken_indices else "?"
            for idx in range(len(self.deck))
        ]

        return {
            "player_hand": list(self.player),
            "dealer_upcard": self.dealer[0],
            "dealer_hand": list(self.dealer) if reveal_dealer else [self.dealer[0], None],
            "player_total": player_total,
            "dealer_total": dealer_total if reveal_dealer else None,
            "player_soft": player_soft,
            "dealer_soft": dealer_soft if reveal_dealer else None,
            "turn": self.turn,
            "legal_actions": ["hit <card_index>", "stand"],
            "deck": deck_view,
            "available_indices": list(self._available_indices),
        }

    def _build_info(
        self,
        terminated: bool,
        truncated: bool,
        reward: float,
        natural: bool,
        player_total: int,
        dealer_total: Optional[int],
        last_drawn_card: Optional[int] = None,
    ) -> dict:
        return {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "raw_reward": float(reward),
            "natural": bool(natural),
            "player_total": player_total,
            "dealer_total": dealer_total,
            "max_turns": self.max_turns,
            "available_indices": list(self._available_indices),
            "dealer_hand": list(self.dealer),
            "player_hand": list(self.player),
            "last_drawn_card": last_drawn_card,
        }

    def _build_deck(self) -> List[Card]:
        deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        self._rng.shuffle(deck)
        return deck

    @staticmethod
    def _card_to_str(card: Card) -> str:
        if card == 1:
            return "A"
        if card == 10:
            return "10"
        return str(card)

    def _take_card(self, explicit_index: Optional[int] = None) -> Card:
        if not self.deck:
            raise RuntimeError("Deck is empty.")

        if not self._available_indices:
            raise RuntimeError("No cards available to draw.")

        if explicit_index is None:
            idx = self._available_indices[0]
        else:
            if explicit_index < 0 or explicit_index >= len(self.deck):
                raise ValueError(f"Card index {explicit_index} is not available.")
            if explicit_index in self._taken_indices:
                raise ValueError(f"Card index {explicit_index} has already been taken.")
            idx = explicit_index

        card = self.deck[idx]
        self._taken_indices.add(idx)
        self._available_indices = [
            i for i in range(len(self.deck)) if i not in self._taken_indices
        ]
        return card


class BlackjackEnvAdapter(BaseEnv):
    """Adapter exposing BlackjackEnv via the rLLM BaseEnv contract."""

    def __init__(
        self,
        env_id: Optional[str] = None,  # Ignored; accepted for compatibility with MultiEpisodeEnv
        env_kwargs: Optional[dict] = None,  # Ignored; accepted for compatibility
        max_turns: int = 30,
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        self.max_turns = max_turns
        self._seed = seed
        self._env = BlackjackEnv(
            max_turns=max_turns,
            seed=seed,
        )

    def reset(self, seed: Optional[int] = None, task: Optional[dict] = None) -> tuple[Any, dict]:
        reset_seed = seed if seed is not None else self._seed
        self._seed = reset_seed
        obs, info = self._env.reset(seed=reset_seed, task=task)
        obs_txt = self._render_observation(obs)
        return self._rules_prompt_with_obs(obs_txt), info

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
        obs_text = self._render_observation(obs)

        last_action = info.get("last_action", {})
        action_name = None
        action_idx = None
        if isinstance(last_action, dict):
            action_name = last_action.get("action")
            action_idx = last_action.get("card_index")
        elif isinstance(last_action, str):
            action_name = last_action

        drawn_card = info.get("last_drawn_card")
        drawn_card_str = (
            self._card_to_str(drawn_card) if drawn_card is not None else None
        )

        player_total_info = info.get("player_total")
        busted = (
            bool(info.get("terminated"))
            and player_total_info is not None
            and player_total_info > 21
        )

        if action_name == "hit":
            action_summary = (
                f"You chose to hit{f' {action_idx}' if action_idx is not None else ''}"
            )
            if drawn_card_str is not None:
                action_summary += f" and drew {drawn_card_str}"
            action_summary += "."
        elif action_name:
            action_summary = f"You chose to {action_name}."
        else:
            action_summary = "You chose an action."

        if done:
            dealer_hand = info.get("dealer_hand", [])
            player_hand = info.get("player_hand", [])
            dealer_str = ", ".join(self._card_to_str(c) for c in dealer_hand)
            player_str = ", ".join(self._card_to_str(c) for c in player_hand)
            outcome = "win" if reward > 0 else "lose or tie"
            if busted:
                obs_text = (
                    f"{action_summary} Dealer cards: {dealer_str}. "
                    f"Your cards: {player_str}. You lose because you bust."
                )
            else:
                obs_text = (
                    f"{action_summary} Dealer cards: {dealer_str}. "
                    f"Your cards: {player_str}. You {outcome}."
                )
        else:
            obs_text = (
                f"{action_summary} Current observation:\n{obs_text}\n\n"
                "Please select your action."
            )
        return obs_text, float(reward), done, enriched_info

    def close(self) -> None:
        # No external resources to release, provided for API completeness.
        return None

    @staticmethod
    def from_dict(info: dict) -> "BlackjackEnvAdapter":
        return BlackjackEnvAdapter(
            max_turns=info.get("max_turns", 30),
            seed=info.get("seed"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    def _render_observation(self, obs: Dict[str, Any]) -> str:
        player_cards = ", ".join(self._card_to_str(c) for c in obs["player_hand"])
        dealer_cards = (
            ", ".join(self._card_to_str(c) for c in obs["dealer_hand"])
            if obs["dealer_total"] is not None
            else f"{self._card_to_str(obs['dealer_upcard'])}, ?"
        )
        deck_display = " ".join(
            f"{i}:{c}"
            for i, c in enumerate(obs["deck"])
        )
        lines = [
            f"Dealer: {dealer_cards}",
            f"Your hand ({obs['player_total']}): {player_cards}",
            f"Deck (index:value, ?=hidden, drawn=revealed): {deck_display}",
            f"Available indices: {obs['available_indices']}",
            "Actions: 'stand' or 'hit <card_index>' (choose an available index)",
        ]
        return "\n".join(lines)

    @staticmethod
    def _rules_prompt_with_obs(obs_text: str) -> str:
        BLACKJACK_ENV_RESET_PROMPT = """
You are an agent playing a simplified game of Blackjack against a dealer.

Game Objective:
Your goal is to choose actions that maximize your chance of winning against the dealer.

- The objective is to get a hand value as close to 21 as possible without exceeding 21.
- If your hand value exceeds 21, you bust and immediately lose.
- After you stand, the game ends and your hand is compared with the dealer's hand.

Card Values:
- Number cards (2–10): face value
- Face cards (J, Q, K): value 10
- Ace (A): value 1 or 11, chosen to give the highest possible hand value not exceeding 21

Dealer Rules:
- The dealer has exactly two cards.
- One dealer card is visible and the other is hidden.
- The dealer does not draw any additional cards.
- The dealer's hand value is computed using the same Ace rule as the player.

Initial Observation:
At the start of the episode, you observe:

1. Dealer’s hand:
   - One visible card and one hidden card
   - Example: Dealer cards: [4, ?]

2. Your hand:
   - A list of cards currently held
   - Example: Your cards: [10, 6]

3. Deck:
   - A list of remaining cards indexed by position
   - Unknown card identities are hidden
   - Example:
     Deck: [0: ?, 1: ?, 2: ?, 3: ?, 4: ?, ...]

Available Actions:
At each step, you must choose exactly one of the following actions:

- Hit: draw a specific card from the deck
  hit <card_index>

- Stand: stop drawing cards and end the game
  stand

Action Output Format:
You must output your action in exactly one line, using the following format:

- Hit example:
  \\boxed{hit 9}

- Stand example:
  \\boxed{stand}

Important constraints:
- Output only the boxed action.
- Do not include explanations, reasoning, or additional text.
- Do not output multiple actions.

Now Begin:
Given the current observation, decide your next action and output it in the required format.
"""
        return BLACKJACK_ENV_RESET_PROMPT.strip() + "\n\nCurrent observation:\n" + obs_text

    @staticmethod
    def _card_to_str(card: Card) -> str:
        if card == 1:
            return "A"
        if card == 10:
            return "10"
        return str(card)