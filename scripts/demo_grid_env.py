#!/usr/bin/env python
"""Quick demo runner for the GridEnvAdapter with a fixed action sequence.

Usage:
    python scripts/demo_grid_env.py --seed 44 --n 4 --m 4 --actions "up" "right"

If no actions are provided, a default sequence is used. The script prints the
observation text (including the rules prompt on reset), rewards, and done flags
so you can see a full trajectory.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Allow running the script directly from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.grid_env_adapter import GridEnvAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo GridEnvAdapter rollout.")
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for deterministic grid resets.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of rows.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=5,
        help="Number of columns.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Max turns per episode.",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=["up", "right", "down", "left", "right", "up"],
        help="Action sequence, e.g., 'up' 'right' 'down'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = GridEnvAdapter(
        n=args.n,
        m=args.m,
        max_turns=args.max_turns,
        seed=args.seed,
    )

    obs, info = env.reset(seed=args.seed)
    print("=== Reset ===")
    print(obs)
    # print(f"info: {info}")
    print()

    for step_idx, action in enumerate(args.actions, start=1):
        print(f"--- Step {step_idx} | action='{action}' ---")
        obs, reward, done, info = env.step(action)
        print(obs)
        print(f"reward={reward}, done={done}, info={info}")
        print()
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
