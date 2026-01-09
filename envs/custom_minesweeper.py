"""Custom Minesweeper environment with only-reveal success condition.

This module provides a MinesweeperEnv that only requires revealing all non-mine
cells (flags are not required for success), without modifying third-party GEM code.
"""

from __future__ import annotations

from typing import Any

try:
    from gem.envs.game_env.minesweeper import MinesweeperEnv as BaseMinesweeperEnv
except ImportError as exc:
    raise ImportError(
        "The 'gem' package is required for OnlyRevealMinesweeperEnv. "
        "Install it via `pip install gem-llm` or ensure it is on PYTHONPATH."
    ) from exc


class OnlyRevealMinesweeperEnv(BaseMinesweeperEnv):
    """Minesweeper variant that only requires revealing all non-mine cells.

    Flags are not required - you only need to reveal all safe cells.
    """

    def _is_solved(self) -> bool:
        """Check if all non-mine cells are revealed."""
        return all(
            self.grid[r][c] == -1 or self.revealed[r][c]
            for r in range(self.rows)
            for c in range(self.cols)
        )

