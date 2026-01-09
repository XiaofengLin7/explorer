"""Auto-registration of the only-reveal Minesweeper environment.

This module automatically registers the only-reveal Minesweeper environment
with GEM's registry when imported, so it can be used with `gem.make()` and
`GEMEnvAdapter` by simply specifying the env_id.
"""

try:
    import gem.envs  # noqa: F401  # Populate registries
    from gem.envs.registration import register
except ImportError as exc:
    raise ImportError(
        "The 'gem' package is required. "
        "Install it via `pip install gem-llm` or ensure it is on PYTHONPATH."
    ) from exc

from envs.custom_minesweeper import OnlyRevealMinesweeperEnv

# Auto-register the environment when this module is imported
register(
    "game:Minesweeper-v0-only-reveal",
    "envs.custom_minesweeper:OnlyRevealMinesweeperEnv",
    rows=5,
    cols=5,
    num_mines=5,
    max_turns=25,
)

