"""Prepare and register simple GEM task datasets for train/val.

Mirrors the frozenlake data prep: generates task dictionaries and
registers them via DatasetRegistry so AgentTrainer can load them.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from rllm.data.dataset import DatasetRegistry  # type: ignore


def prepare_gem_data(
    train_size: int = 1024,
    test_size: int = 128,
    env_id: str = "game:GuessTheNumber-v0-hard",
    seed: int = 42,
) -> Tuple[object, object]:
    """Create and register train/val splits of GEM task specs.

    Args:
        train_size: number of training tasks.
        test_size: number of validation tasks.
        env_id: GEM environment id to target.

    Returns:
        (train_dataset, test_dataset) as registered DatasetRegistry entries.
    """

    rng = np.random.default_rng(seed)
    train_seeds = rng.integers(0, 1_000_000, size=train_size).tolist()
    test_seeds = rng.integers(0, 1_000_000, size=test_size).tolist()

    def task_fn(idx: int, task_seed: int) -> dict:
        return {
            "env_id": env_id,
            "seed": int(task_seed),
            "uid": f"{env_id}-{task_seed}",
            "data_source": env_id,  # Add data_source for metric grouping
        }

    train_data = [task_fn(i, s) for i, s in enumerate(train_seeds)]
    test_data = [task_fn(i, s) for i, s in enumerate(test_seeds)]

    # Temporarily override verl postprocessing to include data_source at top level
    # (verl extracts top-level fields into non_tensor_batch)
    original_postprocessing = DatasetRegistry.apply_verl_postprocessing

    def gem_verl_postprocessing(data: list[dict]) -> list[dict]:
        """Custom verl postprocessing that preserves data_source at top level."""
        processed_data = []
        for entry in data:
            processed_entry = {
                "prompt": [{"role": "user", "content": "placeholder"}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None,
                },
                "extra_info": entry,
            }
            # Preserve data_source at top level so verl can extract it
            if "data_source" in entry:
                processed_entry["data_source"] = entry["data_source"]
            processed_data.append(processed_entry)
        return processed_data

    DatasetRegistry.apply_verl_postprocessing = staticmethod(gem_verl_postprocessing)

    try:
        train_dataset = DatasetRegistry.register_dataset("gem_tasks", train_data, "train")
        test_dataset = DatasetRegistry.register_dataset("gem_tasks", test_data, "test")
    finally:
        # Restore original postprocessing
        DatasetRegistry.apply_verl_postprocessing = original_postprocessing

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_ds, test_ds = prepare_gem_data()
    print(f"Train dataset: {len(train_ds.get_data())} examples; sample: {train_ds.get_data()[0]}")
    print(f"Test dataset: {len(test_ds.get_data())} examples; sample: {test_ds.get_data()[0]}")

