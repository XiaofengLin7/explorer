"""Prepare and register simple GEM task datasets for train/val.

Mirrors the frozenlake data prep: generates task dictionaries and
registers them via DatasetRegistry so AgentTrainer can load them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
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


def prepare_multi_task_gem_data(
    tasks_config_path: Optional[str] = None,
    tasks_config: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
) -> Tuple[object, object]:
    """Create and register train/val splits for multiple GEM tasks.

    Each task can have its own max_turns_per_episode and total_step_cap configuration.
    Tasks are merged into a single dataset for training.

    Args:
        tasks_config_path: Path to YAML config file containing task configurations.
            Expected format:
                tasks:
                  - env_id: "game:TaskName-v0"
                    max_turns_per_episode: 7
                    total_step_cap: 21
                    train_size: 512
                    test_size: 64
        tasks_config: Optional list of task configurations (alternative to config file).
            Each dict should contain: env_id, max_turns_per_episode, total_step_cap,
            and optionally train_size, test_size.
        seed: Random seed for generating task seeds.

    Returns:
        (train_dataset, test_dataset) as registered DatasetRegistry entries.
        Each task dict in the dataset includes:
            - env_id: GEM environment identifier
            - seed: Task seed
            - uid: Unique identifier
            - data_source: env_id (for metric grouping)
            - max_turns_per_episode: Per-task max turns
            - total_step_cap: Per-task step cap

    Raises:
        ValueError: If neither tasks_config_path nor tasks_config is provided.
        FileNotFoundError: If tasks_config_path is provided but file doesn't exist.
    """
    if tasks_config is None:
        if tasks_config_path is None:
            raise ValueError("Either tasks_config_path or tasks_config must be provided")
        config_path = Path(tasks_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {tasks_config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        tasks_config = config.get("tasks", [])
        if not tasks_config:
            raise ValueError(f"No tasks found in config file: {tasks_config_path}")

    rng = np.random.default_rng(seed)
    all_train_data: List[Dict[str, Any]] = []
    all_test_data: List[Dict[str, Any]] = []

    for task_cfg in tasks_config:
        env_id = task_cfg["env_id"]
        max_turns_per_episode = task_cfg.get("max_turns_per_episode")
        total_step_cap = task_cfg.get("total_step_cap")
        train_size = task_cfg.get("train_size", 512)
        test_size = task_cfg.get("test_size", 64)

        # Generate seeds for this task
        train_seeds = rng.integers(0, 1_000_000, size=train_size).tolist()
        test_seeds = rng.integers(0, 1_000_000, size=test_size).tolist()

        def task_fn(idx: int, task_seed: int, is_train: bool) -> dict:
            """Create a task dictionary with per-task configuration."""
            task_dict: Dict[str, Any] = {
                "env_id": env_id,
                "seed": int(task_seed),
                "uid": f"{env_id}-{task_seed}",
                "data_source": env_id,  # For metric grouping per task
            }
            # Include per-task configuration in the task dict
            # These will be extracted by MultiEpisodeEnv.from_dict
            if max_turns_per_episode is not None:
                task_dict["max_turns_per_episode"] = max_turns_per_episode
            if total_step_cap is not None:
                task_dict["total_step_cap"] = total_step_cap
            return task_dict

        # Generate train and test data for this task
        train_data = [task_fn(i, s, True) for i, s in enumerate(train_seeds)]
        test_data = [task_fn(i, s, False) for i, s in enumerate(test_seeds)]

        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

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
        train_dataset = DatasetRegistry.register_dataset("gem_tasks", all_train_data, "train")
        test_dataset = DatasetRegistry.register_dataset("gem_tasks", all_test_data, "test")
    finally:
        # Restore original postprocessing
        DatasetRegistry.apply_verl_postprocessing = original_postprocessing

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test single-task (backward compatibility)
    train_ds, test_ds = prepare_gem_data()
    print(f"Single-task - Train: {len(train_ds.get_data())} examples")
    print(f"Single-task - Test: {len(test_ds.get_data())} examples")
    print(f"Sample: {train_ds.get_data()[0]}")

    # Test multi-task
    config_path = Path(__file__).parent.parent / "configs" / "multi_task_gem_config.yaml"
    if config_path.exists():
        train_ds_multi, test_ds_multi = prepare_multi_task_gem_data(tasks_config_path=str(config_path))
        print(f"\nMulti-task - Train: {len(train_ds_multi.get_data())} examples")
        print(f"Multi-task - Test: {len(test_ds_multi.get_data())} examples")
        print(f"Sample: {train_ds_multi.get_data()[0]}")

