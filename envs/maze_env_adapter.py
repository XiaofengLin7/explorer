"""
Maze environment adapter for rLLM training stack.

This adapter implements a maze navigation environment that integrates with the
rLLM training framework, providing a standardized interface for reinforcement
learning agents to learn navigation and planning skills in discrete grid worlds.

Refactor:
- Maze generation logic is moved into MazeGenerator class.
- MazeEnvAdapter now REQUIRES env_kwargs to include: shapes, max_turns
  (otherwise raises ValueError).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import random
import re

import numpy as np
from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore


class MazeGenerator:
    """Standalone maze generator.

    Convention:
      - maze: 0 = path, 1 = wall, -1 = goal
      - init_position: (x, y)
    """

    def __init__(self, shapes: List[Tuple[int, int]]):
        if not isinstance(shapes, list) or len(shapes) == 0:
            raise ValueError("shapes must be a non-empty list of (width, height) tuples")
        if not all(isinstance(t, tuple) and len(t) == 2 for t in shapes):
            raise ValueError("shapes must be a list of (width, height) tuples, e.g. [(5,5),(6,6)]")
        self.shapes = shapes

    def generate(self, seed: int) -> Tuple[np.ndarray, Tuple[int, int], int]:
        """Generate a maze with many branching paths using improved algorithm."""
        random.seed(seed)
        np.random.seed(seed)

        maze_width, maze_height = random.choice(self.shapes)
        maze = np.ones((maze_height, maze_width), dtype=int)

        # Choose random starting position (not on border)
        start_x = random.randint(1, maze_height - 2)
        start_y = random.randint(1, maze_width - 2)
        init_position = (start_x, start_y)

        # Phase 1: main network
        self._create_branched_network(maze, start_x, start_y)

        # Phase 2: add branches
        self._add_branch_extensions(maze, maze_height, maze_width)

        # Phase 3: add loops
        self._create_strategic_loops(maze, maze_height, maze_width, seed)

        # Phase 4: ensure connectivity
        self._ensure_connectivity(maze, init_position)

        # Place goal at farthest reachable position from start
        goal_position, shortest_path_len = self._find_farthest_position(maze, init_position)
        maze[goal_position[0], goal_position[1]] = -1

        return maze, init_position, shortest_path_len

    def _create_branched_network(self, maze: np.ndarray, start_x: int, start_y: int) -> None:
        maze[start_x, start_y] = 0

        active_fronts = [(start_x, start_y)]
        visited = {(start_x, start_y)}

        while active_fronts:
            front_idx = random.randint(0, len(active_fronts) - 1)
            current_x, current_y = active_fronts[front_idx]

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)

            expanded = False
            branches_created = 0
            max_branches_per_node = 1 if random.random() < 0.7 else 2

            for dx, dy in directions:
                if branches_created >= max_branches_per_node:
                    break

                new_x, new_y = current_x + dx, current_y + dy

                if (
                    1 <= new_x < maze.shape[0] - 1
                    and 1 <= new_y < maze.shape[1] - 1
                    and (new_x, new_y) not in visited
                ):
                    open_neighbors = self._count_open_neighbors(maze, new_x, new_y)
                    if open_neighbors <= 1:
                        maze[new_x, new_y] = 0
                        visited.add((new_x, new_y))
                        active_fronts.append((new_x, new_y))
                        expanded = True
                        branches_created += 1

            if not expanded:
                active_fronts.pop(front_idx)

    def _add_branch_extensions(self, maze: np.ndarray, height: int, width: int) -> None:
        dead_ends: List[Tuple[int, int]] = []
        for x in range(1, height - 1):
            for y in range(1, width - 1):
                if maze[x, y] == 0:
                    if self._count_open_neighbors(maze, x, y) == 1:
                        dead_ends.append((x, y))

        num_extensions = min(len(dead_ends), max(2, len(dead_ends) // 3))
        random.shuffle(dead_ends)

        for i in range(num_extensions):
            x, y = dead_ends[i]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (
                    1 <= new_x < height - 1
                    and 1 <= new_y < width - 1
                    and maze[new_x, new_y] == 1
                ):
                    if self._count_open_neighbors(maze, new_x, new_y) <= 1:
                        maze[new_x, new_y] = 0
                        if random.random() < 0.6:
                            self._extend_branch_randomly(
                                maze,
                                new_x,
                                new_y,
                                height,
                                width,
                                max_length=random.randint(2, 5),
                            )
                        break

    def _extend_branch_randomly(
        self,
        maze: np.ndarray,
        start_x: int,
        start_y: int,
        height: int,
        width: int,
        max_length: int,
    ) -> None:
        current_x, current_y = start_x, start_y

        for _ in range(max_length):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)

            extended = False
            for dx, dy in directions:
                new_x, new_y = current_x + dx, current_y + dy

                if (
                    1 <= new_x < height - 1
                    and 1 <= new_y < width - 1
                    and maze[new_x, new_y] == 1
                    and self._count_open_neighbors(maze, new_x, new_y) <= 1
                ):
                    maze[new_x, new_y] = 0
                    current_x, current_y = new_x, new_y
                    extended = True
                    break

            if not extended:
                break

    def _create_strategic_loops(self, maze: np.ndarray, height: int, width: int, seed: int) -> None:
        random.seed(seed + 42)

        num_loops = random.randint(2, max(3, (height * width) // 30))

        for _ in range(num_loops):
            for _attempt in range(100):
                x = random.randint(1, height - 2)
                y = random.randint(1, width - 2)

                if maze[x, y] == 1:
                    path_neighbors = []
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and maze[nx, ny] == 0:
                            path_neighbors.append((nx, ny))

                    if len(path_neighbors) == 2:
                        maze[x, y] = 0
                        break

    def _count_open_neighbors(self, maze: np.ndarray, x: int, y: int) -> int:
        count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                count += 1
        return count

    def _ensure_connectivity(self, maze: np.ndarray, start_pos: Tuple[int, int]) -> None:
        reachable = self._get_reachable_positions(maze, start_pos)

        all_paths = set()
        for x in range(1, maze.shape[0] - 1):
            for y in range(1, maze.shape[1] - 1):
                if maze[x, y] == 0:
                    all_paths.add((x, y))

        unreachable = all_paths - reachable
        for pos in unreachable:
            self._connect_to_reachable(maze, pos, reachable)

    def _get_reachable_positions(self, maze: np.ndarray, start: Tuple[int, int]) -> set:
        from collections import deque

        queue = deque([start])
        visited = {start}

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    0 <= new_x < maze.shape[0]
                    and 0 <= new_y < maze.shape[1]
                    and maze[new_x, new_y] == 0
                    and (new_x, new_y) not in visited
                ):
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))

        return visited

    def _connect_to_reachable(self, maze: np.ndarray, pos: Tuple[int, int], reachable: set) -> None:
        x, y = pos
        min_distance = float("inf")
        best_connection: Optional[Tuple[int, int]] = None

        for rx, ry in reachable:
            distance = abs(x - rx) + abs(y - ry)
            if distance < min_distance:
                min_distance = distance
                best_connection = (rx, ry)

        if best_connection is None:
            return

        cx, cy = best_connection
        current_x, current_y = x, y

        while current_x != cx:
            current_x += 1 if current_x < cx else -1
            if 0 <= current_x < maze.shape[0] and 0 <= current_y < maze.shape[1]:
                maze[current_x, current_y] = 0

        while current_y != cy:
            current_y += 1 if current_y < cy else -1
            if 0 <= current_x < maze.shape[0] and 0 <= current_y < maze.shape[1]:
                maze[current_x, current_y] = 0

    def _find_farthest_position(
        self, maze: np.ndarray, start: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], int]:
        from collections import deque

        queue = deque([(start, 0)])
        visited = {start}
        farthest_pos = start
        max_distance = 0

        while queue:
            (x, y), distance = queue.popleft()
            if distance > max_distance:
                max_distance = distance
                farthest_pos = (x, y)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if (
                    0 <= new_x < maze.shape[0]
                    and 0 <= new_y < maze.shape[1]
                    and maze[new_x, new_y] == 0
                    and (new_x, new_y) not in visited
                ):
                    visited.add((new_x, new_y))
                    queue.append(((new_x, new_y), distance + 1))

        return farthest_pos, max_distance


class MazeEnvAdapter(BaseEnv):
    """Adapter for maze navigation environments in the rLLM framework."""

    ACTIONS = {"up", "down", "left", "right"}
    DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def __init__(self, env_id: str, env_kwargs: Optional[Dict[str, Any]] = None):
        self.env_id = env_id

        # ---- REQUIRED env_kwargs ----
        if env_kwargs is None:
            raise ValueError("env_kwargs must be provided and must include: shapes, max_turns")

        required_keys = ("shapes", "max_turns")
        missing = [k for k in required_keys if k not in env_kwargs]
        if missing:
            raise ValueError(
                f"Missing required env_kwargs keys: {missing}. "
                f"Required: {list(required_keys)}"
            )

        self.shapes = env_kwargs.get("shapes", [])
        self.max_turns = env_kwargs.get("max_turns", -1)
        self.shortest_path_min_length = env_kwargs.get("shortest_path_min_length", 8)
        self.shortest_path_max_length = env_kwargs.get("shortest_path_max_length", 8)

        # Validate types/values to prevent silent bugs
        # normalize: list[list[int]] -> list[tuple[int,int]]
        try:
            self.shapes = [tuple(x) for x in self.shapes]
        except Exception:
            raise ValueError(
                "env_kwargs['shapes'] must be a list of [width, height], e.g. [[5,5],[6,6]]"
            )
        if not isinstance(self.shapes, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in self.shapes):
            raise ValueError("env_kwargs['shapes'] must be a list of (width, height) tuples, e.g. [(5,5),(6,6)]")
        if not isinstance(self.max_turns, int) or self.max_turns <= 0:
            raise ValueError("env_kwargs['max_turns'] must be a positive int")
        if not isinstance(self.shortest_path_min_length, int) or self.shortest_path_min_length <= 0:
            raise ValueError("env_kwargs['shortest_path_min_length'] must be a positive int")
        if not isinstance(self.shortest_path_max_length, int) or self.shortest_path_max_length <= 0:
            raise ValueError("env_kwargs['shortest_path_max_length'] must be a positive int")
        # Generator (optional injection; if not provided, create from shapes)
        self.generator: MazeGenerator = env_kwargs.get("maze_generator") or MazeGenerator(self.shapes)

        self.map: Optional[np.ndarray] = None
        self.init_position: Optional[Tuple[int, int]] = None
        self.current_position: Optional[Tuple[int, int]] = None
        self.shortest_path_len: Optional[int] = None
        self.current_turn = 0
        self.achieve_goal = False

    def _get_cell_type(self, x: int, y: int) -> str:
        if self.map is None:
            return "wall"

        if x < 0 or x >= self.map.shape[0] or y < 0 or y >= self.map.shape[1]:
            return "wall"

        cell_value = self.map[x, y]
        if cell_value == 1:
            return "wall"
        elif cell_value == -1:
            return "goal"
        else:
            return "path"

    def _generate_observation(self) -> str:
        if self.current_position is None:
            return "Environment not initialized. Please call reset() first."

        x, y = self.current_position
        up = self._get_cell_type(x - 1, y)
        down = self._get_cell_type(x + 1, y)
        left = self._get_cell_type(x, y - 1)
        right = self._get_cell_type(x, y + 1)

        if up == "wall" and down == "wall" and left == "wall" and right == "wall":
            if self.map is not None:
                for xx in range(self.map.shape[0]):
                    for yy in range(self.map.shape[1]):
                        print(self.map[xx][yy])
                    print("\n")
            raise ValueError("Agent is completely surrounded by walls. Invalid state.")

        return (
            f"Now you are at position ({x}, {y}) in the maze. "
            f"Around you, up leads to {up}, down leads to {down}, left leads to {left}, and right leads to {right}. "
        )

    def get_state_id(self) -> Tuple[int, int]:
        """Return a hashable identifier for the current maze state.

        This is intentionally lightweight and stable across steps so wrappers
        (e.g., `MultiEpisodeEnv`) can track unique visited states over an entire
        trajectory. For the maze, the agent position uniquely identifies the
        state for visitation counting.

        Returns:
            A tuple (x, y) representing the agent's current position.

        Raises:
            ValueError: If the environment has not been reset yet.
        """
        if self.current_position is None:
            raise ValueError("Environment not initialized. Please call reset() first.")
        return self.current_position

    def reset(self, seed: int | None = None, task: dict | None = None) -> tuple[str, Dict[str, Any]]:
        if seed is None:
            raise ValueError("Seed must be provided.")

        max_tries = 1000
        for _ in range(max_tries):
            self.map, self.init_position, self.shortest_path_len = self.generator.generate(seed)
            if self.shortest_path_min_length <= self.shortest_path_len <= self.shortest_path_max_length:
                break
            seed += 1
        else:
            raise ValueError(f"Failed to generate a maze within shortest_path_min_length ({self.shortest_path_min_length}) and shortest_path_max_length ({self.shortest_path_max_length}) after many attempts.")

        self.current_turn = 0
        self.current_position = self.init_position
        self.achieve_goal = False

        
        initial_instruction = (
            "You are a maze-solving agent. Your goal is to navigate from the START position "
            "to the GOAL position in the fewest turns possible. You are at the START position. "
            + self._generate_observation()
            + "\nOutput your next move from up/down/left/right within \\boxed{}."
        )

        obs = initial_instruction
        info = {
            "turn": self.current_turn,
            "max_turns": self.max_turns,
            "terminated": False,
            "truncated": False,
            "state_id": self.get_state_id(),
        }
        return obs, info

    def _parse_action(self, raw_text: str) -> str:
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        match = re.search(boxed_pattern, raw_text, re.IGNORECASE)

        if match:
            return match.group(1).strip().lower()

        return raw_text.strip().lower()

    def _execute_action(self, action: str) -> tuple[str, float]:
        if self.current_position is None or self.map is None:
            raise ValueError("Environment not initialized. Please call reset() first.")

        if action not in self.DELTA:
            return "Invalid action. Please try again.", 0.0

        x, y = self.current_position
        dx, dy = self.DELTA[action]
        new_x, new_y = x + dx, y + dy

        if new_x < 0 or new_x >= self.map.shape[0] or new_y < 0 or new_y >= self.map.shape[1]:
            return "You hit the wall. Please choose another direction.", 0.0

        if self.map[new_x, new_y] == 1:
            return "You hit the wall. Please choose another direction.", 0.0

        self.current_position = (new_x, new_y)

        if self.map[new_x, new_y] == -1:
            self.achieve_goal = True
            return "Congratulations! You arrived at the goal! Let's play again. ", 1.0


        return (
            f"You move to {action}. "
            + self._generate_observation()
            + "\nOutput your next move from up/down/left/right within \\boxed{}.",
            0.0,
        )

    def step(self, action: Any) -> tuple[str, float, bool, Dict[str, Any]]:
        raw_action = action.action if isinstance(action, Action) else action
        parsed_action = self._parse_action(str(raw_action))

        self.current_turn += 1
        observation_text, reward = self._execute_action(parsed_action)

        terminated = False
        if self.achieve_goal:
            terminated = True

        if not terminated and self.current_turn >= self.max_turns:
            terminated = True
            observation_text = f"\nYou have reached the maximum number of turns ({self.max_turns}). Let's play again. "

        obs = observation_text
        info = {
            "turn": self.current_turn,
            "max_turns": self.max_turns,
            "is_correct": self.achieve_goal,
            "reward_metadata": None,
            "terminated": terminated,
            "truncated": False,
            "raw_reward": float(reward),
            "state_id": self.get_state_id(),
        }

        return obs, float(reward), terminated, info

    def close(self) -> None:
        pass

    @staticmethod
    def from_dict(info: dict) -> "MazeEnvAdapter":
        env_id = info["env_id"]
        env_kwargs = info.get("env_kwargs", {})
        return MazeEnvAdapter(env_id=env_id, env_kwargs=env_kwargs)
