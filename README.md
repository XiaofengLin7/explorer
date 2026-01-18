# Explorer RL Agent Framework

Goal: train a universal agentic LLM with multi-task reinforcement learning that can exploit context for better multi-turn decision-making. We build on `rllm` for training and `GEM` for environments, both tracked as git submodules for easy updates.

## Installation (conda env: `icx`)
```bash

# Create / activate env (python 3.11 recommended)
conda create -n icx python=3.11 -y
conda activate icx
pip install uv

# Fetch submodules
git submodule update --init --recursive

# Install rllm (editable to track upstream changes)
cd third_party/rllm
uv pip install -e .[verl]

# Install GEM (editable)
cd ../..
uv pip install -e third_party/gem
```

References: [`rllm`](https://github.com/rllm-org/rllm), [`GEM`](https://github.com/axon-rl/gem).

### Quick installation tests
Run inside the activated `icx` environment:
```bash
python - <<'PY'
import importlib
for pkg in ("rllm", "gem"):
    try:
        importlib.import_module(pkg)
        print(f"{pkg}: OK")
    except Exception as exc:
        print(f"{pkg}: FAILED -> {exc}")
PY
```
If both report `OK`, installations are healthy.

## Pull and update submodules
Initialize and keep the upstream repos synced:
```bash
git pull origin main && git submodule update --init --recursive
```

## Training Configuration

### Configuring Training Variables

The training script `scripts/train_single_task_multi_episode.sh` uses several configurable variables at the top of the file. Edit these variables to customize your training run:

```bash
ENV_ID=game:GuessTheNumber-v0-hard
TOTAL_STEP_CAP=21
MAX_TURNS_PER_EPISODE=7
MODEL_PATH=Qwen/Qwen3-1.7B
```

**Variable Descriptions:**

- **`ENV_ID`**: The GEM environment identifier. Format: `game:EnvironmentName-v0-difficulty`
  - Example: `game:GuessTheNumber-v0-hard`
  - **Custom Environments**: The framework includes a custom only-reveal Minesweeper variant:
    - `game:Minesweeper-v0-only-reveal`: Minesweeper variant that only requires revealing all non-mine cells to win (flags are optional, not required for success)
  
- **`TOTAL_STEP_CAP`**: Maximum total steps allowed across all episodes in a single trajectory
  - This value is automatically used for both `rllm.env.env_args.total_step_cap` and `rllm.agent.max_steps` to keep them synchronized

- **`MAX_TURNS_PER_EPISODE`**: Maximum number of turns allowed per individual episode

- **`MODEL_PATH`**: HuggingFace model identifier or local path to the model


**Running Single-Task Training:**

```bash
bash scripts/train_single_task_multi_episode.sh
```

**Enable Reflection:**

```bash
+rllm.env.env_args.enable_reflection=True
```

### Multi-Task Training

The framework supports training on multiple GEM tasks simultaneously, with each task having its own `max_turns_per_episode` and `total_step_cap` configuration. You can also specify different tasks for training and validation.

#### Training Modes

The framework provides two multi-task training modes:

| Mode | Training Script | Training Env | Validation Env | Use Case |
|------|-----------------|--------------|----------------|----------|
| **Multi-Episode** | `train_multi_task_multi_episode.sh` | MultiEpisodeEnv | MultiEpisodeEnv | Model gets multiple attempts per task during both training and validation |
| **Single-Episode** | `train_multi_task_single_episode.sh` | SingleEpisodeEnv | MultiEpisodeEnv | Model trains on single attempts but validates with multiple attempts |

#### Configuration File Format

Create a YAML configuration file with separate `train_tasks` and `val_tasks` sections.

**Configuration Fields:**

- **`env_id`**: GEM environment identifier (required)
- **`max_turns_per_episode`**: Maximum number of turns allowed per episode for this task (required)
- **`total_step_cap`**: Maximum total steps across all episodes for this task (required for multi-episode, optional for single-episode training tasks)
- **`inner_env_class`**: The inner environment adapter class to use (required)
- **`train_size`**: Number of training examples to generate (optional, default: 512) - only for `train_tasks`
- **`test_size`**: Number of validation examples to generate (optional, default: 64) - only for `val_tasks`

**Key Features:**

- **Independent Task Configuration**: Each task can have different `max_turns_per_episode` and `total_step_cap` values
- **Separate Train/Val Tasks**: You can train on some tasks and validate on different tasks
- **Per-Task Metrics**: Training and validation metrics are logged per task (using `data_source` as the task identifier)
  - Training metrics: `traj/{env_id}/{metric}_mean/min/max`
  - Validation metrics: `val/{env_id}/{metric}`

#### Running Multi-Episode Training

Multi-episode training gives the model multiple attempts (episodes) per task within each trajectory. Use this when you want the model to learn from repeated attempts.

```bash
# Uses configs/multi_task_multi_episode_config.yaml by default
bash scripts/train_multi_task_multi_episode.sh

# Or specify a custom config
TASKS_CONFIG=configs/my_config.yaml bash scripts/train_multi_task_multi_episode.sh
```

#### Running Single-Episode Training

Single-episode training trains on one attempt per task but validates with multiple attempts. This is useful when you want efficient training but thorough evaluation.

```bash
# Uses configs/multi_task_single_episode_config.yaml by default
bash scripts/train_multi_task_single_episode.sh

# Or specify a custom config
TASKS_CONFIG=configs/my_config.yaml bash scripts/train_multi_task_single_episode.sh
```

**Single-Episode Config Example** (`configs/multi_task_single_episode_config.yaml`):

```yaml
train_tasks:
  - env_id: "game:Minesweeper-v0-only-reveal"
    max_turns_per_episode: 5
    train_size: 512
    inner_env_class: 'envs.gem_env_adapter.GEMEnvAdapter'

val_tasks:
  # Validation uses MultiEpisodeEnv, so total_step_cap is needed
  - env_id: "game:Mastermind-v0-hard"
    max_turns_per_episode: 25
    total_step_cap: 75  # 3 episodes * 25 turns each
    test_size: 128
    inner_env_class: 'envs.gem_env_adapter.GEMEnvAdapter'
```

**Logged Metrics:**
- `episode/success_rate`: 1.0 if episode succeeded, 0.0 otherwise
- `episode/episode_length`: Number of steps taken
- `episode/truncated`: 1.0 if trajectory was terminated early by execution engine




