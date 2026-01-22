# Explorer RL Agent Framework

Train a universal agentic LLM with multi-task reinforcement learning that can exploit context for better multi-turn decision-making.

This repo builds on two upstream projects tracked as git submodules:
- `rllm`: training framework ([`rllm`](https://github.com/rllm-org/rllm))
- `GEM`: environment suite ([`GEM`](https://github.com/axon-rl/gem))

## Contents
- Installation
- Keeping submodules up to date
- Training
  - Single-task training (multi-episode)
  - Multi-task training (single-episode vs multi-episode)
- Evaluation (OpenAI)
- Outputs and metrics

## Installation (conda env: `icx`)

### Create and activate the environment

```bash
conda create -n icx python=3.11 -y
conda activate icx
pip install uv
```

### Fetch submodules and install dependencies (editable)

```bash
git submodule update --init --recursive

cd third_party/rllm
uv pip install -e .[verl]

cd ../..
uv pip install -e third_party/gem
```

References: [`rllm`](https://github.com/rllm-org/rllm), [`GEM`](https://github.com/axon-rl/gem).

### Quick installation test

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

## Keeping submodules up to date

Pull this repo and sync submodules:

```bash
git pull origin main
git submodule update --init --recursive
```

## Training

### Single-task training (multi-episode)

The script `scripts/train_single_task_multi_episode.sh` uses several configurable variables at the top of the file. Edit these variables to customize your training run:

```bash
ENV_ID=game:GuessTheNumber-v0-hard
TOTAL_STEP_CAP=21
MAX_TURNS_PER_EPISODE=7
MODEL_PATH=Qwen/Qwen3-1.7B
```

Notes:
- `ENV_ID`: GEM environment identifier. Format: `game:EnvironmentName-v0-difficulty`
  - Example: `game:GuessTheNumber-v0-hard`
  - Custom environment included:
    - `game:Minesweeper-v0-only-reveal`: Minesweeper that only requires revealing all non-mine cells to win (flags optional)
- `TOTAL_STEP_CAP`: maximum total steps allowed across all episodes in a single trajectory
  - This value is used for both `rllm.env.env_args.total_step_cap` and `rllm.agent.max_steps` to keep them synchronized
- `MAX_TURNS_PER_EPISODE`: maximum number of turns allowed per episode
- `MODEL_PATH`: HuggingFace model identifier or local path to the model

Run single-task training:

```bash
bash scripts/train_single_task_multi_episode.sh
```

Enable reflection (Hydra override; append to the command):

```bash
bash scripts/train_single_task_multi_episode.sh +rllm.env.env_args.enable_reflection=True
```

### Multi-task training

The framework supports training on multiple GEM tasks simultaneously, with each task having its own `max_turns_per_episode` and `total_step_cap` configuration. You can also specify different tasks for training and validation.

#### Training modes

| Mode | Training Script | Training Env | Validation Env | Use Case |
|------|------------------|--------------|----------------|----------|
| **Multi-episode** | `scripts/train_multi_task_multi_episode.sh` | `MultiEpisodeEnv` | `MultiEpisodeEnv` | Model gets multiple attempts per task during both training and validation |
| **Single-episode** | `scripts/train_multi_task_single_episode.sh` | `SingleEpisodeEnv` | `MultiEpisodeEnv` | Model trains on single attempts but validates with multiple attempts |

#### Configuration file format

Create a YAML configuration file with separate `train_tasks` and `val_tasks` sections.

Configuration fields:
- `env_id`: GEM environment identifier (required)
- `max_turns_per_episode`: maximum number of turns allowed per episode for this task (required)
- `total_step_cap`: maximum total steps across all episodes for this task
  - required for multi-episode
  - optional for single-episode training tasks
- `inner_env_class`: inner environment adapter class to use (required)
- `train_size`: number of training examples to generate (optional, default: 512) - only for `train_tasks`
- `test_size`: number of validation examples to generate (optional, default: 64) - only for `val_tasks`

Key features:
- Independent task configuration (each task can have different `max_turns_per_episode` and `total_step_cap`)
- Separate train/val tasks
- Per-task metrics (using `data_source` as the task identifier)
  - Training metrics: `traj/{env_id}/{metric}_mean/min/max`
  - Validation metrics: `val/{env_id}/{metric}`

#### Running multi-episode training

```bash
# Uses configs/multi_task_multi_episode_config.yaml by default
bash scripts/train_multi_task_multi_episode.sh

# Or specify a custom config
TASKS_CONFIG=configs/my_config.yaml bash scripts/train_multi_task_multi_episode.sh
```

#### Running single-episode training

```bash
# Uses configs/multi_task_single_episode_config.yaml by default
bash scripts/train_multi_task_single_episode.sh

# Or specify a custom config
TASKS_CONFIG=configs/my_config.yaml bash scripts/train_multi_task_single_episode.sh
```

Single-episode config example (`configs/multi_task_single_episode_config.yaml`):

```yaml
train_tasks:
  - env_id: "game:Minesweeper-v0-only-reveal"
    max_turns_per_episode: 5
    train_size: 512
    inner_env_class: "envs.gem_env_adapter.GEMEnvAdapter"

val_tasks:
  # Validation uses MultiEpisodeEnv, so total_step_cap is needed
  - env_id: "game:Mastermind-v0-hard"
    max_turns_per_episode: 25
    total_step_cap: 75  # 3 episodes * 25 turns each
    test_size: 128
    inner_env_class: "envs.gem_env_adapter.GEMEnvAdapter"
```

## Evaluation (OpenAI)

Run OpenAI evaluation via the helper script:

```bash
# Default: single-episode
bash scripts/run_eval_openai.sh

# Multi-episode
ENV_MODE=multi bash scripts/run_eval_openai.sh
```

Key parameters (via env vars or CLI overrides):
- `CONFIG`: task config YAML (default: `configs/eval_config.yaml`)
- `ENV_MODE`: `multi` or `single` (default: `single`)
- `MODEL`: OpenAI model name (default: `gpt-4o-mini`)
- `N_PARALLEL`: number of parallel envs (default: `256`)
- `TEMPERATURE` / `TOP_P`: sampling controls (defaults: `0.6` / `1`)
- `MAX_RESPONSE_LENGTH`: response length cap (default: `16384`)
- `TRAJECTORY_TIMEOUT`: per-trajectory timeout seconds (default: `6000`)
- `N_ROLLOUTS`: rollouts per task for pass@k (default: `1`)

## Outputs and metrics

Evaluation outputs are written into `results/` by default and chat completions are also logged (see `--log-chat-completions` in `scripts/run_eval_openai.sh`).

Logged metrics:
- `episode/success_rate`: 1.0 if episode succeeded, 0.0 otherwise
- `episode/episode_length`: number of steps taken
- `episode/truncated`: 1.0 if trajectory was terminated early by execution engine
