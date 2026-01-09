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

## Submodules
Initialize and keep the upstream repos synced:
```bash
git submodule update --init --recursive
git submodule update --remote --merge third_party/rllm
git submodule update --remote --merge third_party/gem
```

## Training Configuration

### Configuring Training Variables

The training script `scripts/train_gem_multi_episode_env.sh` uses several configurable variables at the top of the file. Edit these variables to customize your training run:

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

**Running Training:**

```bash
bash scripts/train_gem_multi_episode_env.sh
```



