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


