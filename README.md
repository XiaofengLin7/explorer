# Explorer RL Agent Framework

Goal: train a universal agentic LLM with multi-task reinforcement learning that can exploit context for better multi-turn decision-making. We build on `rllm` for training and `GEM` for environments, both tracked as git submodules for easy updates.

## Submodules
Initialize and keep the upstream repos synced:
```bash
git submodule update --init --recursive
git submodule update --remote --merge third_party/rllm
git submodule update --remote --merge third_party/gem
```


