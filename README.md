# Agentic RL Integration

Tool-augmented reinforcement learning for function-calling agents, built on [verl-tool](https://github.com/volcengine/verl.git) and [veRL](https://github.com/volcengine/verl).

This project extends verl-tool with process-level reward modeling, iterative SFT+RL evolution, hint-assisted training, and structural format debugging tools for multi-turn tool-use agents.

## Project Structure

```
.
├── process_reward_pr.py       # Process-level reward: LLM-judge analysis & data augmentation
├── process_error_idx.py       # Training error analysis: filter wrong trajectories for SFT
├── process_format_error.py    # Tool-call format diagnostics & structural collapse visualization
├── compute.py                 # Token probability analysis utilities
├── main.py                    # Entry point
├── verl_tool/                 # Core library (agent, trainer, tool servers, rollout workers)
├── verl/                      # veRL submodule (RL training framework)
├── data/                      # Training/test datasets (FCL, ToolACE, hint variants)
├── examples/
│   ├── train/fcl/             # Training scripts (SFT, RL, evolution, hints, replace)
│   └── test/                  # Evaluation scripts
└── benchmarks/                # Evaluation harnesses (ACEbench)
```

## Installation

```bash
conda create --name agentic-rl-box python=3.10
conda activate agentic-rl-box
pip install -e verl
pip install -e ".[vllm]"
pip install "flash-attn<2.8.0" --no-build-isolation
```

## Quick Start

```bash
# Set your W&B key
export WANDB_API_KEY=your_key_here

# 1. SFT training (2 GPUs)
bash examples/train/fcl/sft_1.5b_fcl.sh

# 2. RL training (2 GPUs)
bash examples/train/fcl/train_1.5b.sh
```

## Training Recipes

See [`examples/train/README.md`](examples/train/README.md) and [`examples/train/fcl/README.md`](examples/train/fcl/README.md) for the full catalog of training scripts, including:

| Category | Description |
|----------|-------------|
| **SFT** | Supervised fine-tuning on FCL or ToolACE data |
| **RL (GRPO)** | RL from pretrained or SFT checkpoints |
| **Iterative RL+SFT** | Multi-turn evolution with process reward |
| **Hint-assisted** | Training with oracle tool hints |
| **SFT Replace** | RL with action replacement for reward refinement |

## Models & GPU Requirements

| Model | GPUs |
|-------|------|
| Qwen2.5-1.5B-Instruct | 2 |
| Qwen3-1.7B | 2 |

## Key Features

- **Process-level reward modeling** — LLM-judge analyzes multi-turn tool interactions, classifies errors, and generates augmentation data
- **Structural format diagnostics** — Monitor tool-call format health during training to detect and prevent format collapse
- **Iterative evolution** — RL training → reward model scoring → SFT distillation → next RL round
- **Multi-tool support** — FCL, ToolACE, and custom tool interfaces via verl_tool's pluggable server architecture

See [`examples/train/fcl/README.md`](examples/train/fcl/README.md) for the full evaluation script.

## License

MIT — see [LICENSE](LICENSE).
