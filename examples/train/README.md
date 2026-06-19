# Training Recipes

This directory contains shell scripts for training tool-augmented language models with reinforcement learning, implementing the methods described in [*"Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It"* (Hao et al., EMNLP 2025)](../../Yupu_emnlp__Copy_%20(3).pdf).

## Paper-to-Code Mapping

| Paper Method | Script / Module | Description |
|-------------|----------------|-------------|
| **SFT Supervision** (§4.4) | `fcl/sft_1.5b_fcl.sh`, `fcl/sft_1.7b_ace.sh`, … | SFT on FCL (BFCL-V3) or ToolACE data before RL |
| **SFT then RL** (§4.4) | `fcl/sft_rl_1.5b.sh`, `fcl/sft_rl_1.7b_ace.sh`, … | GRPO initialized from SFT checkpoint |
| **Vanilla RL (GRPO)** (§4.2) | `fcl/train_1.5b.sh`, `fcl/train_1.7b.sh` | Direct GRPO from pretrained model |
| **Off-Policy Supervision** (§4.4, OPS) | `fcl/sft_replace_1.5b.sh`, `verl_tool/llm_agent/manager.py` (L837) | Replace sampled responses with ground-truth actions; reweight with mixed on/off-policy advantages |
| **Hint-Based Guidance** (§4.4, HBG) | `fcl/hint_1.5b_system.sh`, `verl_tool/llm_agent/manager.py` (`hint_assistant`) | Prepend correct hints during sampling; remove hints before policy optimization |
| **Erroneous Trajectory Supervision** (§4.4, ETS) | `fcl/train_evolution_sft1.5b.sh`, `process_error_idx.py` | Collect failed RL trajectories, construct SFT data from ground-truth solutions, interleave SFT+RL |
| **Process Reflection Supervision** (§4.4, PRS) | `fcl/train_evolution_pr1.5b.sh`, `process_reward_pr.py` | LLM-judge generates textual reflections from intermediate trajectories; joint training with erroneous trajectory SFT |
| **Format Collapse Analysis** (§4.3) | `process_format_error.py` | Monitor tool-call structural states (healthy/polluted/collapsed) during training |

## Available Recipes

| Recipe | Description | Scripts |
|--------|-------------|---------|
| [FCL](./fcl/README.md) | Function Call Leaderboard — train a model to use tools via multi-turn RL (GRPO) with verifiable rewards | [fcl/](./fcl/) |

### Training Variants

All scripts are under [`fcl/`](./fcl/). Models: Qwen2.5-1.5B-Instruct (2 GPUs) and Qwen3-1.7B (2 GPUs).

| Category | Example Scripts | Description |
|----------|----------------|-------------|
| **SFT** | `sft_1.5b_fcl.sh`, `sft_1.7b_ace.sh`, … | Supervised fine-tuning on FCL or ToolACE data |
| **RL from base** | `train_1.5b.sh`, `train_1.7b.sh` | GRPO training from a pretrained checkpoint |
| **RL from SFT** | `sft_rl_1.5b.sh`, `sft_rl_1.7b_ace.sh`, … | GRPO initialized from an SFT-tuned checkpoint |
| **Iterative RL+SFT** | `train_evolution_sft1.5b.sh`, … | Multi-turn evolution: RL → reward model → SFT → repeat |
| **Tool Hints (HBG)** | `hint_1.5b_system.sh`, … | Training with oracle tool hints injected into prompts |
| **SFT Replace (OPS)** | `sft_replace_1.5b.sh`, … | RL with SFT-based action replacement for reward refinement |
| **Evaluation** | `evaluation.sh` | Validation-only inference over a list of model checkpoints |

See [`fcl/README.md`](./fcl/README.md) for detailed setup, environment configuration, and the full execution workflow.

## Quick Start

```bash
# 1. Install dependencies
conda create --name agentic-rl-box python=3.10
conda activate agentic-rl-box
pip install -e verl
pip install -e ".[vllm]"
pip install "flash-attn<2.8.0" --no-build-isolation

# 2. Set your WANDB key
export WANDB_API_KEY=your_key_here

# 3. Run SFT first
bash examples/train/fcl/sft_1.5b_fcl.sh

# 4. Then run RL
bash examples/train/fcl/train_1.5b.sh
```
