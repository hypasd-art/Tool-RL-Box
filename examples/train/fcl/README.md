# FCL Training Recipes

Function Call Leaderboard — train Qwen2.5-1.5B-Instruct and Qwen3-1.7B for multi-turn tool use with GRPO.

## Installation

```bash
conda create --name verl-tool-env python=3.10
conda activate verl-tool-env
pip install -e verl
pip install -e ".[vllm]"
pip install "flash-attn<2.8.0" --no-build-isolation
```

## GPU Requirements

| Model | GPUs |
|-------|------|
| Qwen2.5-1.5B-Instruct | 2 |
| Qwen3-1.7B | 2 |

## Workflow

```
SFT (stage 1) → RL from SFT / RL from scratch → Iterative RL+SFT evolution
```

1. **Run SFT first** — produces the base checkpoint for downstream RL
2. **Run RL experiments** — can be launched in parallel once SFT completes
3. **Iterative evolution** — alternates RL training, process-reward scoring, and SFT distillation across turns

## Preprocessing

Before running any script, update:
- Model paths (`model.partial_pretrain`, `actor_rollout_ref.model.path`)
- `WANDB_API_KEY` environment variable

## Training Scripts

### SFT

| Script | Data | Model |
|--------|------|-------|
| `sft_1.5b_fcl.sh` | FCL | Qwen2.5-1.5B-Instruct |
| `sft_1.5b_ace.sh` | ToolACE | Qwen2.5-1.5B-Instruct |
| `sft_1.7b_fcl.sh` | FCL | Qwen3-1.7B |
| `sft_1.7b_ace.sh` | ToolACE | Qwen3-1.7B |

### RL from Pretrained

| Script | Description |
|--------|-------------|
| `train_1.5b.sh` | GRPO from Qwen2.5-1.5B-Instruct base |
| `train_1.7b.sh` | GRPO from Qwen3-1.7B base |

### RL from SFT

| Script | SFT Source |
|--------|------------|
| `sft_rl_1.5b.sh` | FCL SFT (1.5B) |
| `sft_rl_1.5b_ace.sh` | ToolACE SFT (1.5B) |
| `sft_rl_1.7b.sh` | FCL SFT (1.7B) |
| `sft_rl_1.7b_ace.sh` | ToolACE SFT (1.7B) |
| `sft_rl_1.7b_no_think.sh` | FCL SFT (1.7B, no thinking) |

### Iterative RL+SFT Evolution

| Script | Description |
|--------|-------------|
| `train_evolution_sft1.5b.sh` | 3-turn evolution: RL → SFT → repeat (1.5B) |
| `train_evolution_sft1.5b_difflr.sh` | Same, with differential LR per turn (1.5B) |
| `train_evolution_sft1.7b_difflr.sh` | 3-turn evolution with diff LR (1.7B) |
| `train_evolution_pr1.5b.sh` | Evolution with process-reward data generation (1.5B) |
| `train_evolution_pr1.7b.sh` | Evolution with process-reward data generation (1.7B) |

### Hint-Assisted Training

| Script | Description |
|--------|-------------|
| `hint_1.5b_system.sh` | Oracle hints injected in system prompt (1.5B) |
| `hint_1.7b_system.sh` | Oracle hints injected in system prompt (1.7B) |

### SFT-Augmented RL

| Script | Description |
|--------|-------------|
| `sft_replace_1.5b.sh` | RL with SFT-based action replacement (1.5B) |
| `sft_replace_1.7b.sh` | RL with SFT-based action replacement (1.7B) |

### Evaluation

| Script | Description |
|--------|-------------|
| `evaluation.sh` | Validation-only inference over model checkpoints |

## BFCL Evaluation

Create a separate environment to avoid conflicts:

```bash
conda create -n BFCL python=3.10
conda activate BFCL
cd benchmarks/berkeley-function-call-leaderboard
pip install -e .
pip install -e .[oss_eval_vllm]
```

Replace `CHECKPOINT_DIR` with your checkpoint path:

```bash
MODEL="Qwen/Qwen2.5-1.5B-Instruct-FC"
TEST_CATEGORY="multi_turn_miss_param_training,multi_turn_long_context_training,\
multi_turn_base_training,multi_turn_miss_func_training,\
multi_turn_miss_param_testing,multi_turn_long_context_testing,\
multi_turn_base_testing,multi_turn_miss_func_testing"

CHECKPOINT_DIR=(
    "./checkpoints/fcl_sft/fc_sft_qwen2.5-1.5b-instruct_batch_128_learning_rate_1e-5_epoch_3"
)

for dir in "${CHECKPOINT_DIR[@]}"; do
    for path in "$dir"/*; do
        echo "Testing $path..."
        bfcl generate \
            --model "$MODEL" \
            --test-category "$TEST_CATEGORY" \
            --backend vllm \
            --num-gpus 1 \
            --gpu-memory-utilization 0.6 \
            --temperature 0 \
            --local-model-path "$path" \
            --result-dir "test/$(basename "$path")" \
            --allow-overwrite
        bfcl evaluate \
            --model "$MODEL" \
            --test-category "$TEST_CATEGORY" \
            --result-dir "test/$(basename "$path")" \
            --score-dir "score/$(basename "$path")"
        echo "Finished testing $path."
    done
done
```
