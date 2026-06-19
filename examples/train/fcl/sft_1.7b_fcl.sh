#!/bin/bash

export WANDB_API_KEY=${WANDB_API_KEY:-}

set -x

nproc_per_node=2
epoch=3
train_batch_size=128
lr=1e-5
save_path="${CHECKPOINT_DIR:-./checkpoints}/fcl_sft/fc_sft_qwen3-1.7b_batch_${train_batch_size}_learning_rate_${lr}_epoch_${epoch}"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/fcl_sft_perturn/training_data/train.parquet \
    data.val_files=./data/fcl_sft_perturn/training_data/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=$train_batch_size \
    data.custom_cls.path=./verl/verl/utils/dataset/sft_dataset_wo_apply_template.py \
    data.custom_cls.name=SFTDataset_wo_tp \
    optim.lr=$lr \
    data.max_length=8192 \
    data.truncation=left \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=${HF_MODEL_DIR:-./models}/Qwen3-1.7B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=fc-sft \
    trainer.experiment_name=fc-sft-qwen-3-1.7b \
    trainer.logger=['console'] \
    trainer.save_freq=23 \
    trainer.total_epochs=$epoch $@ \



nproc_per_node=2
epoch=3
train_batch_size=128
lr=1e-6
save_path="${CHECKPOINT_DIR:-./checkpoints}/fcl_sft/fc_sft_qwen3-1.7b-instruct_batch_${train_batch_size}_learning_rate_${lr}_epoch_${epoch}"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/fcl_sft_perturn/training_data/train.parquet \
    data.val_files=./data/fcl_sft_perturn/training_data/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=$train_batch_size \
    data.custom_cls.path=./verl/verl/utils/dataset/sft_dataset_wo_apply_template.py \
    data.custom_cls.name=SFTDataset_wo_tp \
    optim.lr=$lr \
    data.max_length=8192 \
    data.truncation=left \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=${HF_MODEL_DIR:-./models}/Qwen3-1.7B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=fc-sft \
    trainer.experiment_name=fc-sft-qwen-3-1.7b \
    trainer.logger=['console'] \
    trainer.save_freq=23 \
    trainer.total_epochs=$epoch $@ \
