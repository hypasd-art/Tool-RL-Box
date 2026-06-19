#!/bin/bash

export WANDB_API_KEY=${WANDB_API_KEY:-}

set -x
# model_name="${CHECKPOINT_DIR:-./checkpoints}/ace_sft/fc_sft_qwen2.5-1.5b-instruct_batch_128_learning_rate_1e-5_epoch_3/global_step_231"
model_name="${HF_MODEL_DIR:-./models}/Qwen2.5-1.5B-Instruct" 
train_data="./data/fcl/training_data/train.parquet"
val_data="./data/fcl/training_data/test.parquet"

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')

for turn in {1..3}
do
    echo "Turn ${turn}"
    rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
    n_gpus_per_node=2
    n_nodes=1
    n=8
    total_epochs=100
    total_training_steps=100 # 50 
    batch_size=60
    ppo_mini_batch_size=20
    max_prompt_length=8192
    max_action_length=2048
    max_response_length=8192
    max_obs_length=1024
    temperature=1.0
    top_p=1.0
    enable_agent=True # enable agent for tool use
    strategy="fsdp"
    max_turns=30
    kl_loss_coef=0.0001
    kl_coef=0
    entropy_coeff=0
    kl_loss_type=low_var_kl
    lr=1e-6
    reward_manager=fcl_eval
    ppo_micro_batch_size_per_gpu=1
    log_prob_micro_batch_size_per_gpu=8
    tensor_model_parallel_size=1
    action_stop_tokens=""
    gpu_memory_utilization=0.6 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
    do_offload=True # control actor's fsdp.[param|optimizer]_offload and actor_rollout_ref.rollout.fsdp.[param|optimizer]_offload; if gpu_memory_utilization is set to > 0.6, then do_offload should be set to True otherwise it will cause OOM
    use_dynamic_bsz=True # faster
    ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
    fsdp_size=-1
    additional_eos_token_ids=[151645] # <|im_end|> token id
    mask_observations=True # mask observations for kl loss and gradient descent
    enable_mtrl=True # enable multi-turn training
    mtrl_role="user"
    if [ "$turn" -eq "1" ]; then
        total_training_steps=50
    elif [ "$turn" -eq "2" ]; then
        total_training_steps=100
        previ_training_steps=50
    elif [ "$turn" -eq "3" ]; then
        total_training_steps=150
        previ_training_steps=100
    fi
    
    run_name_postfix="debug"
    if [ "$enable_agent" = "True" ]; then
        run_name="${reward_manager}-grpo-${strategy}-agent-${model_pretty_name}-${rl_alg}-step${total_training_steps}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}${run_name_postfix}_sft_turn_${turn}"
    else
        run_name="${reward_manager}-grpo-${strategy}-${model_pretty_name}-${rl_alg}-step${total_training_steps}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}${run_name_postfix}_sft_turn_${turn}"
    fi
    pre_run_name="${reward_manager}-grpo-${strategy}-agent-${model_pretty_name}-${rl_alg}-step${previ_training_steps}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}${run_name_postfix}_sft_turn_$((turn-1))"
    echo "---------------------------------------${run_name}---------------------------------------"
    export VERL_RUN_ID=$run_name
    export NCCL_DEBUG=INFO
    export VLLM_USE_V1=1
    lr_warmup_steps=5
    rollout_mode='async'
    if [ "$turn" -gt "1" ]; then 
        n_id=$(ls -1 ${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${pre_run_name}_sft/fc_sft_$((turn-1)) | grep -oP '\d+' | sort -V | tail -n 1) 
        model_name="${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${pre_run_name}_sft/fc_sft_$((turn-1))/global_step_${n_id}" 
        lr_warmup_steps=5
    fi

    
    echo "------------------loading from ${model_name}------------------" # ()
    

    # temp file for action tokens as verl cannot pass special strs as params
    action_stop_tokens_file="$(pwd)$(mktemp)"
    mkdir -p $(dirname $action_stop_tokens_file)
    echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
    echo "action_stop_tokens_file=$action_stop_tokens_file"


    host=$(hostname -i | awk '{print $1}')
    port=$(shuf -i 30000-31000 -n 1)
    tool_server_url=http://$host:$port/get_observation
    python -m verl_tool.servers.serve --host $host --port $port --tool_type "fcl_interface" --workers_per_tool 16 &
    server_pid=$!

    echo "Server (pid=$server_pid) started at $tool_server_url"

    if [ "$turn" -eq "1" ]; then
        echo "Finish Turn 1"
    else
        PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
            algorithm.adv_estimator=$rl_alg \
            data.train_files=$train_data \
            data.val_files=$val_data \
            data.train_batch_size=$batch_size \
            data.val_batch_size=2048 \
            data.max_prompt_length=$max_prompt_length \
            data.max_response_length=$max_response_length \
            data.truncation='right' \
            reward_model.reward_manager=$reward_manager \
            reward_model.launch_reward_fn_async=True \
            actor_rollout_ref.model.path=$model_name \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.optim.lr=$lr \
            actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.model.trust_remote_code=True \
            actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
            actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
            actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.strategy=$strategy \
            actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
            actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
            actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
            actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
            actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
            actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
            actor_rollout_ref.agent.enable_agent=$enable_agent \
            actor_rollout_ref.agent.tool_server_url=$tool_server_url \
            actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
            actor_rollout_ref.agent.max_response_length=$max_response_length \
            actor_rollout_ref.agent.max_start_length=$max_prompt_length \
            actor_rollout_ref.agent.max_obs_length=$max_obs_length \
            actor_rollout_ref.agent.max_turns=$max_turns \
            actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
            actor_rollout_ref.agent.mask_observations=$mask_observations \
            actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
            actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
            actor_rollout_ref.agent.mtrl_role=$mtrl_role \
            actor_rollout_ref.agent.max_action_length=$max_action_length \
            actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
            actor_rollout_ref.rollout.enforce_eager=False \
            actor_rollout_ref.rollout.free_cache_engine=True \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
            actor_rollout_ref.rollout.temperature=$temperature \
            actor_rollout_ref.rollout.top_p=$top_p \
            actor_rollout_ref.rollout.top_k=-1 \
            actor_rollout_ref.rollout.n=$n \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
            actor_rollout_ref.rollout.max_num_seqs=512 \
            actor_rollout_ref.rollout.mode=$rollout_mode \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
            actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
            actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
            critic.optim.lr=1e-5 \
            critic.strategy=$strategy \
            critic.model.path=$model_name \
            critic.model.fsdp_config.fsdp_size=$fsdp_size \
            critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
            critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
            algorithm.kl_ctrl.kl_coef=$kl_coef \
            trainer.logger=['console','wandb'] \
            trainer.project_name=$reward_manager \
            trainer.experiment_name=$run_name \
            trainer.val_before_train=True \
            trainer.default_hdfs_dir=null \
            trainer.n_gpus_per_node=$n_gpus_per_node \
            trainer.nnodes=$n_nodes \
            trainer.resume_mode="disable" \
            +trainer.remove_previous_ckpt_in_save=False \
            trainer.save_freq=99999 \
            trainer.test_freq=5 \
            trainer.total_epochs=$total_epochs \
            trainer.default_local_dir="${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${run_name}" \
            trainer.total_training_steps=$total_training_steps \

        pkill -P -9 $server_pid
        kill -9 $server_pid
    fi
    
    echo "start merging"
    last_checkpoint_id=$(< ${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${run_name}/latest_checkpointed_iteration.txt)

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${run_name}/global_step_${last_checkpoint_id}/actor \
        --target_dir ${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${run_name}_merged
    python process_error_idx.py --input_dir ./verl_step_records/${run_name} --train_dir ./data/fcl_sft/training_data/train.parquet --min_step $((total_training_steps - 10)) --max_step ${total_training_steps}
    if [ "$turn" -ne "3" ]; then
        nproc_per_node=2
        save_path="${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${run_name}_sft/fc_sft_${turn}" 

        echo "------------------------------------------start SFT------------------------------------------"
        
        torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
            -m verl.trainer.fsdp_sft_trainer \
            data.train_batch_size=128 \
            data.train_files=./verl_step_records/${run_name}/train_sft.parquet \
            data.val_files=./data/fcl_sft_perturn/training_data/test.parquet \
            data.prompt_key=prompt \
            data.response_key=response \
            data.custom_cls.path=./verl/verl/utils/dataset/sft_dataset_wo_apply_template.py \
            data.custom_cls.name=SFTDataset_wo_tp \
            optim.lr=1e-6 \
            data.max_length=8192 \
            data.truncation=left \
            data.micro_batch_size_per_gpu=2 \
            model.partial_pretrain=${CHECKPOINT_DIR:-./checkpoints}/${reward_manager}/${run_name}_merged \
            trainer.default_local_dir=$save_path \
            trainer.project_name=fc-sft-${turn} \
            trainer.experiment_name=fc-sft-${model_pretty_name}_${turn} \
            trainer.logger=['console'] \
            trainer.total_epochs=2 \
            trainer.save_freq=500 \
            optim.warmup_steps_ratio=0.3 $@
    fi

done