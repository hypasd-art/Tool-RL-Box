# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    response_mask, 
    cliprange, 
    cliprange_low,
    cliprange_high,
    clip_ratio_c,
    sft_prefix, 
    off_cliprange=None, 
    off_cliprange_low=None, 
    off_cliprange_high=None,
    off_clip_ratio_c=None,
    # target_probs=None,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        sft_prefix: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )
    # off-policy loss
    # compute off-policy probability
    
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl) # [bsz, l]
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    
    
    on_pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    on_pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        on_pg_losses1, on_pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses1).float(), response_mask)

    on_pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(on_pg_losses3, clip_pg_losses1)

    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, on_pg_losses3) * (advantages < 0).float(), response_mask
    )

    on_pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - sft_prefix) * response_mask)
    
    
    # compute off-policy loss
    off_ratio = torch.exp(log_prob) # [bsz, l]
    off_ratio = off_ratio / (off_ratio + 0.1)
    off_pg_losses1 = -advantages * off_ratio
        
    # clip off-policy ratio
    if off_cliprange is None:
        off_cliprange = cliprange
    if off_cliprange_low is None:
        off_cliprange_low = off_cliprange
    if off_cliprange_high is None:
        off_cliprange_high = off_cliprange

    off_ratio = torch.clamp(off_ratio, max=1 + off_cliprange_high)
    off_ratio = torch.clamp(off_ratio, min=1 - off_cliprange_low)
    off_pg_losses2 = -advantages * off_ratio  
    off_clip_pg_losses1 = torch.maximum(
        off_pg_losses1, off_pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    off_pg_clipfrac = verl_F.masked_mean(torch.gt(off_pg_losses2, off_pg_losses1).float(), response_mask)

    off_pg_losses3 = -advantages * off_clip_ratio_c
    off_clip_pg_losses2 = torch.min(off_pg_losses3, off_clip_pg_losses1)

    off_pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(off_clip_pg_losses1, off_pg_losses3) * (advantages < 0).float(), response_mask
    )

    off_pg_losses = torch.where(advantages < 0, off_clip_pg_losses2, off_clip_pg_losses1)

    

    
    off_pg_loss = verl_F.masked_mean(off_pg_losses, sft_prefix * response_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    
    sft_prefix = sft_prefix.float()
    pg_losses = off_pg_losses * sft_prefix + on_pg_losses * (1 - sft_prefix)
            
    off_policy_prob = verl_F.masked_mean(
        torch.exp(log_prob),
        sft_prefix * response_mask,
    )
    on_policy_prob = verl_F.masked_mean(
        torch.exp(old_log_prob),
        (1.0 - sft_prefix) * response_mask,
    )
    
        
    
    pg_loss = verl_F.masked_mean(pg_losses, response_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "pg_clipfrac_lower": pg_clipfrac_lower,
        "off_pg_clipfrac_lower": off_pg_clipfrac_lower,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,

    }

class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct") #  if self.config.use_sft_loss else None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(
                        batch=batch, max_token_len=max_token_len
                    )

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    

    def _forward_get_logits(self, micro_batch, temperature=1.0):
        """
        Forward pass without computing gradients.
        Returns:
            logits: (batch_size, response_len, vocab_size)
            probs:  (batch_size, response_len, vocab_size)
        """
        input_ids = micro_batch["input_ids"]
        decode = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # print(decode)
        
        batch_size, seq_len = input_ids.shape
        response_len = micro_batch["responses"].size(-1)
        # print(response_len)
        response_ids = micro_batch["responses"]
        response_mask = micro_batch["response_mask"]
        decode_response = self.tokenizer.decode(response_ids[0], skip_special_tokens=False)
        with open("./txt.log", "a") as f:
            f.write(decode + "\n-------------------"*100 + decode_response + "------\n"*100)
            f.write(str(list(response_mask[0])))
        attention_mask = micro_batch.get("attention_mask", None)
        position_ids = micro_batch.get("position_ids", None)
        multi_modal_inputs = {}


        with torch.no_grad():
            # Forward pass
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = output.logits[:, -response_len-1 : -1, :]   # Extract response tokens
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            

        return logits, probs

    

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",

            'sft_replace',
            'hint_assistant'
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # non_tensors_keys = ["extra_info", "sft_replace", "hint_assistant"] # add extra keys, 
        batch = data.select(batch_keys=select_keys).batch # , non_tensors_keys
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"] 
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        

        
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for current_original_indices, text_mb_td in zip(
                            textual_indices, rearranged_text_micro_batches_tds, strict=True
                        ):
                            current_mm_inputs_list = [
                                all_multi_modal_inputs_list[idx] for idx in current_original_indices
                            ]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                
                self.actor_optimizer.zero_grad()
                # # ====== SFT Training Stage ====== #
                # loss_fct = nn.CrossEntropyLoss(reduction="none")
                # # total_loss = torch.tensor(0).to(get_device_id())
                # if self.config.use_sft_loss:
                #     sft_steps = self.config.sft_train_config.sft_steps_per_update  # how many SFT steps per PPO update

                #     # Reload SFT dataloader on each update_policy (data changes)
                #     sft_dataloader = self.get_fresh_sft_dataloader(
                #         data_paths=self.config.sft_train_config.sft_train_paths,
                #         data_config=self.config.sft_train_config.sft_data_config,
                #         tokenizer=self.tokenizer,
                #     )

                #     sft_iter = iter(sft_dataloader)
                #     if sft_steps == -1:
                #         sft_steps = len(sft_dataloader)
                #     for idx in range(sft_steps):
                #         try:
                #             batch_sft = next(sft_iter)
                #         except StopIteration:
                #             break

                #         # -------------------------------
                #         # SFT loss (cross-entropy)
                #         # -------------------------------
                #         print(f"[SFT] device: {get_device_id()}")
                #         # Support all hardwares
                #         # if isinstance(batch_sft, DataProto):
                #         #     batch_sft = {**batch_sft.batch.to(get_device_id()), **batch_sft.non_tensor_batch}
                #         # elif isinstance(batch_sft, dict):
                #         #     for k, v in batch_sft.items():
                #         #         if isinstance(v, torch.Tensor):
                #         #             batch_sft[k] = v.to(get_device_id())
                #         #         elif k == "multi_modal_inputs" and v is not None:
                #         #             batch_sft[k] = [
                #         #                 {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                #         #             ]
                #         #         else:
                #         #             batch_sft[k] = v
                #         # else:
                #         #     batch_sft = batch_sft.to(get_device_id())  # actor device is cpu when using offload
                #         for key, value in batch_sft.items():
                #             batch_sft[key] = batch_sft[key].to(get_device_id()) if torch.is_tensor(batch_sft[key]) else batch_sft[key]
                #         dp_size = 1
                #         loss_mask = batch_sft.pop("loss_mask")[:, :-1].reshape(-1)
                #         valid_token_this_rank = torch.sum(loss_mask)
                #         position_ids = batch_sft["position_ids"]
                #         labels = batch_sft["input_ids"][:, 1:].contiguous()
                #         output = self.actor_module(
                #             input_ids=batch_sft["input_ids"],
                #             attention_mask=batch_sft["attention_mask"],
                #             position_ids=position_ids,
                #             use_cache=False,
                #         )
                #         logits = output.logits

                #         shift_logits = logits[..., :-1, :].contiguous()
                #         shift_labels = labels.contiguous()
                #         # Flatten the tokens
                #         shift_logits = shift_logits.view(-1, self.actor_module.config.vocab_size)
                #         shift_labels = shift_labels.view(-1)
                #         # Enable model parallelism
                #         shift_labels = shift_labels.to(shift_logits.device)
                #         loss = loss_fct(shift_logits, shift_labels)
                #         loss = loss * loss_mask.to(loss.device)
                #         sft_loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size / 8 * self.config.sft_train_config.sft_loss_weight

                #         # -------------------------------
                #         # SFT backward (same optimizer)
                #         # -------------------------------
                #         if idx % 8 == 0:
                #             self.actor_optimizer.zero_grad()
                #         '''self.actor_optimizer.zero_grad()'''
                #         print(f"sft loss: {sft_loss}")
                #         sft_loss.backward()
                #         count = 0
                #         for idx, (name, param) in enumerate(self.actor_module.named_parameters()):
                #             if 'layers.0' in name:  # filter params with "v_proj" in name
                #                 if count < 1:  # only print first two
                #                     if param.grad is not None:
                #                         # print parameter values
                #                         print(f"Parameter {name} {param.requires_grad}: {param.data[:3]}\nGradient for {name}: {param.grad[:3]}")
                #                         print("-" * 50)  # separator
                #                     else:
                #                         # print parameter values
                #                         print(f"Parameter {name} {param.requires_grad}: {param.data[:3]}\nGradient for {name} is None")
                #                     count += 1
                #         torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), 1.0)
                #         if idx % 8 == 0:
                #             print("start optim")
                #             self.actor_optimizer.step()
                #             # print(grad_norm)
                #         # metrics["actor/sft_loss"] = sft_loss.detach().item() # total_loss
                #         # clean
                #         # del sft_loss, shift_logits, shift_labels
                #         # torch.cuda.empty_cache()

                #         # total_loss += sft_loss.detach().item()

                        
                #     # sft_metric = self.run_sft()

                for data in micro_batches:
                    # self.actor_optimizer.zero_grad()
                    micro_batch_metrics = {}

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [
                                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                                ]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    response_mask = data["response_mask"]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        if self.config.policy_loss.sft_replace:
                            print("[actor] Using SFT-replace off-policy loss")
                            print(data.keys())
                            results = compute_token_on_off_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                sft_prefix=data["sft_replace"],
                            )
                            pg_loss = results["pg_loss"]
                            pg_clipfrac = results["on_pg_clipfrac"]
                            ppo_kl = results["ppo_kl"]
                            pg_clipfrac_lower = results["pg_clipfrac_lower"]
                            micro_batch_metrics["actor/off_policy_prob"] = results["off_policy_prob"].detach().item()
                            micro_batch_metrics["actor/on_policy_prob"] = results["on_policy_prob"].detach().item()
                            micro_batch_metrics["actor/on_pg_loss"] = results["on_pg_loss"].detach().item()
                            micro_batch_metrics["actor/off_pg_loss"] = results["off_pg_loss"].detach().item()
                        else:  
                            print("[actor] Using vanilla policy loss")
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=advantages,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    if self.config.trainer_test:
                        # -------------------------------------------------------------------------------
                        os.makedirs("./tmp", exist_ok=True)
                        logits, probs = self._forward_get_logits(micro_batch=data, temperature=1.0)
                        probs = probs # .cpu()

                        batch_size, response_len, vocab_size = logits.shape
                        response_tokens = data["responses"]  # [B, L]
                        

                        # Accumulate full logit gradient per token
                        token_logits_grad = torch.zeros(vocab_size, dtype=torch.float).to(probs.device) # defaultdict(lambda: torch.zeros(vocab_size, dtype=torch.float))
                        token_logits_grad_self = torch.zeros(vocab_size, dtype=torch.float).to(probs.device) # defaultdict(lambda: torch.zeros(vocab_size, dtype=torch.float))
                        token_logits_grad_other = torch.zeros(vocab_size, dtype=torch.float).to(probs.device) # defaultdict(lambda: torch.zeros(vocab_size, dtype=torch.float))


                        for b in range(batch_size):
                            for t in range(response_len):
                                if not response_mask[b, t]:
                                    continue
                                token_id = response_tokens[b, t].item()  # current token
                                adv = advantages[b, t].item()            # current token advantage
                                p = probs[b, t] # .cpu()                           # current token probability
                                p_vec = probs[b, t]  # vocab_size

                                # Build full gradient: dL/dz_k = adv * (1[k=x_t] - p_k)
                                grad_vec = -adv * p_vec.clone()
                                grad_vec[token_id] += adv

                                # Accumulate to token logit gradient
                                token_logits_grad += grad_vec / response_mask[b].sum() # response_len # [token_id]

                                grad_vec_self = adv - adv * p_vec[token_id]

                                # Accumulate to token logit gradient
                                token_logits_grad_self[token_id] += grad_vec_self / response_mask[b].sum() # response_len

                                grad_vec_other = -adv * p_vec.clone()
                                grad_vec_other[token_id] += adv * p_vec[token_id].clone()

                                # Accumulate to token logit gradient
                                token_logits_grad_other += grad_vec_other / response_mask[b].sum() # response_len # [token_id]

                        token_ids_to_check = [
                            self.tokenizer.convert_tokens_to_ids("{\""),
                            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                            self.tokenizer.convert_tokens_to_ids("<tool_call>")
                        ]
                        tokens_name = ["{\"", "<|im_end|>", "<tool_call>"]

                        for idx, tid in enumerate(token_ids_to_check):
                            grad_vec = token_logits_grad[tid]
                            micro_batch_metrics[f"actor/logits_{tokens_name[idx]}"] = grad_vec.item()
                            micro_batch_metrics[f"actor/self_logits_{tokens_name[idx]}"] = token_logits_grad_self[tid].item()
                            micro_batch_metrics[f"actor/other_logits_{tokens_name[idx]}"] = token_logits_grad_other[tid].item()
                            print(f"token: {tokens_name[idx]}; grad: {grad_vec}; self_grad: {token_logits_grad_self[tid]}, other_grad: {token_logits_grad_other[tid]}")
                        
                        all_probs = []
                        response_tokens_str = "<|im_start|>assistant\n<tool_call>\n"
                        response_tokens_ids = self.tokenizer.encode(response_tokens_str, add_special_tokens=False)
                        sub_len = len(response_tokens_ids)

                        # Accumulate key token probabilities
                        p_lbrace_results = []
                        p_im_end_results = []
                        p_tool_call_results = []

                        token_lbrace_id = self.tokenizer.convert_tokens_to_ids("{\"")
                        token_im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                        token_tool_call_id = self.tokenizer.convert_tokens_to_ids("<tool_call>")

                        for b in range(probs.shape[0]):
                            # Find substring position in sequence
                            start_idx = response_len + 1
                            seq = response_tokens[b].tolist()  # [L]
                            # Find substring start position
                            for i in range(len(seq) - sub_len + 1):
                                if seq[i:i+sub_len] == response_tokens_ids:
                                    start_idx = i + sub_len  # position of first token after substring
                                    break
                            else:
                                # Substring not found, skip this batch
                                continue

                            # Get probability of token after substring
                            if start_idx < probs.shape[1]:
                                last_probs = probs[b, start_idx, :]  # [V]
                                all_probs.append(probs[b, start_idx, :].cpu())
                                p_lbrace_results.append(last_probs[token_lbrace_id].item())
                                p_im_end_results.append(last_probs[token_im_end_id].item())
                                p_tool_call_results.append(last_probs[token_tool_call_id].item())

                        # Compute mean probability
                        micro_batch_metrics["actor/p_lbrace"] = sum(p_lbrace_results) / len(p_lbrace_results)
                        micro_batch_metrics["actor/p_im_end"] = sum(p_im_end_results) / len(p_im_end_results)
                        micro_batch_metrics["actor/p_tool_call"] = sum(p_tool_call_results) / len(p_tool_call_results)

                        print(f"p_lbrace: {micro_batch_metrics['actor/p_lbrace']}, "
                            f"p_im_end: {micro_batch_metrics['actor/p_im_end']}, "
                            f"p_tool_call: {micro_batch_metrics['actor/p_tool_call']}")
                        import pickle
                        from datetime import datetime
                        
                        save_dict = {
                            "token_logits_grad": token_logits_grad.cpu(),
                            "token_logits_grad_self": token_logits_grad_self.cpu(),
                            "token_logits_grad_other": token_logits_grad_other.cpu(),
                            "all_probs": all_probs
                        }
                        file_name = f"./tmp/micro_batch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                        with open(file_name, "wb") as f:
                            pickle.dump(save_dict, f)
                            print(f"save to "+ file_name)
                        
                    loss.backward()
                    

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
    
    def get_fresh_sft_dataloader(self, data_paths, data_config, tokenizer):
        """Get a fresh dataloader for SFT training"""
        dataset = self.create_sft_dataset(data_paths, data_config, tokenizer)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=data_config.train_batch_size,
            shuffle=data_config.shuffle,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader
    
    
    def create_sft_dataset(self, data_paths, data_config, tokenizer):
        """Create a dataset."""
        # build dataset
        # First check if a custom dataset class is specified
        if data_config.custom_cls.get("path", None):
            from verl.utils.import_utils import load_extern_type
            dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Then check if multi-turn dataset should be used
        elif data_config.get("multiturn", {}).get("enable", False):
            dataset_cls = MultiTurnSFTDataset
        # Default to single-turn dataset
        else:
            dataset_cls = SFTDataset

        # Create datasets based on the selected class
        dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
        return dataset
