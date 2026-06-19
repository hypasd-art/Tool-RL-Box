import os
from pathlib import Path
from datetime import datetime
import asyncio
from collections import defaultdict
from typing import Dict, Any, List
from typing import Tuple, Dict, Any, Optional, Union, List

from verl import DataProto
from verl.workers.reward_manager import register  # type: ignore
from verl_tool.servers.tools.multi_turn_eval.multi_turn_checker import multi_turn_checker

import torch
import random
import regex as re
import json
import numpy as np
from collections import defaultdict
# import pandas as pd
# from openai import OpenAI

def compute_score(
    sequences_list: List[str],
    ground_truth: Dict[str, Any],
    test_category: str = "",
    test_entry: dict = {},
    model_name: str = "model",
    trajectory_id: str = "",
    left_tag: int = 0,
    right_tag: int = 0,
) -> float:
    """
    The scoring function for Search-R1 style exact match (EM).

    Args:
        sequences_list: the list of solution texts
        ground_truth: the ground truth dict with 'target' field
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    result = multi_turn_checker(
        multi_turn_model_result_list_decoded=sequences_list, # []
        multi_turn_ground_truth_list=ground_truth, # [[.get('target', ground_truth)]]
        test_entry=test_entry,
        test_category=test_category,
        model_name=model_name,
        trajectory_id=trajectory_id,
    )
    # do_print = random.randint(1, 64) == 1

    # if do_print:
    #     print("--------------------------------")
    #     # ground truth
    #     print(f"Golden answers: {ground_truth}") # .get('target', ground_truth)

    #     # raw output of the model
    #     print(f"Solution string: {sequences_list}")

    # if result["valid"]:
    #     return 1
    # else:
    #     return 0
    if result["valid"]:
        return 1, result["error_turn"]
    else:
        return 0, result["error_turn"]


def compute_score_v2(
    sequences_list: List[str],
    ground_truth: Dict[str, Any],
    test_category: str = "",
    test_entry: dict = {},
    model_name: str = "model",
    trajectory_id: str = "",
    left_tag: int = 0,
    right_tag: int = 0,
) -> float:
    """
    The scoring function for Search-R1 style exact match (EM).

    Args:
        sequences_list: the list of solution texts
        ground_truth: the ground truth dict with 'target' field
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    result = multi_turn_checker(
        multi_turn_model_result_list_decoded=sequences_list, # []
        multi_turn_ground_truth_list=ground_truth, # [[.get('target', ground_truth)]]
        test_entry=test_entry,
        test_category=test_category,
        model_name=model_name,
        trajectory_id=trajectory_id,
    )
    # do_print = random.randint(1, 64) == 1

    # if do_print:
    #     print("--------------------------------")
    #     # ground truth
    #     print(f"Golden answers: {ground_truth}") # .get('target', ground_truth)

    #     # raw output of the model
    #     print(f"Solution string: {sequences_list}")

    if result["valid"]:
        num = abs(left_tag - right_tag)
        return 1 - num / (left_tag + right_tag), 1
    else:
        num = abs(left_tag - right_tag)
        return 0 - num / (left_tag + right_tag), 0

# MODEL_NAME = os.getenv("MODEL_NAME")   
# API_KEY = os.getenv("API_KEY") 
# base_url= os.getenv("BASE_URL") 
# client = OpenAI(api_key=API_KEY, base_url=base_url)

# prompt_template = """
# You are an expert in intelligent agent interaction analysis.

# The following data records the **entire multi-turn interaction process** between a model and tools, along with the corresponding ground-truth sequences. Your task is to conduct a **comprehensive and layered evaluation**.

# You need to conduct Turn-by-Turn Detailed Analysis, focusing on *fine-grained reasoning* for each turn.

# ### Please structure your output as follows:

# For each turn, we suggest you to analyze:
# 1. **User Intent Understanding, the correct Behavior and Logic and Success Points of the action**  
# 2. **Mistake Classification (Fine-grained) and Root Cause Analysis if error (Why)**  
# 3. **Improvement Plan**  
# - Give corrected reasoning and ideal tool call sequence for every turns and the format should be in <tool_call></tool_call>.
# - Explain how it aligns with the ground truth and the logic of the action.

# ---

# ### Available Tools:
# {available_tools}


# ### Model Interaction Log (All Turns):
# {formatted_summary}

# ---

# Notes:
# - If a turn includes "Missing Function" or "Missing Parameter", the model should have requested clarification. And no tool calls should be made at this turn.
# - Output should be comprehensive and actionable.
# - All tool calls must follow the format below (no markdown or ```json blocks):
# <tool_call>
# {{"name": "<function_name>", "arguments": {{<args-json-object>}}}}
# </tool_call>
# """   

# def to_tool_call_format(s: str) -> str:
#     """
#     Convert a string like find(path='.',name='test_document.txt')
#     into <tool_call> JSON format.
#     """
#     match = re.match(r"(\w+)\((.*)\)", s.strip())
#     if not match:
#         raise ValueError("Invalid format. Expected: func(a='x', b='y')")
    
#     func_name, args_str = match.groups()
    
#     args = {}
#     for k, v in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str):
#         args[k] = v

#     tool_call = {
#         "name": func_name,
#         "arguments": args
#     }

#     json_str = json.dumps(tool_call, ensure_ascii=False)
    
#     formatted = f"<tool_call>\n{json_str}\n</tool_call>" # \\
#     return formatted

# def process_turn_together(interaction_json: str, ground_truth: list = None, question: str=None, data_source: str = None, available_tools: str = None):
#     formatted_summary = f"User question at Turn 1: {question[0][0]['content']}\n"
#     turn_num = 0
#     num = 0
#     for i, step in enumerate(interaction_json):
#         if "<tool_response>" in step.get('obs'):
#             formatted_summary += (
#                 f" - Step {num+1}:\n"
#                 f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
#                 f"   - obs: {step.get('obs', '').strip()}\n"
#             )
#             num += 1
#         else:
#             formatted_summary += (
#                 f" - Step {num+1}:\n"
#                 f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
                
#             )
#             ground_truth_summary = ""
#             step_ground_truth = ground_truth[turn_num]
#             if len(step_ground_truth) == 0:
#                 if "miss_func" in data_source:
#                     ground_truth_summary += f"Missing Function, No tool calls at this turn\n"
#                 elif "miss_param" in data_source:
#                     ground_truth_summary += f"Missing Parameter, No tool calls at this turn\n"
#                 else:
#                     raise Exception
#             for v in step_ground_truth:
#                 ground_truth_summary += to_tool_call_format(v) + "\n"
#             formatted_summary += f"\n*ground truth action sequence for this turn*:\n{ground_truth_summary}"
#             num = 0
#             turn_num += 1
#             if i < len(interaction_json) - 1:
#                 formatted_summary += f"- User question at Turn {turn_num+1}: {step.get('obs', '').strip()}\n"
    
#     prompt = prompt_template.format(
#         available_tools=available_tools,
#         formatted_summary=formatted_summary,
#     )
        
#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are an intelligent agent behavior analysis assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7,
#         max_tokens=8192
#     )
#     return prompt, response.choices[0].message.content.strip().replace("<think>", "").replace("</think>", "")

@register("fcl_eval_pr")
class FCLPRInterfaceEvaluatorRM:
    """
    Reward manager that proxies scoring to FCL evaluators.

    For each sample, it:
    - Decodes the model response
    - Locates the original FCL task JSON via extra_info["task"]
    - Runs the task's evaluators (fcl_interface.evaluator)
    - Scores = (#passed / #total); writes to the last response token

    Args (via reward_kwargs):
    - configs_root: Root folder to resolve relative task paths. Defaults to
      "verl-tool/benchmarks/FCL/verl_tool/servers/tools/fcl_interface/benchmark/configs"
    """

    name = "fcl_eval_pr"

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        configs_root: str | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.compute_score = compute_score
        self.step = None

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        has_tool_call = False
        if "</tool_call>" in action: # action.endswith("</tool_call>")
            # Extract the JSON part from the action
            json_part = re.findall(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
            if json_part:
                action_list = []
                for json_str in json_part:
                    action = json_str.strip()
                    # Parse the JSON string
                    try:
                        action = json.loads(action)
                        assert "name" in action, "Action JSON must contain 'name' field"
                        assert "arguments" in action, "Action JSON must contain 'arguments' field"
                        action_name = action["name"]
                        action = f"{action_name}({', '.join([f'{k}={repr(v)}' for k, v in action['arguments'].items()])})"
                        action_list.append(action)
                    except:
                        continue
                        # action_list.append("Error JSON format: {}".format(action)) # Invalid 
                        # return "", False
                
                has_tool_call = True
        if not has_tool_call:
            return ""
        
        return action_list
    
    


    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards for FC style responses."""
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / f"torl-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)

        # check the last step index
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']
 
        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        scores = [{} for _ in range(len(data))]
        acc = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        to_save_records = []

        for i in range(len(data)):
            data_item = data[i]
            sequences_list = []
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            if data_source == "process_knowledge":
                traj_ids = data_item.non_tensor_batch["traj_ids"]
                if traj_ids.endswith("_0") or traj_ids.endswith("_1"):
                    # try:
                    #     assert self.tokenizer.decode(data_item.batch['responses'], skip_special_tokens=False) == data_item.non_tensor_batch.get("extra_info")["ground_truth"][0][0], f"Response does not match ground truth for {traj_ids}, {data_item.batch['responses']} != {data_item.non_tensor_batch.get('ground_truth')}"
                    # except:
                    #     print(f"Response does not match ground truth for {traj_ids}, {self.tokenizer.decode(data_item.batch['responses'], skip_special_tokens=False)} != {data_item.non_tensor_batch.get('extra_info')['ground_truth'][0][0]}")
                    scores[i] = 1.0
                    to_save_records.append({
                        'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                        'traj_ids': data_item.non_tensor_batch["traj_ids"],
                        'data_source': data_source,
                        "prompt": self.tokenizer.decode(data_item.batch['prompts'], skip_special_tokens=False),
                        "response": self.tokenizer.decode(data_item.batch['responses'], skip_special_tokens=False),
                        'ground_truth': data_item.non_tensor_batch.get('ground_truth', []),
                        'score': 1.0,
                        'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
                        'extra_info': data_item.non_tensor_batch.get('extra_info', None),
                    })
                    continue
                else:
                    scores[i] = 0.0
                    to_save_records.append({
                        'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                        'traj_ids': data_item.non_tensor_batch["traj_ids"],
                        'data_source': data_source,
                        "prompt": self.tokenizer.decode(data_item.batch['prompts'], skip_special_tokens=False),
                        "response": self.tokenizer.decode(data_item.batch['responses'], skip_special_tokens=False),
                        'ground_truth': data_item.non_tensor_batch.get('ground_truth', []),
                        'score': 0.0,
                        'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
                        'extra_info': data_item.non_tensor_batch.get('extra_info', None),
                    })
                    continue

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # print(data_item.non_tensor_batch.keys())
            # print(data_item.non_tensor_batch['reward_model'])
            # print(data_item.non_tensor_batch["tool_interact_info"])
            messages = data_item.non_tensor_batch["tool_interact_info"] # rollout_messages
            turn_actions = []
            left_tag = 0
            right_tag = 0
            for message in messages:
                left_tag += message["action"].count("<tool_call>")
                right_tag += message["action"].count("</tool_call>")

                actions = self.parse_action(message["action"])
                if actions == "":
                    sequences_list.append(turn_actions)
                    turn_actions = []
                else:
                    turn_actions.append(actions)
            

            # Get ground truth
            # Fallback to direct ground truth or golden_answers
            # with open("./fcl_action.txt", "a") as f:
            #     f.write(json.dumps(data_item.non_tensor_batch, indent=2) + "\n----------------------------------------------------------------------------\n")
            ground_truth = data_item.non_tensor_batch["reward_model"].get('ground_truth', 
                            data_item.non_tensor_batch.get('golden_answers', []))
            # print(f"sequences_list: {sequences_list}\nground_truth: {ground_truth}")
            test_entry = {
                "initial_config": json.loads(data_item.non_tensor_batch['reward_model']['initial_config']),
                "involved_classes": data_item.non_tensor_batch['reward_model']['involved_classes'],
                "id": data_item.non_tensor_batch['reward_model']['id'],
            }
            # Compute score
            score, error_turn = compute_score(
                sequences_list=sequences_list, 
                ground_truth=ground_truth, 
                test_entry = test_entry,
                trajectory_id = data_item.non_tensor_batch["traj_ids"],
                left_tag = left_tag,
                right_tag = right_tag,
            )

            # score, acc_score = compute_score_v2(
            #     sequences_list=sequences_list, 
            #     ground_truth=ground_truth, 
            #     test_entry = test_entry,
            #     trajectory_id = data_item.non_tensor_batch["traj_ids"],
            #     left_tag = left_tag,
            #     right_tag = right_tag,
            # )

            # TODO: check if logic is correct
            # update this score to the scores
            scores[i] = score # {"score": score}
            # acc[i] = acc_score

            # reward_tensor[i, valid_response_length - 1] = score
            """if data_item.non_tensor_batch['extra_info']['id'] == "multi_turn_base_174":
                print(f"reward\nresponse: {self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False)}\n{data[i].non_tensor_batch.get('tool_interact_info', None)}")"""
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            train_data = []
            # if (data_item.non_tensor_batch["traj_ids"].split("_")[-1] == '0' or data_item.non_tensor_batch["traj_ids"].split("_")[-1] == '1') and self.num_examine != 1:
            #     interaction_json = data[i].non_tensor_batch.get('tool_interact_info', None)
            #     ground_truth = ground_truth
            #     question = data_item.non_tensor_batch["extra_info"]["question"]
            #     match = re.search("<tools>\n(.*?)\n</tools>", data[i].non_tensor_batch["prompt"], re.DOTALL)
            #     available_tools_n = ""
            #     if match:
            #         available_tools = match.group(1).strip()
            #         tools = available_tools.split("\n")
            #         for t in tools:
            #             t_json = json.loads(t)
            #             available_tools_n += f"{t_json['name']};" 
            #     prompt_result, analysis_result = process_turn_together(interaction_json, ground_truth, question, data_source, available_tools_n) 
            #     train_data.append({
            #         "prompt": prompt_result,
            #         "response": analysis_result,
            #     })
            
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            # Save the records
            to_save_records.append({
                'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                'traj_ids': data_item.non_tensor_batch["traj_ids"],
                'data_source': data_source,
                "prompt": self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False),
                "response": self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False),
                'ground_truth': ground_truth,
                'score': score,
                'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
                'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
                'extra_info': data_item.non_tensor_batch.get('extra_info', None),
                'error_turn': error_turn,
                # 'prompt_analysis': prompt_result,
                # 'analysis_result': analysis_result,
            })
            if "turns_stats" in data_item.non_tensor_batch:
                to_save_records[i]['num_turn'] = data[i].non_tensor_batch["turns_stats"]
                to_save_records[i]['num_valid_action'] = data[i].non_tensor_batch["valid_action_stats"]
                to_save_records[i]['is_done'] = not data[i].non_tensor_batch["active_mask"]
        # df_row = pd.DataFrame(train_data)
        # print(df_row.head())
        # print(f"✅ Data aggregation complete, {len(df_row)} samples total")
        # # Save processed DataFrame
        # train_output_file_path = self.record_dir + f"/train-{self.step}.parquet" # output_path.replace(".json", "_train.parquet") #  + 
        # df_row.to_parquet(train_output_file_path, index=False)
        # df_row.to_parquet(self.record_dir / f"train-current.parquet", index=False)
        # print(f"Saved {len(df_row)} processed rows to {train_output_file_path}")

        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"{self.name}-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"{self.name}-step-{self.step}.json"
            self.step += 1
            # if temp_file.exists():
            #     # with open(temp_file, "r") as f:
            #     #     existing_records = json.load(f)
            #     # existing_records.extend(to_save_records)
            #     # with open(temp_file, "w") as f:
            #     #     json.dump(existing_records, f, indent=4)
            #     with open(temp_file, "w") as f:
            #         json.dump(to_save_records, f, indent=4)
            # else:
            with open(temp_file, "w") as f:
                json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")

        for i, score in enumerate(scores):
            # Add the score to the reward tensor
            length_i = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum().item()
            reward_tensor[i, length_i - 1] = score
            
        # reward_extra_info["acc"] = acc

        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


