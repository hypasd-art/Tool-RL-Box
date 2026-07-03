def get_first_user_message(msgs):
    for m in msgs:
        if m["role"] == "user":
            return m["content"].strip()
    # return None

import pandas as pd
import re

def normalize_question(text: str):

    return text
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct")

def build_question_key_from_prompt(prompt):
    q = get_first_user_message(prompt)
    return normalize_question(q) if q else None

def build_question_key_from_messages(messages):
    q = get_first_user_message(messages)
    return normalize_question(q) if q else None
import pandas as pd

def load_part1(path):
    df = pd.read_parquet(path)
    return df

def load_part2(path):
    df = pd.read_parquet(path)
    return df
def merge_by_question(p1_path, p2_path, out_path):
    df1 = load_part1(p1_path)
    df2 = load_part2(p2_path)
    merged = []
    print(f"len part1: {len(df1)}")
    print(f"len part2: {len(df2)}")

    for i, item in df1.iterrows():
        extra_field = item.get("extra_info", {})
        data_source = extra_field["id"]
        find = False
        for j, item2 in df2.iterrows():
            data_source_2 = item2["id"]
            if data_source == data_source_2:
                ground_truth_list = []
                tmp = []
                for p_item in item2["messages"][2:]:
                    if p_item["role"] == "assistant":
                        tmp.append(p_item["content"] + "<|im_end|>")
                    elif p_item["role"] == "user":
                        if "<tool_response>" not in p_item["content"]:
                            ground_truth_list.append(tmp)
                            tmp = []
                ground_truth_list.append(tmp)
                assert "extra_info" in item.keys(), item.keys()
                assert len(ground_truth_list) == len(item["extra_info"]["question"]), f"len ground_truth_list: {len(ground_truth_list)}, len questions: {len(item['extra_info']['question'])}\n{item2['messages']}\n"
                prompt = tokenizer.apply_chat_template(item['prompt'], add_generation_prompt=True, tokenize=False)
                prompt2 = tokenizer.apply_chat_template(item2["messages"], add_generation_prompt=False, tokenize=False)
                prompt2 = prompt2[len(prompt):].strip()
                
                item['reward_model']["ground_truth_str"] = prompt2
                item['extra_info']["ground_truth_str"] = prompt2
                item['extra_info']["ground_truth_list"] = ground_truth_list
                merged.append(item)
                find = True
                break
                # breakpoint()
        if not find:
            print("Not find question:", question)

    merges = pd.DataFrame(merged)
    merges.to_parquet(out_path, index=False)
    print(f"✅ merged {len(merges)} samples")
merge_by_question(
    "/mnt/userdata/FC/verl-tool-hyp/data/fcl/training_data/train.parquet",
    "/mnt/userdata/FC/verl-tool-hyp/data/fcl_sft/training_data/train.parquet",
    "./data/fcl_m/training_data/train.parquet"
)
