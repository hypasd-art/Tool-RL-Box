import os
import re
import argparse
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time
import copy
import random

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct")

    
def extract_step(filename: str) -> int | None:
    """
    Extract the first integer from a filename.
    e.g. step-12.json -> 12
         fcl_eval-step-345.json -> 345
    """
    m = re.search(r'(\d+)', filename)
    if m:
        return int(m.group(1))
    return None


def monitor_and_process(input_dir: str, train_data_dir: str, min_step: int, max_step: int):
    """
    Monitor input_dir and automatically process new JSON files.
    """
    from tqdm import tqdm
    print(f"👀 Monitoring directory: {input_dir}")
    output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    
    train_idx = set()
    processed_files = []
    sft_data = []
    train_data = []

    # Find all pending files
    files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".json") and "-val-" not in f],
        key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', x)]
    )
    new_files = []
    for f in files:
        step = extract_step(f)
        if step is None:
            continue

        if min_step < step <= max_step:
            new_files.append(f)

    print(f"🔍 Found new files to process: {new_files}")
    for filename in tqdm(new_files, desc="File processing progress", unit="file"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(f"🚀 Processing new file: {filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        traj_groups = dict()
        for item in content:
            traj_ids = item.get("traj_ids", "")
            if traj_ids is None:
                continue

            traj_group_id = traj_ids[:-2]   # Key: group by trajectory (strip last 2 chars)

            if traj_group_id not in traj_groups:
                traj_groups[traj_group_id] = []

            traj_groups[traj_group_id].append(item)

        for traj_group_id, items in traj_groups.items():
            # Check if all responses in the group are wrong
            all_wrong = all(item.get(“score”, 0) == 0 for item in items)
            assert all(item["id"] == items[0]["id"] for item in items), "id not equal"

            if all_wrong:
                last_item = items[-1]   # Only keep the last response
                train_idx.add(last_item["id"])

            for idx, item in enumerate(content):
                if item["score"] == 0:
                    train_idx.add(item["id"])


    data = pd.read_parquet(train_data_dir)
    for idx, item in data.iterrows():
        if item["id"] in train_idx:
            sft_data.append(item)

    for idx, full_messages in enumerate(sft_data):
        for idx, item in enumerate(full_messages["messages"]):
            if item["role"] == "assistant":
                prompt = tokenizer.apply_chat_template(full_messages["messages"][:idx], add_generation_prompt=True, tokenize=False)
                response = item["content"]
                train_data.append(
                    {
                        "prompt": prompt,
                        "response": response
                    }
                )

    df_row = pd.DataFrame(train_data)
    print(df_row.head())
    print(f"✅ Data aggregation complete, {len(df_row)} samples total")
    # Save processed DataFrame
    train_output_file_path = output_dir + "/train_sft.parquet" 
    df_row.to_parquet(train_output_file_path, index=False)
    print(f"Saved {len(df_row)} processed rows to {train_output_file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tool interaction logs of an AI model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input JSON path")
    parser.add_argument("--train_dir", type=str, required=True, help="")
    parser.add_argument("--min_step", type=int, default=20, help="")
    parser.add_argument("--max_step", type=int, default=20, help="")

    args = parser.parse_args()
    monitor_and_process(args.input_dir, args.train_dir, args.min_step, args.max_step)
