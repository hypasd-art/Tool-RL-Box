# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import argparse
import logging
import os
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs
from transformers import AutoTokenizer
import re
from verl_tool.servers.tools.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MULTI_TURN_FUNC_DOC_PATH = "./benchmarks/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"
MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}
tokenizer = AutoTokenizer.from_pretrained("/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct")

def normalize_mixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list/dict/object columns to strings to avoid parquet serialization errors.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return df

def load_file(file_path, sort_by_id=False, allow_concatenated_json=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            try:
                content = json.loads(line)
                result.append(content)
            except Exception as e:
                if not allow_concatenated_json:
                    raise e

                # Although this really shouldn't happen, sometimes a result file might have more than one JSON objects concatenated on a single line instead of one per line (e.g. '{"id": 1, xxx}{"id": 2, xxx}').
                # We can parse them incrementally by using `json.JSONDecoder.raw_decode`, which returns both the parsed object and the index where it stopped parsing.
                line_jsons = []
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(line):
                    # Skip whitespace between objects (if any)
                    while idx < len(line) and line[idx].isspace():
                        idx += 1

                    if idx >= len(line):
                        break

                    try:
                        json_obj, idx = decoder.raw_decode(line, idx)
                        line_jsons.append(json_obj)
                    except json.JSONDecodeError:
                        # If decoding fails at any point, the entire line is invalid.
                        raise e

                # After parsing, we must ensure the entire line has been consumed.
                # If `idx` is not at the end of the line, it means there's trailing
                # garbage, which is an error.
                if idx < len(line):
                    raise e

                if not line_jsons:
                    # If the line was non-empty but contained no JSON objects (e.g., only whitespace),
                    # it's an error.
                    raise e

                result.extend(line_jsons)

    if sort_by_id:
        result.sort(key=sort_key)
    return result

def to_tool_call_format(s: str) -> str:
    """
    Convert a string like find(path='.',name='test_document.txt')
    into <tool_call> JSON format.
    """
    match = re.match(r"(\w+)\((.*)\)", s.strip())
    if not match:
        raise ValueError("Invalid format. Expected: func(a='x', b='y')")
    
    func_name, args_str = match.groups()
    
    args = {}
    for k, v in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str):
        args[k] = v

    tool_call = {
        "name": func_name,
        "arguments": args
    }

    json_str = json.dumps(tool_call, ensure_ascii=False)
    
    formatted = f"<tool_call>\n{json_str}\n</tool_call>" # \\
    return formatted
from openai import OpenAI

# MODEL_NAME = os.getenv("MODEL_NAME") # "Qwen/Qwen3-32B"  
# API_KEY = os.getenv("API_KEY") # "EMPTY"  
# base_url= os.getenv("BASE_URL") # ""
# client = OpenAI(api_key=API_KEY, base_url=base_url)
def get_llm_response(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt + [{"role": "user", "content": "Based on the above conversation, give the response by summarizing the information and do not call tools."}],
        temperature=0.7,
        max_tokens=8192
    )
    return response.choices[0].message.content

def process_single_row(row, current_split_name, row_index):
    """
    First perform complete sampling of the entire record (execute all tool calls
    and get the final assistant reply for each turn), then split into per-tool-call
    samples. Each sample is a pd.Series with fields:
      - prompt: str (system/tools + context up to the current tool call, including
        the current assistant's tool_call and tool_response)
      - response: str (the assistant's final reply for this turn)
      - split, id, turn_index, call_index
    """
    from copy import deepcopy

    function = []
    involved_classes = row.get("involved_classes") or []
    for func_collection in involved_classes:
        func_doc = load_file(
            MULTI_TURN_FUNC_DOC_PATH + "/" + MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
        )
        function.extend(func_doc)

    # Handle missed_function (same as original logic)
    if "missed_function" in row and not pd.isna(row["missed_function"]):
        for turn_index, missed_func_names in row["missed_function"].items():
            row["missed_function"][turn_index] = []
            for missed_func_name in missed_func_names:
                for i, func_doc in enumerate(function):
                    if func_doc["name"] == missed_func_name:
                        row["missed_function"][turn_index].append(func_doc)
                        function.pop(i)
                        break

    # Build tools section (system, as string)
    formatted_prompt = ""
    formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for tool in function:
        formatted_prompt += f"\n{json.dumps(tool)}"
    formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'

    question = row.get("question", [])
    processed_question = []
    if "missed_function" in row and not pd.isna(row["missed_function"]):
        for idx, item in enumerate(question):
            if str(idx) in row["missed_function"]:
                processed_question.append([{"role": "user", "content": json.dumps(row["missed_function"][str(idx)]) + "\nI have updated some more functions you can choose from. What about now?"}])
                assert len(item) == 0
            else:
                processed_question.append(item)
    else:
        processed_question = question

    ground_truth = row.get("golden_answers", [])
    initial_config = row.get("initial_config")
    test_entry_id = row.get("id")
    trajectory_id = ""

    # ---- Step 1: Complete sampling of entire data (execute all tool calls and record their results, plus the assistant's final reply per turn) ----
    # Store tool execution results for each (turn_idx, call_idx)
    execution_store = {}

    # Build messages (list of dict) for complete sampling, compatible with get_llm_response interface
    full_messages = [{"role": "system", "content": formatted_prompt}]

    for turn_idx, user_turn in enumerate(processed_question):
        # append user message for this turn
        user_content = user_turn[0]["content"]
        full_messages.append({"role": "user", "content": user_content})

        # If this turn has no tool calls, continue to the next turn
        if turn_idx >= len(ground_truth) or len(ground_truth[turn_idx]) == 0:
            # Directly generate assistant final reply from the model (when no tool calls)
            try:
                assistant_final = get_llm_response(full_messages).split("</think>")[-1].strip()
            except Exception:
                assistant_final = ""
            full_messages.append({"role": "assistant", "content": assistant_final})
            continue

        # This turn has tool calls: execute sequentially and add tool_call + tool_response to full_messages
        for call_idx, item in enumerate(ground_truth[turn_idx]):
            tool_call_str = to_tool_call_format(item)
            full_messages.append({"role": "assistant", "content": tool_call_str})

            parsed_action = [item]
            execution_results, involved_instances = execute_multi_turn_func_call(
                parsed_action,
                initial_config,
                involved_classes,
                "model",
                test_entry_id,
                long_context=True if "long_context" in test_entry_id else False,
                trajectory_id=trajectory_id,
            )
            tool_response = execution_results[0] if execution_results else ""
            execution_store[(turn_idx, call_idx)] = tool_response
            full_messages.append({"role": "user", "content": "<tool_response>" + tool_response + "</tool_response>"})

        # After all tool calls and responses for this turn are added, call the model for the final assistant reply (full)
        try:
            assistant_final = get_llm_response(full_messages).split("</think>")[-1].strip()
        except Exception:
            assistant_final = ""
        full_messages.append({"role": "assistant", "content": assistant_final})
    
    train_data = []
    for idx, item in enumerate(full_messages):
        if item["role"] == "assistant":
            prompt = tokenizer.apply_chat_template(full_messages[:idx], add_generation_prompt=True, tokenize=False)
            response = item["content"]
            train_data.append(
                {
                    "prompt": prompt,
                    "response": response
                }
            )
    return train_data

import pyarrow.parquet as pq
import pandas as pd
from typing import Iterator, Dict, Any, Optional
def iterate_parquet_rows_pandas(path: str) -> Iterator[Dict[str, Any]]:
    """
    Simple pandas approach (loads entire file into memory; not suitable for very large files).
    """
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield row.to_dict()

def main():

    import os
    import json
    import random
    import pandas as pd
    import numpy as np

    
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_data = []
    test_data = []
    result = []
    for idx, full_messages in enumerate(iterate_parquet_rows_pandas("/mnt/userdata/FC/verl-tool-hyp/data/fcl_sft/training_data/train.parquet")):
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

    for idx, full_messages in enumerate(iterate_parquet_rows_pandas("/mnt/userdata/FC/verl-tool-hyp/data/fcl_sft/training_data/test.parquet")):
        for idx, item in enumerate(full_messages["messages"]):
            if item["role"] == "assistant":
                prompt = tokenizer.apply_chat_template(full_messages["messages"][:idx], add_generation_prompt=True, tokenize=False)
                response = item["content"]
                test_data.append(
                    {
                        "prompt": prompt,
                        "response": response
                    }
                )
    
    print(len(train_data))
    # Convert to DataFrame
    train_df_processed = pd.DataFrame(train_data)
    test_df_processed = pd.DataFrame(test_data)

    # Save processed DataFrame
    train_output_file_path = os.path.join(local_save_dir, f"train.parquet")
    train_df_processed.to_parquet(train_output_file_path, index=False)
    logger.info(f"Saved {len(train_df_processed)} processed rows to {train_output_file_path}")

    test_output_file_path = os.path.join(local_save_dir, f"test.parquet")
    test_df_processed.to_parquet(test_output_file_path, index=False)
    logger.info(f"Saved {len(test_df_processed)} processed rows to {test_output_file_path}")

    # except Exception as e:
    #     logger.error(f"Error processing dataset: {e}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get fcl training data and convert")
    parser.add_argument(
        "--input_dir", default="", help="Input directory to load the Parquet files."
    )
    parser.add_argument(
        "--local_dir",
        default="/mnt/userdata/FC/verl-tool-hyp/data/fcl_sft_perturn/training_data/", # netdisk/yphao
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    main()
