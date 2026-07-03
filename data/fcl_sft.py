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

MULTI_TURN_FUNC_DOC_PATH = "./benchmarks/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"
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

MODEL_NAME = os.getenv("MODEL_NAME") # "Qwen/Qwen3-32B"  
API_KEY = os.getenv("API_KEY") # "EMPTY"  
base_url= os.getenv("BASE_URL") # ""
client = OpenAI(api_key=API_KEY, base_url=base_url)
def get_llm_toolcall(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        temperature=0.7,
        max_tokens=8192
    )
    return response.choices[0].message.content
def get_llm_response(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt + [{"role": "user", "content": "Based on the above conversation, give the response by summarizing the information and do not call tools, please dirctly give the response without indicate this is a summarization."}],
        temperature=0.7,
        max_tokens=8192
    )
    return response.choices[0].message.content
# [
#             {"role": "system", "content": "You are an intelligent agent behavior analysis assistant."},
#             {"role": "user", "content": prompt}
#         ]
def process_single_row(row, current_split_name, row_index):
    """

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    # if "miss_param_52" in row["id"]:
    #     print("Processing miss_param_52 row")
    # else:
    #     return
    import numpy as np
    function = []
    involved_classes = row.get("involved_classes")
    for func_collection in involved_classes:
        # func_doc is a list of dict
        func_doc = load_file(
            MULTI_TURN_FUNC_DOC_PATH  + "/" + MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
        )
        function.extend(func_doc)
    # Handle Miss Func category; we need to remove the holdout function doc
    if "missed_function" in row and not pd.isna(row["missed_function"]):
        # print(row["missed_function"])
        new_missed = {}
        for turn_index, missed_func_names in row["missed_function"].items():
            row["missed_function"][turn_index] = []
            for missed_func_name in missed_func_names:
                for i, func_doc in enumerate(function):
                    if func_doc["name"] == missed_func_name:
                        # Add the missed function doc to the missed_function list
                        row["missed_function"][turn_index].append(func_doc) # 
                        # Remove it from the function list
                        function.pop(i)
                        break
            #  = new_missed
    formatted_prompt = ""
    formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    for tool in function:
        tool["description"] += " Note that the provided function is in Python 3 syntax."
        formatted_prompt += f"\n{json.dumps(tool)}"
    formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'

    question = row.get("question", "")
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


    # Build prompt structure
    # user_content = question[0][0]["content"]
    prompt = [{"role": "system", "content": formatted_prompt}] # , {"role": "user", "content": user_content}
    
    # token_len = tokenizer.apply_chat_template(prompt, tokenize=False)
    # token_len = len(tokenizer.tokenize(token_len))
    # if token_len > 2048:
    #     print(f"Row {row_index} has token length {token_len}")

    # Extract ground truth from reward_model or fallback to golden_answers
    ground_truth = row.get("golden_answers", [])
    initial_config = row.get("initial_config")
    involved_classes = row.get("involved_classes")
    test_entry_id = row.get("id")
    trajectory_id = ""
    for idx, q in enumerate(processed_question):
        prompt.append({"role": "user", "content": q[0]["content"]})
        if len(ground_truth[idx]) == 0:
            # print(row)
            # assert "miss_func" in row["id"] or "miss_param" in row["id"], "Ground truth is empty but not a miss_func or miss_param case."
            if "miss_func" in row["id"]:
                content_response = "I am sorry, but I cannot provide a valid tool call as the required function is not available."
            elif "miss_param" in row["id"]:
                content_response = "I am sorry, but I cannot provide a valid tool call as the required parameters are missing."
            prompt.append({"role": "assistant", "content": content_response + ""})
            continue
        for item in ground_truth[idx]:
            prompt_toolcall = prompt + [{"role": "user", "content": """Based on the above conversation, give the tool call in <tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call> format.""" + "The ground truth tool call is as follows:\n" + item}]
            tool_call = get_llm_toolcall(prompt_toolcall)
            print(tool_call)
            # if "miss_param_52" in row["id"]:
            #     print("----------------" + tool_call + "----------------")
            # tool_call = to_tool_call_format(item)
            prompt.append({"role": "assistant", "content": tool_call + ""})
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
            prompt.append({"role": "user", "content": "<tool_response>" + execution_results[0] + "</tool_response>"}) # >
        content_response = get_llm_response(prompt).split("</think>")[-1].strip()

        print(content_response)
        prompt.append({"role": "assistant", "content": content_response + ""})
    data_source_tagged = "fcl" + "_" + "_".join(row.get("id").split("_")[:-1])
    return pd.Series({"messages": prompt, "split": current_split_name, "id": row.get("id")})



def main():


    import os
    import json
    import random
    import pandas as pd
    import numpy as np

    data_path = "./benchmarks/berkeley-function-call-leaderboard/bfcl_eval/data"
    results = []
    selected_data = []

    data_types = [
        "BFCL_v4_multi_turn_base.json",
        "BFCL_v4_multi_turn_miss_param.json",
        # "BFCL_v4_multi_turn_long_context.json",
        "BFCL_v4_multi_turn_miss_func.json"
    ]

    # Fix random seed for reproducibility
    random.seed(42)
    train_ratio = 0.5   # Ratio: 0.5 means half train, half test

    # Load all data types
    type_datasets = {}
    for data_type in data_types:
        type_results = []

        # Read main file
        with open(os.path.join(data_path, data_type), "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    type_results.append(data)

        # Merge possible_answer
        possible_path = os.path.join(data_path, "possible_answer", data_type)
        if os.path.exists(possible_path):
            with open(possible_path, "r") as f:
                possible_data = [json.loads(line.strip()) for line in f if line.strip()]
            possible_dict = {d["id"]: d["ground_truth"] for d in possible_data}
            for item in type_results:
                if item["id"] in possible_dict:
                    item["golden_answers"] = possible_dict[item["id"]]

        type_datasets[data_type] = type_results

    # === Find minimum length across all types ===
    min_len = min(len(v) for v in type_datasets.values())
    print(f"✅ All data types loaded, min samples: {min_len}")

    # === Generate fixed random train/test indices ===
    indices = list(range(min_len))
    random.shuffle(indices)
    split_idx = int(train_ratio * min_len)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    print(f"Randomly selected {len(train_indices)} train samples, {len(test_indices)} test samples")

    # === Sample from each type using the same indices ===
    for data_type, dataset in type_datasets.items():
        type_train = [dataset[i] for i in train_indices]
        type_test = [dataset[i] for i in test_indices]

        for d in type_train:
            d["split"] = "train"
            d["source_type"] = data_type
        for d in type_test:
            d["split"] = "test"
            d["source_type"] = data_type

        selected_data.extend(type_train + type_test)
    
    # df_row = pd.DataFrame(results)
    df_row = pd.DataFrame(selected_data)
    print(df_row.head())
    print(f"✅ Data aggregation complete, {len(df_row)} samples total")
    print(df_row.groupby('source_type')['split'].value_counts())
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)


    # try:
        # train_df, test_df = train_test_split(df_row, test_size=0.5, random_state=42)
    

    def parallel_apply(df, func, max_workers=32):
        results = [None] * len(df)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(func, row): i for i, row in df.iterrows()
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"❌ Error processing row {i}: {e}")
        return results


    def apply_process_row(row):
        return process_single_row(row, current_split_name=row["split"], row_index=row.name)


    # Parallel execution
    results = parallel_apply(df_row, apply_process_row, max_workers=32)

    # Convert to DataFrame
    df_processed = pd.DataFrame(results)
    # def apply_process_row(row, split_name="split"):
    #     return process_single_row(row, current_split_name=row["split"], row_index=row.name)

    # df_processed = df_row.apply(apply_process_row, axis=1)

    train_df_processed = df_processed[df_row["split"] == "train"]
    test_df_processed = df_processed[df_row["split"] == "test"]
    # train_df_processed = train_df.apply(apply_process_row, axis=1)
    # test_df_processed = test_df.apply(apply_process_row, axis=1)

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

def process_data(args):
    input_dir = args.local_dir

    # Read all .parquet files in directory
    parquet_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ]

    all_dfs = []
    for pfile in parquet_files:
        print(f"Loading file: {pfile}")
        df = pd.read_parquet(pfile)

        for idx, item in df.iterrows():
            formatted_prompt = ""
            assert item["messages"][0]["role"] == "system"
            content = item["messages"][0]["content"]
            # print(content)
            match = re.search(r"<tools>\n(.*?)\n</tools>", content, re.DOTALL)
            tool = match.group(1).strip()
            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for t in tool.split("\n"):
                t = json.loads(t)
                t["description"] = t["description"].replace(" Note that the provided function is in Python 3 syntax.", "")
                t["description"] += " Note that the provided function is in Python 3 syntax."
                formatted_prompt += f"\n{json.dumps(t)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'
            item["messages"][0]["content"] = formatted_prompt
            # print(item["messages"][0]["content"])
            # breakpoint()
        df.to_parquet(pfile, index=False)

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get fcl training data and convert")
    parser.add_argument(
        "--input_dir", default="", help="Input directory to load the Parquet files."
    )
    parser.add_argument(
        "--local_dir",
        default="/mnt/userdata/FC/verl-tool-hyp/data/fcl_sft/training_data/", # netdisk/yphao
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()
    # process_data(args)
    main()
