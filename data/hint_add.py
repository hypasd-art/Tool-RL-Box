import os
import json
import random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import argparse
from typing import Iterator, Dict, Any, Optional

from openai import OpenAI
MODEL_NAME = os.getenv("MODEL_NAME")   
API_KEY = os.getenv("API_KEY")
base_url= os.getenv("BASE_URL", "")
client = OpenAI(api_key=API_KEY, base_url=base_url)
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
def process_item(item):
    prompt = "The following content is the conversation between a user and an assistant. The assistant can use external tools to help or finish the user's questions.\n\n" # item["prompt"]
    for turn, question in enumerate(item["extra_info"]["question"]):
        gt = ';\n'.join(item['extra_info']['ground_truth'][turn])
        prompt += f"\nUser: {question[0]['content']}\nAssistant: {gt}"
    prompt +='\n\nPlease summarize the key reasoning steps, tool usage patterns, and reusable hints that would help solve similar tasks in the future no more than 300 words. In your summary, you need to abstract the user\'s questions for each and provide step-by-step prompts or tool calling examples within <tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call> based on each question.'
    item["extra_info"]["hint"] = "There is some note for you to help you get the correct answer, but you don't need to follow them strictly.\n" + get_hint(prompt = prompt) + "\nAttention the tool call with function name and arguments should be within <tool_call></tool_call>, and the response to the user should end with <|im_end|> only."
    print("---------------" + item["extra_info"]["hint"] + "-------------------")
    return item
def iterate_parquet_rows_pandas(path: str) -> Iterator[Dict[str, Any]]:
    """
    Simple pandas approach (loads entire file into memory; not suitable for very large files).
    """
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield row.to_dict()

def get_hint(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192
        )
    except Exception as e:
        print(str(e))
        return ""
    # breakpoint()
    return response.choices[0].message.content

MAX_WORKERS = 32
def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_data = []
    test_data = []
    result = []
    raw_train_data = list(iterate_parquet_rows_pandas("/mnt/userdata/FC/verl-tool-hyp/data/fcl/training_data/train.parquet"))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Show progress bar with tqdm
        futures = [executor.submit(process_item, item) for item in raw_train_data]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Train Data"):
            result_item = future.result()
            train_data.append(result_item)

    for idx, item in enumerate(iterate_parquet_rows_pandas("/mnt/userdata/FC/verl-tool-hyp/data/fcl/training_data/test.parquet")):
        test_data.append(item)
    
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
        default="/mnt/userdata/FC/verl-tool-hyp/data/hint_fcl/training_data/", # netdisk/yphao
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")
    args = parser.parse_args()

    main()