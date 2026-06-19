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

MODEL_NAME = os.getenv("MODEL_NAME")  
API_KEY = os.getenv("API_KEY")  
base_url = os.getenv("BASE_URL")
client = OpenAI(api_key=API_KEY, base_url=base_url)


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
    
    formatted = f"<tool_call>\n{json_str}\n</tool_call>"
    return formatted



def process_turn_together(interaction_json: str, ground_truth: list = None, question: str=None, data_source: str = None, available_tools: str = None):
    formatted_summary = f"User question at Turn 1: {question[0][0]['content']}\n"
    turn_num = 0
    num = 0
    for i, step in enumerate(interaction_json):
        if "<tool_response>" in step.get('obs'):
            formatted_summary += (
                f" - Step {num+1}:\n"
                f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
                f"   - obs: {step.get('obs', '').strip()}\n"
            )
            num += 1
        else:
            formatted_summary += (
                f" - Step {num+1}:\n"
                f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
                
            )
            ground_truth_summary = ""
            step_ground_truth = ground_truth[turn_num]
            if len(step_ground_truth) == 0:
                if "miss_func" in data_source:
                    ground_truth_summary += f"Missing Function, No tool calls at this turn\n"
                elif "miss_param" in data_source:
                    ground_truth_summary += f"Missing Parameter, No tool calls at this turn\n"
                else:
                    raise Exception
            for v in step_ground_truth:
                ground_truth_summary += to_tool_call_format(v) + "\n"
            formatted_summary += f"\n*ground truth action sequence for this turn*:\n{ground_truth_summary}"
            
            num = 0
            turn_num += 1
            if i < len(interaction_json) - 1:
                formatted_summary += f"- User question at Turn {turn_num+1}: {step.get('obs', '').strip()}\n"
            
    
    prompt = f"""
You are an expert in intelligent agent interaction analysis.

The following data records the **entire multi-turn interaction process** between a model and tools, along with the corresponding ground-truth sequences. Your task is to conduct a **comprehensive and layered evaluation**.

You need to conduct Turn-by-Turn Detailed Analysis, focusing on *fine-grained reasoning* for each turn.

### Please structure your output as follows:

For each turn, we suggest you to analyze:
1. **User Intent Understanding, the correct Behavior and Logic and Success Points of the action**  
2. **Mistake Classification (Fine-grained) and Root Cause Analysis if error (Why)**  
3. **Improvement Plan**  
   - Give corrected reasoning and ideal tool call sequence for every turns and the format should be in <tool_call></tool_call>.
   - Explain how it aligns with the ground truth and the logic of the action.

---

### Available Tools:
{available_tools}


### Model Interaction Log (All Turns):
{formatted_summary}

---

Notes:
- If a turn includes "Missing Function" or "Missing Parameter", the model should have requested clarification. And no tool calls should be made at this turn.
- Output should be comprehensive and actionable.
- All tool calls must follow the format below (no markdown or ```json blocks):
<tool_call>
{{"name": "<function_name>", "arguments": {{<args-json-object>}}}}
</tool_call>
"""   
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an intelligent agent behavior analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=16384
    )
    return prompt, response.choices[0].message.content.strip()
    

def process_error_turn(interaction_json: str, ground_truth: list = None, question: str=None, data_source: str = None, available_tools: str = None, error_turn: int = None):
    if error_turn > 0:
        p_question = "\n".join([question[i][0]['content'] for i in range(error_turn)])
        formatted_summary = f"Previous user questions: {p_question}\nCurrent user question: {question[error_turn][0]['content']}\n"
    else:
        formatted_summary = f"Current user question: {question[error_turn][0]['content']}\n"
    turn_num = 0
    num = 0
    for i, step in enumerate(interaction_json):
        if i == error_turn:
            if "<tool_response>" in step.get('obs'):
                formatted_summary += (
                    f" - Step {num+1}:\n"
                    f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
                    f"   - obs: {step.get('obs', '').strip()}\n"
                )
                num += 1
            else:
                formatted_summary += (
                    f" - Step {num+1}:\n"
                    f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
                    
                )
                ground_truth_summary = ""
                step_ground_truth = ground_truth[turn_num]
                if len(step_ground_truth) == 0:
                    if "miss_func" in data_source:
                        ground_truth_summary += f"Missing Function, No tool calls at this turn\n"
                    elif "miss_param" in data_source:
                        ground_truth_summary += f"Missing Parameter, No tool calls at this turn\n"
                    else:
                        raise Exception
                for v in step_ground_truth:
                    ground_truth_summary += to_tool_call_format(v) + "\n"
                formatted_summary += f"\n*ground truth action sequence for this turn*:\n{ground_truth_summary}"
                
    prompt = f"""
You are an expert in intelligent agent interaction analysis.

The following data records the **error interaction process** between a model and tools, along with the corresponding ground-truth sequences. Your task is to conduct a **comprehensive and layered evaluation**.

The current turn has the error and you need to analyze the error reason and how to get the correct answer.

### Please structure your output as follows:

we suggest you to analyze:
1. **User Intent Understanding**  
2. **Mistake Classification (Fine-grained) and Root Cause Analysis if error (Why)**  
3. **Improvement Plan**  
   - Give corrected reasoning and ideal tool call sequence for every turns and the format should be in <tool_call></tool_call>.
   - Explain how it aligns with the ground truth and the logic of the action.

---

### Available Tools:
{available_tools}


### Model Interaction Log (Current Turns):
{formatted_summary}

---

Notes:
- If the turn includes "Missing Function" or "Missing Parameter", the model should have requested clarification. And no tool calls should be made at this turn.
- Output should be comprehensive and actionable.
- All tool calls must follow the format below (no markdown or ```json blocks):
<tool_call>
{{"name": "<function_name>", "arguments": {{<args-json-object>}}}}
</tool_call>
"""   
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an intelligent agent behavior analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=16384
    )
    return prompt, response.choices[0].message.content.strip()

def process_error_turn_n(interaction_json: str, ground_truth: list = None, question: str=None, data_source: str = None, available_tools: str = None, error_turn: int = None):
    if error_turn > 0:
        p_question = "\n".join([question[i][0]['content'] for i in range(error_turn)])
        formatted_summary = f"Previous user questions: {p_question}\nCurrent user question: {question[error_turn][0]['content']}\n"
    else:
        formatted_summary = f"Current user question: {question[error_turn][0]['content']}\n"
    turn_num = 0
    num = 0
    for i, step in enumerate(interaction_json):
        if i == error_turn:
            if "<tool_response>" in step.get('obs'):
                formatted_summary += (
                    f" - Step {num+1}:\n"
                    f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
                    f"   - obs: {step.get('obs', '').strip()}\n"
                )
                num += 1
            else:
                formatted_summary += (
                    f" - Step {num+1}:\n"
                    f"   - action: {step.get('action', '').replace('<|im_end|>', '').strip()}\n"
                    
                )
                ground_truth_summary = ""
                step_ground_truth = ground_truth[turn_num]
                if len(step_ground_truth) == 0:
                    if "miss_func" in data_source:
                        ground_truth_summary += f"Missing Function, No tool calls at this turn\n"
                    elif "miss_param" in data_source:
                        ground_truth_summary += f"Missing Parameter, No tool calls at this turn\n"
                    else:
                        raise Exception
                for v in step_ground_truth:
                    ground_truth_summary += to_tool_call_format(v) + "\n"
                formatted_summary += f"\n*ground truth action sequence for this turn*:\n{ground_truth_summary}"
                
    prompt = f"""
You are an expert in intelligent agent tool-use analysis.

Below is the erroneous interaction between a model and tools (including the ground-truth tool-use sequence for reference). Your task is to provide a **comprehensive, layered, and actionable analysis**, and **generalize to similar scenarios** to create augmentation data.  
**Note:** You should write in natural language (not structured JSON), but every “correct tool call” must strictly follow the raw <tool_call> ... </tool_call> format.

Inputs to be filled by caller:
- Available Tools: {available_tools}
- Model Interaction Log (Error Turn): {formatted_summary}

Your required outputs:

1. **Core Error Analysis (must be first)**  
   - Briefly state the root cause of the error (1-2 lines)  
   - Provide 2-4 pieces of evidence from the interaction log to support your conclusion  
   - Provide one immediate and actionable fix recommendation

2. **Generalization: Generate 3-5 similar scenarios**  
   For each scenario, describe in natural paragraphs:
   - One-sentence scenario description (what & why it occurs)  
   - The user request and necessary context  
   - A typical model mistake that might happen  
   - **The correct tool call sequence**  
     (each step must use the raw `<tool_call>\n{{...}}\n</tool_call>` block, containing exactly one JSON object with only `name` and `arguments`)  
   - What reasoning procedures can be borrowed from the current failed case  
   - What aspects differ and require different handling  
   - A brief validation strategy for confirming the fix (auto or manual)

   The scenarios must cover different error categories:
   - Missing parameters  
   - Wrong tool selection  
   - Output mismatch  
   - Timeout/failure  
   - Ambiguous request or multi-intent

3. **Provide one scenario that is similar but requires a different solution**  
   - Explain the similarity (shared reasoning or decision features)  
   - Explain clearly why different tool(s) are required  
   - Provide correct tool calls in the same `<tool_call>` format  
   - Provide rationale

4. **Output style and quality requirements**  
   - Natural language, clear paragraph structure  
   - No JSON schemas, no markdown code fences, no meta-explanations  
   - Tool calls must be valid JSON and executable  
   - Provide 1-2 acceptance criteria per scenario for filtering low-quality synthetic data

5. **Example (format reference only, do not repeat unless needed)**:
<tool_call>
{{"name": "search_database", "arguments": {{"query": "latest model evaluation", "top_k": 5}}}}
</tool_call>

---

Please begin by analyzing the root cause of the provided error turn, then generalize to 3–5 similar scenarios (including at least 1 scenario that requires different handling).
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an intelligent agent behavior analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=16384
    )
    return prompt, response.choices[0].message.content.strip()


def batch_process_file(input_path, output_path, args):
    from tqdm import tqdm
        
    def process_item_together(item):
        """Analyze a single sample (runs in independent thread)."""
        interaction_json = item["tool_interact_info"]
        ground_truth = item.get("ground_truth")
        question = item["extra_info"]["question"]
        match = re.search("<tools>\n(.*?)\n</tools>", item["prompt"], re.DOTALL)
        available_tools_n = ""
        if match:
            available_tools = match.group(1).strip()
            tools = available_tools.split("\n")
            for t in tools:
                t_json = json.loads(t)
                available_tools_n += f"{t_json['name']};"
        return_result = copy.deepcopy(item)
        return_result["prompt_analysis"], return_result["analysis_result"] = process_turn_together(interaction_json, ground_truth, question, item["data_source"], available_tools_n)
        if item["error_turn"] is not None:
            return_result["error_turn_analysis"], return_result["error_turn_reason"] = process_error_turn_n(interaction_json, ground_truth, question, item["data_source"], available_tools_n, item["error_turn"])

        return [return_result]


    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    valid_items = [item for item in content if "tool_interact_info" in item and item["traj_ids"][-2:] in ["_1"]]
    print(f"  ➤ Valid samples: {len(valid_items)}")
    progress_bar = tqdm(total=len(valid_items), desc=f"🔧 Sample progress ({os.path.basename(input_path)})", unit="item")
    

    analysis_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor: 
        future_to_item = {executor.submit(process_item_together, item): item for item in valid_items}
        for future in as_completed(future_to_item):
            try:
                analysis_results.extend(future.result())
            except Exception as e:
                print(f"⚠️ Error processing item: {e}")
            finally:
                progress_bar.update(1)
    progress_bar.close()

    # Save results
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(analysis_results, out, indent=2, ensure_ascii=False)

    print(f"💾 Analysis complete, saved to: {output_path}")
    return analysis_results
    
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
    
def process_files(new_files, min_step=0, max_step=0):
    
    n_files = []
    for f in new_files:
        step = extract_step(f)
        if step is None:
            continue

        if min_step < step <= max_step:
            n_files.append(f)
    return n_files

def monitor_and_process(input_dir: str, output_dir: str, args, check_interval: int = 100, target_num = 50, min_step=0, max_step=0):
    """
    Monitor input_dir and automatically process new JSON files.
    """
    from tqdm import tqdm
    print(f"👀 Monitoring directory: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    processed_files = set(process_files(set(os.listdir(output_dir)), min_step, max_step))
    sft_data = []
    train_data = []

    while True:
        try:
            # Find all pending files
            files = sorted(
                [f for f in os.listdir(input_dir) if f.endswith(".json") and "-val-" not in f],
                key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', x)]
            )
            new_files = process_files([f for f in files if f not in processed_files], min_step, max_step)
            
            if len(processed_files) == target_num:
                break
            print(f"Processed: {len(processed_files)}, files: {processed_files}")
            print(f"Pending: {len(new_files)}, files: {new_files}")

            if new_files:
                print(f"🔍 Found {len(new_files)} new files: {new_files}")
                for filename in tqdm(new_files, desc="File processing progress", unit="file"):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    print(f"🚀 Processing: {filename}")

                    try:
                        analysis_results = batch_process_file(input_path, output_path, args) 
                        for item in analysis_results:
                            response = item["analysis_result"] if "</think>" not in item["analysis_result"] else item["analysis_result"].replace("<think>", "").replace("</think>", "") 
                            row_index = len(train_data)
                            row_id = "process_knowledge_" + str(row_index)
                            reward_model_data = {
                                    "style": "rule",
                                    "index": row_index,
                                    "question": None,
                                    "ground_truth": [[response]],
                                    "id": row_id,
                            }
                            extra_info = {
                                "data_source": "process_knowledge",
                                "index": row_index,
                                "question": None,
                                "split": "train",
                                "ground_truth": [[response]],
                                "id": row_id,
                            }
                            train_data.append(
                                {
                                    "data_source": "process_knowledge",
                                    "prompt": [{"role": "user", "content": item["prompt_analysis"]}],
                                    "ability": "error_turn_reasoning",
                                    "reward_model": reward_model_data,
                                    "extra_info": extra_info,
                                    "metadata": None,
                                }
                            )
                            if "error_turn_analysis" in item:
                                row_index = len(train_data)
                                row_id = "process_knowledge_" + str(row_index)
                                reward_model_data = {
                                    "style": "rule",
                                    "index": row_index,
                                    "question": None,
                                    "ground_truth": [[item["error_turn_reason"].replace("<think>", "").replace("</think>", "")]],
                                    "id": row_id,
                                }
                                extra_info = {
                                    "data_source": "process_knowledge",
                                    "index": row_index,
                                    "question": None,
                                    "split": "train",
                                    "ground_truth": [[item["error_turn_reason"].replace("<think>", "").replace("</think>", "")]],
                                    "id": row_id,
                                }
                                train_data.append(
                                    {
                                        "data_source": "process_knowledge",
                                        "prompt": [{"role": "user", "content": item["error_turn_analysis"]}],
                                        "ability": "error_turn_reasoning",
                                        "reward_model": reward_model_data,
                                        "extra_info": extra_info,
                                        "metadata": None,
                                    }
                                    )
                        for item in analysis_results:
                            response = item["analysis_result"] if "</think>" not in item["analysis_result"] else item["analysis_result"].replace("<think>", "").replace("</think>", "") 
                            sft_data.append(
                                {
                                    "prompt": [{"role": "user", "content": item["prompt_analysis"]}],
                                    "response": response
                                }
                            )
                            if "error_turn_analysis" in item:
                                sft_data.append(
                                    {
                                        "prompt": [{"role": "user", "content": item["error_turn_analysis"]}],
                                        "response": item["error_turn_reason"].replace("<think>", "").replace("</think>", "")
                                    }
                                    )
                        print(json.dumps(train_data[:2], indent=2))
                        processed_files.add(filename)
                        print(f"✅ File {filename} processed.")
                    except Exception as e:
                        print(f"❌ File {filename} failed: {e}")

            else:
                print("No new files, waiting...")
                time.sleep(check_interval)
                continue

        except Exception as e:
            print(f"⚠️ Monitor loop error: {e}")
    df_row = pd.DataFrame(train_data)
    print(df_row.head())
    print(f"✅ Data aggregation complete, {len(df_row)} samples total")
    # Save processed DataFrame
    train_output_file_path = output_dir + "/train_rl.parquet"
    df_row.to_parquet(train_output_file_path, index=False)
    print(f"Saved {len(df_row)} processed rows to {train_output_file_path}")
    df_sft = pd.DataFrame(sft_data)
    print(df_sft.head())
    print(f"✅ SFT data aggregation complete, {len(df_sft)} samples total")
    # Save processed DataFrame
    sft_output_file_path = output_dir + "/sft_data.parquet"
    df_sft.to_parquet(sft_output_file_path, index=False)

    df_row = pd.DataFrame(random.sample(train_data, 100))
    print(df_row.head())
    print(f"✅ Data aggregation complete, {len(df_row)} samples total")
    # Save processed DataFrame
    train_output_file_path = output_dir + "/train_rl_subset.parquet"
    df_row.to_parquet(train_output_file_path, index=False)
    print(f"Saved {len(df_row)} processed rows to {train_output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tool interaction logs of an AI model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input JSON path")
    parser.add_argument("--output_dir", type=str, required=True, help="")
    parser.add_argument("--workers", type=int, default=8, help="")
    parser.add_argument("--target_num", type=int, default=20, help="")
    parser.add_argument("--min_step", type=int, default=20, help="")
    parser.add_argument("--max_step", type=int, default=20, help="")

    args = parser.parse_args()
    monitor_and_process(args.input_dir, args.output_dir, args, target_num=args.target_num, min_step=args.min_step, max_step=args.max_step)
