import json
from datasets import load_dataset
from transformers import AutoTokenizer
import copy

def parse_tool_calls(text):
    # 匹配 <tool_calls>...</tool_calls> 中的内容
    tool_call_str = text
    try:
        # 尝试整体解析为 Python 表达式
        expr = ast.parse(tool_call_str, mode="eval")
        body = expr.body

        # 多个调用用列表包装
        if isinstance(body, ast.List):
            call_nodes = body.elts
        else:
            call_nodes = [body]

        parsed_calls = []
        for call in call_nodes:
            if not isinstance(call, ast.Call):
                continue
            # 支持模块名前缀：如 data_privacy.generate_compliance_report
            func = call.func
            if isinstance(func, ast.Attribute):
                parts = []
                while isinstance(func, ast.Attribute):
                    parts.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    parts.append(func.id)
                func_name = ".".join(reversed(parts))
            elif isinstance(func, ast.Name):
                func_name = func.id
            else:
                continue  # 不支持的函数结构
            kwargs = {}
            for kw in call.keywords:
                kwargs[kw.arg] = ast.literal_eval(kw.value)

            parsed_calls.append({
                "name": func_name,
                "arguments": kwargs
            })

        return parsed_calls

    except Exception as e:
        print(f"解析失败: {e}")
        return []

dataset = load_dataset("./ToolACE")
print(dataset)
tokenizer = AutoTokenizer.from_pretrained("/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct")

convert_data = []
for data in dataset["train"]:
    con_data = {"prompt": [], "response": ""}
    available_tools = data["system"].split("Here is a list of functions in JSON format that you can invoke:\n")[-1].split(".\nShould you decide to return the function call(s).")[0]
    try:
        available_tools_list = json.loads(available_tools.strip())
    except:
        print(available_tools)
        continue
    trans_dict = {}
    for index, item in enumerate(available_tools_list):
        if " " in item["name"]:
            trans_dict[item["name"]] = item["name"].replace(" ", "_")
            available_tools_list[index]["name"] = item["name"].replace(" ", "_")
    available_tools = json.dumps(available_tools_list, indent=2)
    profix = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    system_prompt = profix + available_tools + """\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n"""
    con_data["prompt"] = [{"role": "system", "content": system_prompt}]
    for item in data["conversations"]:
        if item["from"] == "assistant": # and 
            if item["value"].startswith("["):
                tool_call_str = ""
                content = item["value"]
                for api, trans_api in trans_dict.items():
                    if api in item["value"]:
                        content = item["value"].replace(api, trans_api)
                parse_content = parse_tool_calls(content)
                parse_content = json.dumps(parse_content)
                for tool_call in parse_content:
                    tool_call_str += "<tool_call>\n" + json.dumps(tool_call) + "\n</tool_call>\n"
                con_data["response"] = tool_call_str
                con_data["prompt"] = tokenizer.apply_chat_template(con_data["prompt"], add_generation_prompt=True, tokenize=False)
                convert_data.append(copy.deepcopy(con_data))
                con_data["prompt"].append({"role": item["from"], "content": tool_call_str})
            else:
                con_data["response"] = item["value"]
                con_data["prompt"].append({"role": item["from"], "content": item["value"]})
        else:
            con_data["prompt"].append({"role": item["from"], "content": item["value"]})
import pandas as pd

df = pd.DataFrame(convert_data)
df.to_parquet("../toolace_sft_perturn/training_data/train.parquet", engine="pyarrow")
# with open("toolace/training_data/train.parquet", "w") as f:
#     json.dump(convert_data, f, indent=2)