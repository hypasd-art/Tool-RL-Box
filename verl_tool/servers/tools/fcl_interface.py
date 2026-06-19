"""
add-apt-repository ppa:deki/firejail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install firejail firejail-profiles
"""
import os
import json
from .base import BaseTool, register_tool
import regex as re

from typing import Tuple, Dict, Any, Optional, Union, List
from verl_tool.servers.tools.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call, clear_multi_turn_instances

@register_tool
class FCLInterfaceTool(BaseTool):
    tool_type = "fcl_interface"
    def __init__(self, num_workers=1):
        super().__init__(num_workers=num_workers)
        self.fcl_tools = {}
        self.env = {}
        # Load FCL tool schema

    def get_usage_inst(self):
        return "You are able to write and execute Python code securely inside a Firejail sandbox."
    
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
        valid = True
        if "</tool_call>" in action:
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
                        action_list.append("Error JSON format: {}".format(action))
                
                has_tool_call = True
        if not has_tool_call:
            return "", valid
        
        return action_list, valid
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute FCL action.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string containing search query
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        
        if trajectory_id not in self.env:
            self.env[trajectory_id] = {"turn": 0, "action": {}, "extra_field": extra_field}
        final_actions_is_gt = extra_field.get("final_actions_is_gt", False)
        if final_actions_is_gt:
            print(f"final_actions_is_gt {trajectory_id}")
            return "", True, True
        parsed_action, is_valid = self.parse_action(action)
        has_tool_call = False if parsed_action == "" else True
                    question = extra_field["question"]
        missed_function = extra_field["missed_function"]
        if not has_tool_call:
            question = extra_field["question"]
            for idx, q in enumerate(question):
                if idx <= self.env[trajectory_id]["turn"]:
                    continue
                q = q[0]["content"]
                
                if self.env[trajectory_id]["turn"] not in self.env[trajectory_id]["action"]:
                    self.env[trajectory_id]["action"][self.env[trajectory_id]["turn"]] = []
                self.env[trajectory_id]["action"][self.env[trajectory_id]["turn"]].append({"action": parsed_action, "valid": is_valid, "observation": q})
                self.env[trajectory_id]["turn"] += 1
                assert idx == self.env[trajectory_id]["turn"], "Turn order is not correct"
                self.clean_up(trajectory_id, extra_field)
                return q, False, True
            assert idx == len(question) - 1
            return "", True, True
        
        if not is_valid:
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            try:
                # Call the tool
                initial_config = json.loads(extra_field["initial_config"])
                involved_classes = extra_field["involved_classes"]
                test_entry_id = extra_field["id"]
                execution_results, involved_instances = execute_multi_turn_func_call(
                    parsed_action,
                    initial_config,
                    involved_classes,
                    "model",
                    test_entry_id,
                    long_context=True if "long_context" in test_entry_id else False,
                    trajectory_id=trajectory_id,
                )

                # Format observation similar to Search-R1
                observation = ""
                for execution_result in execution_results:
                    observation += f'\n<tool_response>\n{execution_result.strip()}\n</tool_response>'
                done = False
                valid = True
                
            except Exception as e:
                print(f"FCL error for trajectory {trajectory_id}: {e}")
                execution_result = f"FCL error: {str(e)}"
                observation = f'\n<tool_response>\ntool temporarily unavailable\n</tool_response>'
                done = False
                valid = False
        observation = observation.strip()
        if self.env[trajectory_id]["turn"] not in self.env[trajectory_id]["action"]:
            self.env[trajectory_id]["action"][self.env[trajectory_id]["turn"]] = []
        self.env[trajectory_id]["action"][self.env[trajectory_id]["turn"]].append({"action": parsed_action, "valid": is_valid, "observation": observation})
        
        return observation, done, valid

    def clean_up(self, trajectory_id, extra_field):
        test_entry_id = extra_field["id"]
        involved_classes = extra_field["involved_classes"]
        clear_multi_turn_instances("model", test_entry_id, involved_classes, trajectory_id)
    def clean_all(self):
        for trajectory_id in self.env.keys():
            self.clean_up(trajectory_id, self.env[trajectory_id]["extra_field"])
