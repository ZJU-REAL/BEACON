# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List
import re

def sciworld_projection(actions: List[str]):
    """
    Process Science World actions.
    Expected format:
        <think>reasoning...</think><action>action text</action>

    Args:
        actions: List of action strings from the model
    Returns:
        processed_actions: List of extracted actions
        valids: List of validity flags (1 if valid format, 0 otherwise)
    """
    valids = [0] * len(actions)
    processed_actions = []

    for i in range(len(actions)):
        original_str = actions[i]
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = original_str.find(start_tag)
        end_idx = original_str.find(end_tag)

        try:
            if start_idx == -1 or end_idx == -1:
                # No valid action tags found
                processed_actions.append(original_str[-20:])
                continue

            # Extract action between tags
            extracted_action = original_str[start_idx + len(start_tag):end_idx].strip()
            processed_actions.append(extracted_action)
            valids[i] = 1
        except:
            processed_actions.append(original_str[-20:])

        # Check for <think> tags (required in verl-agent)
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # Check for Chinese characters (invalid)
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return processed_actions, valids
