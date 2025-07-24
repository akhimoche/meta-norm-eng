import requests
import json
import re
from typing import Dict, List, Tuple, Any
import heapq
from operator_funcs import a_star,move_agent
import ast

# This is the file where LLM integration is tackled. 
# Note that the metaconventions/relevant actions discussed are for the Coins envt, not Commons Harvest 
# General outline: 
# 1. Sends a parsed (RGB-> Natual language dicrionary/grid) observation to the local LLM via Ollama API.
    # Includes meta-conventions for the speicfic game (eg prioritise coins))
# 2. LLM returns a JSON response with movement recommendations for each agent, which is parsed back into Python. 
# 3. A* and operator_funcs are used to compute the actual movement actions based on the LLM's recommendations.
# 4. Move agent to execute 
# Still no potential refusal mechanism for the agent? (emphasised in Notion)

def query_ollama(state_info: Dict[str, Any]) -> Dict[str, Any]:
    messages = [{
    "role": "user",
    "content": f"""
You directly control the actions of a number of agents in a sequential social dilemma (SSD). The SSD is Coins,
a Python meltingpot environment where each agent has the action set ACTION_SET = (FORWARD, TURN_ANTICLOCKWISE, TURN_CLOCKWISE). ORIENTATION MATTERS -
The player needs to TURN to go forward in a particular direction. This information is given (north, south, east, west) in the observation. Players can collide.

An example of the observation is:
"p_purple_south: [(1, 1)], coin_purple: [(3, 1)], coin_blue: [(2, 2)], p_blue_south: [(2, 0)]",
where:
- `p_purple_south` indicates that player purple is facing south, located at (1,1).
- `coin_purple` indicates a purple coin at (3,1).
- `coin_blue` indicates a blue coin at (2,2).
- `p_blue_south` indicates that player blue is facing south, located at (2,0).

{json.dumps(state_info)}

**Meta Convention**: "Always prioritize collecting coins that matches the agent's color."
Analyze the situation and provide only movement recommendations in the **strict JSON format** like below:

```
{{
  "Recommendation for agent purple": [
    {{
      "Initial State": "[1,1]",
      "Destination State": "[3,1]",
      "Reason": "Matching the coin",
      "Obstacles": "Any obstacles in the path, if none, output 'None'"
    }}
  ]
  "Recommendation for agent blue": [
    {{
      "Initial State": "[2,0]",
      "Destination State": "[2,2]",
      "Reason": "Matching the coin",
      "Obstacles": "Any obstacles in the path, if none, output 'None'"
    }}
  ]
}}
Remember the output should only contains the json formate response!
""" }]


    data = {
        "model": "llama3.3:70b",  # 或其他可用的模型
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=data)
        # if response.status_code == 404:
        #     # 尝试使用旧的 API 端点
        #     old_url = self.ollama_url.replace('/api/chat', '/api/generate')
        #     old_data = {
        #         "model": "llama3.3:70b",
        #         "prompt": messages[0]["content"],
        #         "stream": False
        #     }
        #     response = requests.post(old_url, json=old_data)

        response.raise_for_status()
        result = response.json()


        # 处理不同的响应格式
        if "message" in result:
            return {"response": result["message"]["content"]}
        elif "response" in result:
            return {"response": result["response"]}
        else:
            return result

    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        print(f"URL tried: {self.ollama_url}")
        print(f"Request data: {json.dumps(data, indent=2)}")
        return None


def extract_recommendations_from_response(response):
    """
    Extract recommendations from the given response.

    Args:
        response (str): A string containing JSON data enclosed in triple backticks.

    Returns:
        dict: A dictionary containing extracted recommendations for agents, or an error message if extraction fails.
    """
    # Extract the JSON content within the backticks
    json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)

    if not json_match:
        return {"error": "No JSON data found in response"}

    json_data = json_match.group(1)  # Extract the JSON string

    try:
        data = json.loads(json_data)  # Parse the JSON string into a Python dictionary
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}"}

    # Extract recommendations
    recommendations = {}
    for key in data:
        if key.startswith("Recommendation for agent"):
            agent_name = key.split("for agent ")[1]
            agent_data = data[key][0]  # Assuming each recommendation list contains a single dictionary
            recommendations[agent_name] = {
                "Initial State": agent_data.get("Initial State", "N/A"),
                "Destination State": agent_data.get("Destination State", "N/A"),
                "Reason": agent_data.get("Reason", "N/A"),
                "Obstacles": agent_data.get("Obstacles", "N/A")
            }

    return recommendations

def main():
    # 示例观察数据
    observation = {
        'p_purple_south': [(1, 0)],
        'coin_blue': [(10, 13)],
        'coin_purple': [(4, 13)],
        'p_blue_south': [(22, 17)]
    }


    ollama_response = query_ollama(observation)
    if ollama_response:
        recommendation = extract_recommendations_from_response(ollama_response["response"])
        print("\nOllama analysis:")
        print(json.dumps(ollama_response, indent=2))
        print(recommendation['purple']['Initial State'],recommendation['purple']['Destination State'],recommendation['purple']['Obstacles'])
    purple_path = a_star(tuple(ast.literal_eval(recommendation['purple']['Initial State'])), tuple(ast.literal_eval(recommendation['purple']['Destination State'])), tuple(ast.literal_eval(recommendation['purple']['Obstacles'])) if ast.literal_eval(recommendation['purple']['Obstacles']) is not None else None, (6,6))
    print(purple_path)

    for i in range(len(purple_path)-1):
        coord1 = purple_path[i]
        coord2 = purple_path[i+1]
        print(move_agent(coord1,coord2))
if __name__ == "__main__":
    main()