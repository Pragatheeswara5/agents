import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

def calculator(a:float, b:float):
    return a*b

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]

messages = [
    {
        "role": "user",
        "content": "What is the capital of India?"
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message)

message = response.choices[0].message

import json

tool_call = message.tool_calls[0]

arguments = json.loads(tool_call.function.arguments)

results = calculator(
    arguments["a"],
    arguments["b"]
)

print("Tool result: ", results)

messages.append(message)
messages.append(
    {
        "role":"tool",
        "tool_call_id":tool_call.id,
        "content":str(results)
    }
)

final_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=messages,
    tools=tools
)

print(
    final_response.choices[0].message.content
)