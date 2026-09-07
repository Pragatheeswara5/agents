import os
from database import (init_db, add_task, list_tasks)
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

tools = [
    {
        "type":"function",
        "function":{
            "name":"add_task",
            "description":"Add a new task to the user's task list",
            "parameters": {
                "type":"object",
                "properties":{
                    "task":{
                        "type":"string",
                        "description":"The task to add"
                    },
                    "due_date":{
                        "type":"string",
                        "description":"The due date for the task (optional)"
                    }
                },
                "required":["task"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"list_tasks",
            "description":"Get the user's incomplete tasks.",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[]
            }
        }
    }
]

init_db()

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

messages=[
    {
        "role":"user",
        "content":"Add a task to call the dentist next Monday and show me my pending tasks."
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

available_tools = {
    "add_task": add_task,
    "list_tasks": list_tasks
}

messages.append(message)

import json

for tool in message.tool_calls:
    tool_name = tool.function.name
    function = available_tools[tool_name]
    
    arguments = json.loads(tool.function.arguments) if tool.function.arguments else {}
    
    result = function(**arguments)

    messages.append(
        {
            "role":"tool",
            "tool_call_id":tool.id,
            "content":str(result)
        }
    )

final_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=messages,
    tools=tools
)

print(final_response.choices[0].message.content)