from email.mime import message
import os
from database import (delete_task, get_memories, get_task, init_db, add_task, list_tasks, save_memory, search_documents)
# from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from openai import OpenAI

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
                        "description":"Due date for the task. If the user specifies a due date, such as tomorrow or Friday, pass that date here. Otherwise, omit it (optional)"
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
    },
    {
        "type":"function",
        "function":{
            "name":"get_task",
            "description":"Get a specific task by its ID.",
            "parameters":{
                "type":"object",
                "properties":{
                    "task_id":{
                        "type":"integer",
                        "description":"The ID of the task to retrieve"
                    }
                },
                "required":["task_id"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"delete_task",
            "description":"Delete a specific task by its ID.",
            "parameters":{
                "type":"object",
                "properties":{
                    "task_id":{
                        "type":"integer",
                        "description":"The ID of the task to delete"
                    }
                },
                "required":["task_id"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"save_memory",
            "description":"Save a memory for the user.",
            "parameters":{
                "type":"object",
                "properties":{
                    "memory":{
                        "type":"string",
                        "description":"The memory to save"
                    }
                },
                "required":["memory"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_memories",
            "description":"Get all memories for the user.",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_documents",
            "description":"Search for relevant documents based on a query.",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The search query"
                    }
                },
                "required":["query"]
            }
        }
    }
]

init_db()

load_dotenv()

client = OpenAI(
    base_url="https://ollama.com/v1",
    api_key=os.getenv("OLLAMA_API_KEY")
)

available_tools = {
            "add_task": add_task,
            "list_tasks": list_tasks,
            "get_task": get_task,
            "delete_task": delete_task,
            "save_memory": save_memory,
            "get_memories": get_memories,
            "search_documents": search_documents
        }

import json
def run_agent(user_message):
    
    state={
        "messages":[
            {
                "role":"user",
                "content":user_message
            }
            ],
            "tool_results":[]
    }
    
    tool_call_counts = {}
    MAX_TOOL_CALLS = 3
    
    MAX_ITERATIONS = 10
    iteration_count = 0
    
    while True:
        iteration_count += 1
        
        if iteration_count > MAX_ITERATIONS:
            return (
            f"Agent stopped: exceeded the maximum of "
            f"{MAX_ITERATIONS} iterations. "
            "Please try a simpler request."
        )
        
        response = client.chat.completions.create(
        model="gemma4:31b",
        messages=state["messages"],
        tools=tools,
        tool_choice="auto"
        )

        message = response.choices[0].message
        
        if not message.tool_calls:
            return message.content

        state["messages"].append(message)


        for tool in message.tool_calls:
            tool_name = tool.function.name
            function = available_tools[tool_name]
            
            arguments = json.loads(tool.function.arguments) if tool.function.arguments else {}
            
            result = function(**arguments)
            
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            
            if tool_call_counts[tool_name] > MAX_TOOL_CALLS:
                state["messages"].append({
        "role": "tool",
        "tool_call_id": tool.id,
        "content": (
            f"Error: Tool '{tool_name}' exceeded its "
            f"maximum of {MAX_TOOL_CALLS} calls. "
            "Choose another action or provide a final answer."
        )
    })
                continue

            state["messages"].append(
                {
                    "role":"tool",
                    "tool_call_id":tool.id,
                    "content":str(result)
                }
            )
            
            state["tool_results"].append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })

        # final_response = client.chat.completions.create(
        #     model="Qwen/Qwen2.5-72B-Instruct",
        #     messages=messages,
        #     tools=tools
        # )

        # print(final_response.choices[0].message.content)
            
answer = run_agent(
    "List all my current tasks."
)

print("\nAssistant:")
print(answer)