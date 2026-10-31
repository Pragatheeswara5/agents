from langgraph.graph import MessagesState, StateGraph, START, END
from dotenv import load_dotenv
import os
from openai import OpenAI
from tools import multiply, subtract, add
import json
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

state = MessagesState(messages=[])

load_dotenv()

client = OpenAI(
    base_url="https://ollama.com/v1",
    api_key=os.getenv("OLLAMA_API_KEY")
)

tools = [
    {
        "type":"function",
        "function": {
            "name":"multiply",
            "description":"Multiply two numbers",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{"type":"number"},
                    "b":{"type":"number"}
                },
                "required":["a", "b"]
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"subtract",
            "description":"Subtract second number from first number",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{"type":"number"},
                    "b":{"type":"number"}
                },
                "required":["a", "b"]
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"add",
            "description":"Add two numbers",
            "parameters":{
                "type":"object",
                "properties":{
                    "a":{"type":"number"},
                    "b":{"type":"number"}
                },
                "required":["a", "b"]
            }
        }
    }
]

def chatbot(state: MessagesState):
    
    messages = []

    for msg in state["messages"]:
        if msg.type == "human":
            messages.append({
                "role": "user",
                "content": msg.content
            })

        elif msg.type == "ai":
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["args"])
                            }
                        }
                        for call in msg.tool_calls
                    ]
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or ""
                })

        elif msg.type == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": str(msg.content)
            })
        
        
    response = client.chat.completions.create(
        model="gemma4:31b",
        messages = messages,
        tools=tools,
        tool_choice="auto"
        )
    
    assistant_message = response.choices[0].message
    
    if assistant_message.tool_calls:
        return {
            "messages": [
                AIMessage(
                    content=assistant_message.content or "",
                    tool_calls=[
                        {
                            "name":call.function.name,
                            "args":json.loads(call.function.arguments),
                            "id":call.id,
                        }
                        for call in assistant_message.tool_calls
                    ]
                )
            ]
            
        }
    else:
        return {
            "messages": [
                AIMessage(content=assistant_message.content or "")
            ]
        }
        
available_tools = [multiply, add, subtract]

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(available_tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

memory = MemorySaver()

graph=graph_builder.compile(
    checkpointer=memory
)

config = {
    "configurable": {
        "thread_id": "conversation_1"
    }
}

result1 = graph.invoke({
    "messages": [
        {"role": "user", "content": "My name is Prag"}
    ]
},
    config=config
)

print("First response")
print(result1["messages"][-1].content)

result2=graph.invoke(
    {
        "messages":[
            {"role":"user", "content":"What is my name"}
        ]
    },
    config=config
)

print("Second response:")
print(result2["messages"][-1].content)