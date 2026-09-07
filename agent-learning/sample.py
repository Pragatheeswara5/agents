import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[
        {
            "role":"user",
            "content":"Explain what an AI agent is in one sentence."
        }
    ]
)

print(response.choices[0].message.content)