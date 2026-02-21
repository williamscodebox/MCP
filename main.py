import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_TOKEN_LLM")

resp = requests.post(
    "http://localhost:1234/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "your-model-name-here",
        "messages": [{"role": "user", "content": "Hello"}]
    }
)

print(resp.json())
