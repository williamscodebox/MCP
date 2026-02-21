import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

LM_API_KEY = os.getenv("LM_API_KEY")
LM_API_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL = os.getenv("LM_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct")


async def call_lmstudio_http(messages):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            LM_API_URL,
            headers={"Authorization": f"Bearer {LM_API_KEY}"},
            json={
                "model": LM_MODEL,
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "fetch_video_transcript",
                            "description": "Fetch a YouTube transcript from a video URL",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"}
                                },
                                "required": ["url"]
                            }
                        }
                    }
                ],
                "tool_choice": "auto"
            },
        ) as resp:
            return await resp.json()

