import os
import sys
import asyncio
import json
from dotenv import load_dotenv
from agents.mcp.server import MCPServerStdio
from lmstudio_http import call_lmstudio_http

# Load environment variables
load_dotenv()

LM_API_KEY = os.getenv("LM_API_KEY")
LM_MODEL = os.getenv("LM_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct")
LM_API_URL = "http://localhost:1234/v1/chat/completions"


async def run_chat_loop(mcp_server):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("== Local LM Studio HTTP + MCP + Tool Calling ==")

    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user})

        # Send to LM Studio
        response = await call_lmstudio_http(messages)
        msg = response["choices"][0]["message"]

        if "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                name = tool_call["function"]["name"]

                # arguments come in as a JSON STRING
                raw_args = tool_call["function"]["arguments"]
                args = json.loads(raw_args)

                if name == "fetch_video_transcript":
                    if "url" not in args:
                        print("Tool call missing 'url' argument:", args)
                        continue
                    url = args["url"]

                    # Call MCP tool
                    result = await mcp_server.call_tool("fetch_video_transcript", {"url": url})

                    # Extract the actual text from TextContent
                    # tool_result.content is a list of TextContent objects
                    result_text = result.content[0].text

                    # Return tool result to LM Studio
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result_text
                    })

            # Ask LM Studio to continue the conversation
            response = await call_lmstudio_http(messages)
            msg = response["choices"][0]["message"]

        # Normal assistant message
        messages.append(msg)
        print("\nAssistant:", msg["content"], "\n")



async def main():
    """Start MCP server and chat loop."""
    async with MCPServerStdio(
        name="YouTube MCP Server",
        params={
            "command": sys.executable,
            "args": ["mcp-server/yt-mcp.py"],
        },
    ) as mcp_server:
        await run_chat_loop(mcp_server)


if __name__ == "__main__":
    asyncio.run(main())