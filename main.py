import asyncio
import sys
import json
from dotenv import load_dotenv
from agents.mcp.server import MCPServerStdio
from lmstudio_http import call_lmstudio_http, call_lmstudio_no_tools

load_dotenv()


def chunk_text(text, max_chars=6000):
    """Split transcript into safe chunks for LM Studio."""
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]


async def summarize_large_text(text):
    """Summarize long transcripts safely using chunking."""
    partial_summaries = []

    # 1. Summarize each chunk
    for chunk in chunk_text(text):
        summary_request = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Summarize the following part of a YouTube transcript. "
                    "Do NOT repeat the text.\n\n" + chunk
                ),
            },
        ]

        response = await call_lmstudio_no_tools(summary_request)

        if "choices" not in response:
            raise RuntimeError("LM Studio returned an error during chunk summary.")

        partial_summaries.append(response["choices"][0]["message"]["content"])

    # 2. Summarize the summaries
    final_request = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Combine these partial summaries into one concise summary:\n\n"
                + "\n\n".join(partial_summaries)
            ),
        },
    ]

    final_response = await call_lmstudio_no_tools(final_request)

    if "choices" not in final_response:
        raise RuntimeError("LM Studio returned an error during final summary.")

    return final_response["choices"][0]["message"]["content"]


async def run_chat_loop(mcp_server):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("== Local LM Studio HTTP + MCP + Tool Calling ==")
    print("Type 'exit' to quit.\n")

    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user})

        # First LM Studio call (tools enabled)
        response = await call_lmstudio_http(messages)
        msg = response["choices"][0]["message"]

        # -------------------------------
        # TOOL CALL HANDLING
        # -------------------------------
        if "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                args = json.loads(raw_args)

                if name == "fetch_video_transcript":
                    url = args.get("url")

                    # Call MCP tool
                    tool_result = await mcp_server.call_tool(
                        "fetch_video_transcript",
                        {"url": url}
                    )

                    transcript_text = tool_result.content[0].text

                    # Replace transcript with placeholder
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": "TRANSCRIPT_RECEIVED"
                    })

                    # Summarize safely
                    final_summary = await summarize_large_text(transcript_text)

                    print("\nAssistant:", final_summary, "\n")

                    messages.append({
                        "role": "assistant",
                        "content": final_summary
                    })

                    continue

        # -------------------------------
        # NORMAL ASSISTANT MESSAGE
        # -------------------------------
        messages.append(msg)
        print("\nAssistant:", msg["content"], "\n")


async def main():
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
