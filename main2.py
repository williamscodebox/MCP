import asyncio
import sys
import re
from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServerStdio, MCPServer
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv

load_dotenv()


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL."""
    match = re.search(r"v=([A-Za-z0-9_-]{6,})", url)
    return match.group(1) if match else "unknown_video"


async def main():
    async with MCPServerStdio(
        name="YouTube MCP Server",
        params={
            "command": sys.executable,
            "args": ["mcp-server/yt-mcp.py"],
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="YT MCP Agent Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server)


async def run(mcp_server: MCPServer):
    # Load system prompt from MCP server
    prompt_result = await mcp_server.get_prompt("system_prompt")
    instructions = prompt_result.messages[0].content.text

    # Create agent
    agent = Agent(
        name="YouTube Video Agent",
        instructions=instructions,
        mcp_servers=[mcp_server],
        model="lmstudio:default",
    )

    input_items = []

    print("=== YouTube Agent ===")
    print("Type 'exit' to end the conversation")

    while True:
        user_input = input("\nUser: ").strip()
        input_items.append({"content": user_input, "role": "user"})

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        result = Runner.run_streamed(
            agent,
            input=input_items,
        )

        print("\nAgent: ", end="", flush=True)

        async for event in result.stream_events():

            # Stream assistant text
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

            # Tool call started
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    tool_name = event.item.raw_item.name

                    if tool_name == "fetch_video_transcript":
                        print("\n-- Fetching transcript...", flush=True)
                    else:
                        print(f"\n-- Calling {tool_name}...", flush=True)

                # Tool call output (transcript returned)
                elif event.item.type == "tool_call_output_item":
                    transcript_text = event.item.output

                    # Extract video ID from the original user input
                    video_id = extract_video_id(user_input)
                    filename = f"transcript_{video_id}.txt"

                    print(f"-- Saving transcript to {filename}...", flush=True)

                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(transcript_text)

                    print("-- Transcript saved.", flush=True)

                    # Add a message to the conversation so the agent knows the tool completed
                    input_items.append({
                        "content": f"Transcript saved to {filename}.",
                        "role": "assistant"
                    })

                # Assistant message output
                elif event.item.type == "message_output_item":
                    input_items.append({
                        "content": event.item.raw_item.content[0].text,
                        "role": "assistant"
                    })

        print("\n")  # spacing after each response


if __name__ == "__main__":
    asyncio.run(main())
