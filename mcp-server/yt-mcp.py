from mcp.server.fastmcp import FastMCP
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os

# Suppress MCP INFO logs to reduce console output
import logging
logging.getLogger("mcp").setLevel(logging.WARNING)

# Create an MCP server
mcp = FastMCP("yt-mcp")

# Create prompt
@mcp.prompt()
def system_prompt() -> str:
    """Instructions for YouTube video agent"""
    script_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(script_dir, "prompts", "system_instructions.md")
    with open(prompt_path, "r") as file:
        return file.read()