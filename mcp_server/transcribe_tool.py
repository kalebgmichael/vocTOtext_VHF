"""MCP server exposing transcribe_audio tool backed by the VHF upload API.

Transports:
  stdio (default)        — for local Claude Code usage
  streamable-http        — set MCP_TRANSPORT=http, runs on MCP_PORT (default 8001)
                           used when running inside docker-compose

Django URL:
  UPLOAD_URL env var — defaults to localhost:8000, override to http://django:8000
  in docker-compose so the container can reach the Django service.
"""

import os
import pathlib

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "vhf-transcribe",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8001")),
)

UPLOAD_URL = os.environ.get(
    "UPLOAD_URL", "http://localhost:8000/api/transcribe/upload/"
)


@mcp.tool()
async def transcribe_audio(file_path: str, language: str | None = None) -> dict:
    """Transcribe an audio file saved in the temp directory.

    Sends the file to the VHF Django server's upload endpoint and returns the
    full transcription result.

    Args:
        file_path: Absolute path to the audio file (wav, webm, mp3, ogg, etc.)
        language:  Optional ISO 639-1 language hint, e.g. 'en' or 'es'.
                   Omit to let the model auto-detect.

    Returns:
        Dict with keys: transcription, language, language_probability,
        duration, segments (list of {start, end, text}), filename.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    form_data = {}
    if language:
        form_data["language"] = language

    async with httpx.AsyncClient(timeout=120) as client:
        with path.open("rb") as f:
            response = await client.post(
                UPLOAD_URL,
                files={"audio_file": (path.name, f, "audio/octet-stream")},
                data=form_data,
            )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
