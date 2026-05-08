"""
Jarvis Realtime Voice — Web Server
Serves the web UI and proxies WebSocket connections to the OpenAI Realtime API.
Uses ephemeral tokens so the API key never reaches the browser.
Supports configurable voice, model, VAD, and latency parameters.
"""

import json
import os
import sys
from pathlib import Path

from aiohttp import web
from openai import AsyncOpenAI

DEFAULT_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-mini-realtime-preview")
DEFAULT_VOICE = os.getenv("REALTIME_VOICE", "ash")

AVAILABLE_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
AVAILABLE_MODELS = [
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-realtime-preview",
]

JARVIS_SYSTEM_PROMPT = """You are Jarvis, a highly capable and articulate AI assistant.
You speak with confident clarity and a touch of refined wit — think of a brilliant chief of staff
who anticipates needs and delivers precise, actionable responses.

Key traits:
- Address the user respectfully but warmly
- Be concise yet thorough — never ramble
- When uncertain, say so honestly rather than guessing
- Offer proactive suggestions when you spot opportunities
- Maintain a calm, composed demeanor even with complex requests

You are currently operating in voice mode. Keep responses conversational and natural —
avoid overly long responses since you are speaking aloud.
Keep responses under 3 sentences for fast, low-latency interactions."""

client: AsyncOpenAI | None = None


async def handle_index(request: web.Request) -> web.Response:
    """Serve the web UI."""
    index_path = Path(__file__).parent / "index.html"
    return web.Response(
        text=index_path.read_text(),
        content_type="text/html",
    )


async def handle_config(request: web.Request) -> web.Response:
    """Return available configuration options."""
    return web.json_response({
        "voices": AVAILABLE_VOICES,
        "models": AVAILABLE_MODELS,
        "defaults": {
            "model": DEFAULT_MODEL,
            "voice": DEFAULT_VOICE,
            "vad_type": "semantic_vad",
            "vad_eagerness": "high",
            "temperature": 0.6,
            "max_response_output_tokens": 512,
        },
    })


async def handle_session(request: web.Request) -> web.Response:
    """Create an ephemeral token for browser WebSocket connections.

    Accepts JSON body with optional overrides:
      - model, voice, vad_type, vad_eagerness, vad_threshold,
        vad_silence_ms, temperature, max_response_output_tokens
    """
    assert client is not None

    body = {}
    if request.content_type == "application/json":
        body = await request.json()

    model = body.get("model", DEFAULT_MODEL)
    voice = body.get("voice", DEFAULT_VOICE)
    vad_type = body.get("vad_type", "semantic_vad")
    temperature = body.get("temperature", 0.6)
    max_tokens = body.get("max_response_output_tokens", 512)

    # Build turn detection config optimized for low latency
    if vad_type == "semantic_vad":
        turn_detection = {
            "type": "semantic_vad",
            "eagerness": body.get("vad_eagerness", "high"),
        }
    elif vad_type == "server_vad":
        turn_detection = {
            "type": "server_vad",
            "threshold": body.get("vad_threshold", 0.5),
            "prefix_padding_ms": body.get("vad_prefix_padding_ms", 200),
            "silence_duration_ms": body.get("vad_silence_ms", 400),
        }
    else:
        turn_detection = None

    session_params = {
        "model": model,
        "voice": voice,
        "modalities": ["text", "audio"],
        "instructions": JARVIS_SYSTEM_PROMPT,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "temperature": temperature,
        "max_response_output_tokens": max_tokens,
        "turn_detection": turn_detection,
    }

    session = await client.beta.realtime.sessions.create(**session_params)

    return web.json_response({
        "token": session.client_secret.value,
        "model": model,
        "voice": voice,
        "vad_type": vad_type,
        "expires_at": session.client_secret.expires_at,
    })


async def handle_output(request: web.Request) -> web.Response:
    """Serve saved audio files."""
    filename = request.match_info["filename"]
    filepath = Path(__file__).parent / "output" / filename
    if not filepath.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(filepath)


async def on_startup(app: web.Application) -> None:
    global client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    client = AsyncOpenAI(api_key=api_key)
    print(f"\n  Jarvis Realtime Voice Server")
    print(f"  Model: {DEFAULT_MODEL}  |  Voice: {DEFAULT_VOICE}")
    print(f"  Open http://localhost:8090 in your browser\n")


def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get("/", handle_index)
    app.router.add_get("/config", handle_config)
    app.router.add_post("/session", handle_session)
    app.router.add_get("/output/{filename}", handle_output)
    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8090"))
    web.run_app(create_app(), host="0.0.0.0", port=port)
