"""
OpenAI Realtime API — CLI Client
Supports text-to-speech, speech-to-speech (microphone), and interactive modes.
Configurable voice, model, and VAD settings targeting 250-500ms latency.
"""

import argparse
import asyncio
import base64
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

# ── Configuration ────────────────────────────────────────────────────────────

AVAILABLE_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
AVAILABLE_MODELS = ["gpt-4o-mini-realtime-preview", "gpt-4o-realtime-preview"]
OUTPUT_DIR = Path(__file__).parent / "output"

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


def build_session_config(args) -> dict:
    """Build optimized session configuration from CLI args."""
    config = {
        "modalities": ["text", "audio"],
        "voice": args.voice,
        "instructions": JARVIS_SYSTEM_PROMPT,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "temperature": args.temperature,
        "max_response_output_tokens": args.max_tokens,
    }

    if args.vad == "semantic":
        config["turn_detection"] = {
            "type": "semantic_vad",
            "eagerness": args.eagerness,
        }
    elif args.vad == "server":
        config["turn_detection"] = {
            "type": "server_vad",
            "threshold": args.vad_threshold,
            "prefix_padding_ms": 200,
            "silence_duration_ms": args.silence_ms,
        }
    else:
        config["turn_detection"] = None

    return config


def save_wav(audio_chunks: list[bytes], path: Path) -> float:
    """Save PCM16 chunks as a WAV file. Returns duration in seconds."""
    raw = b"".join(audio_chunks)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(raw)
    return len(raw) / (24000 * 2)


async def run_text_to_audio(args, user_text: str) -> None:
    """Send a text message and stream back audio + transcript."""
    client = AsyncOpenAI()
    audio_chunks: list[bytes] = []
    transcript_parts: list[str] = []

    print(f"\n{'─' * 60}")
    print(f"  Jarvis Realtime Voice")
    print(f"  Model: {args.model}  |  Voice: {args.voice}")
    print(f"{'─' * 60}")
    print(f"\n  User: {user_text}\n")

    t_start = time.perf_counter()

    async with client.realtime.connect(model=args.model) as conn:
        session_config = build_session_config(args)
        session_config["turn_detection"] = None  # Manual for text mode
        await conn.session.update(session=session_config)

        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        })
        await conn.response.create()

        t_first_audio = None
        print("  Jarvis: ", end="", flush=True)
        async for event in conn:
            if event.type == "response.audio.delta":
                chunk = base64.b64decode(event.delta)
                audio_chunks.append(chunk)
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()

            elif event.type == "response.audio_transcript.delta":
                transcript_parts.append(event.delta)
                print(event.delta, end="", flush=True)

            elif event.type == "response.done":
                print("\n")
                break

            elif event.type == "error":
                print(f"\n  ERROR: {event.error}")
                break

    # Latency report
    if t_first_audio:
        latency_ms = (t_first_audio - t_start) * 1000
        print(f"  First audio latency: {latency_ms:.0f}ms")

    # Save audio
    if audio_chunks:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = OUTPUT_DIR / f"jarvis_{timestamp}.wav"
        duration = save_wav(audio_chunks, wav_path)
        raw_size = sum(len(c) for c in audio_chunks)
        print(f"  Audio saved: {wav_path}")
        print(f"  Duration:    {duration:.1f}s  |  Size: {raw_size / 1024:.0f} KB")

    if transcript_parts:
        full_transcript = "".join(transcript_parts)
        print(f"\n{'─' * 60}")
        print(f"  Transcript: {full_transcript}")
        print(f"{'─' * 60}\n")


async def run_speech_to_speech(args) -> None:
    """Full speech-to-speech pipeline: mic input → Realtime API → audio output.

    Requires PyAudio (pip install pyaudio).
    """
    try:
        import pyaudio
    except ImportError:
        print("ERROR: PyAudio is required for speech-to-speech mode.")
        print("  Install with: pip install pyaudio")
        sys.exit(1)

    client = AsyncOpenAI()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    RATE = 24000
    CHANNELS = 1
    CHUNK = 2400  # 100ms of audio at 24kHz

    pa = pyaudio.PyAudio()

    print(f"\n{'━' * 60}")
    print(f"  Jarvis Realtime Voice — Speech-to-Speech Mode")
    print(f"  Model: {args.model}  |  Voice: {args.voice}")
    print(f"  VAD: {args.vad} | Eagerness: {args.eagerness}")
    print(f"  Press Ctrl+C to exit.")
    print(f"{'━' * 60}\n")

    async with client.realtime.connect(model=args.model) as conn:
        session_config = build_session_config(args)
        await conn.session.update(session=session_config)

        # Open mic input stream
        mic_stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        # Open speaker output stream
        spk_stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK,
        )

        audio_chunks: list[bytes] = []
        turn = 0
        is_speaking = False
        t_speech_end = 0.0

        # Background task to continuously send mic audio
        async def send_mic_audio():
            loop = asyncio.get_event_loop()
            while True:
                try:
                    data = await loop.run_in_executor(
                        None, lambda: mic_stream.read(CHUNK, exception_on_overflow=False)
                    )
                    b64 = base64.b64encode(data).decode()
                    await conn.input_audio_buffer.append(audio=b64)
                except Exception:
                    break

        mic_task = asyncio.create_task(send_mic_audio())

        try:
            print("  Listening... (speak naturally, Jarvis will respond)")
            async for event in conn:
                if event.type == "input_audio_buffer.speech_started":
                    if not is_speaking:
                        print("\n  [You are speaking...]", end="", flush=True)
                    is_speaking = True

                elif event.type == "input_audio_buffer.speech_stopped":
                    is_speaking = False
                    t_speech_end = time.perf_counter()
                    print(" done.", flush=True)

                elif event.type == "input_audio_buffer.committed":
                    if t_speech_end == 0:
                        t_speech_end = time.perf_counter()
                    turn += 1
                    audio_chunks = []

                elif event.type == "response.audio_transcript.delta":
                    if not audio_chunks:
                        print(f"  Jarvis: ", end="", flush=True)
                    print(event.delta, end="", flush=True)

                elif event.type == "response.audio.delta":
                    chunk = base64.b64decode(event.delta)
                    audio_chunks.append(chunk)

                    # Measure latency on first chunk
                    if len(audio_chunks) == 1 and t_speech_end > 0:
                        latency = (time.perf_counter() - t_speech_end) * 1000
                        print(f"\n  [Latency: {latency:.0f}ms]", end="", flush=True)

                    # Play audio in real time
                    spk_stream.write(chunk)

                elif event.type == "response.done":
                    print()
                    if audio_chunks:
                        wav_path = OUTPUT_DIR / f"jarvis_s2s_turn_{turn:03d}.wav"
                        duration = save_wav(audio_chunks, wav_path)
                        print(f"  [Saved: {wav_path.name} — {duration:.1f}s]")
                    audio_chunks = []
                    t_speech_end = 0
                    print("  Listening...", flush=True)

                elif event.type == "error":
                    print(f"\n  ERROR: {event.error}")
                    break

        except KeyboardInterrupt:
            print("\n\n  Goodbye!")
        finally:
            mic_task.cancel()
            mic_stream.stop_stream()
            mic_stream.close()
            spk_stream.stop_stream()
            spk_stream.close()
            pa.terminate()


async def interactive_mode(args) -> None:
    """Interactive text chat with audio responses."""
    client = AsyncOpenAI()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'━' * 60}")
    print(f"  Jarvis Realtime Voice — Interactive Mode")
    print(f"  Model: {args.model}  |  Voice: {args.voice}")
    print(f"  Type a message and press Enter. Type 'quit' to exit.")
    print(f"{'━' * 60}\n")

    async with client.realtime.connect(model=args.model) as conn:
        session_config = build_session_config(args)
        session_config["turn_detection"] = None
        await conn.session.update(session=session_config)

        turn = 0
        while True:
            try:
                user_text = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if not user_text or user_text.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break

            turn += 1
            audio_chunks: list[bytes] = []
            t_start = time.perf_counter()
            t_first_audio = None

            await conn.conversation.item.create(item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            })
            await conn.response.create()

            print("  Jarvis: ", end="", flush=True)
            async for event in conn:
                if event.type == "response.audio.delta":
                    chunk = base64.b64decode(event.delta)
                    audio_chunks.append(chunk)
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                elif event.type == "response.audio_transcript.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.done":
                    print()
                    break
                elif event.type == "error":
                    print(f"\n  ERROR: {event.error}")
                    break

            if audio_chunks:
                raw = b"".join(audio_chunks)
                wav_path = OUTPUT_DIR / f"jarvis_turn_{turn:03d}.wav"
                duration = save_wav(audio_chunks, wav_path)
                latency_str = ""
                if t_first_audio:
                    latency_str = f" | Latency: {(t_first_audio - t_start) * 1000:.0f}ms"
                print(f"  [Audio: {wav_path.name} — {duration:.1f}s{latency_str}]\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jarvis Realtime Voice — CLI client for OpenAI Realtime API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello Jarvis!"                          # Text-to-speech
  %(prog)s --interactive                             # Interactive text chat
  %(prog)s --speech                                  # Speech-to-speech (mic)
  %(prog)s --voice coral --model gpt-4o-realtime-preview "Hi there"
  %(prog)s --speech --vad semantic --eagerness high   # Low-latency speech mode
        """,
    )

    parser.add_argument("text", nargs="*", help="Text message to send (text-to-speech mode)")
    parser.add_argument("--interactive", action="store_true", help="Interactive text chat mode")
    parser.add_argument("--speech", action="store_true", help="Speech-to-speech mode (requires PyAudio)")

    voice_group = parser.add_argument_group("Voice & Model")
    voice_group.add_argument("--voice", default=os.getenv("REALTIME_VOICE", "ash"),
                             choices=AVAILABLE_VOICES, help="Voice selection (default: ash)")
    voice_group.add_argument("--model", default=os.getenv("REALTIME_MODEL", "gpt-4o-mini-realtime-preview"),
                             choices=AVAILABLE_MODELS, help="Model to use")

    vad_group = parser.add_argument_group("Turn Detection (VAD)")
    vad_group.add_argument("--vad", default="semantic", choices=["semantic", "server", "none"],
                           help="VAD mode (default: semantic)")
    vad_group.add_argument("--eagerness", default="high", choices=["low", "medium", "high"],
                           help="Semantic VAD eagerness — higher = faster response (default: high)")
    vad_group.add_argument("--vad-threshold", type=float, default=0.5,
                           help="Server VAD threshold 0.0-1.0 (default: 0.5)")
    vad_group.add_argument("--silence-ms", type=int, default=400,
                           help="Server VAD silence duration in ms (default: 400)")

    resp_group = parser.add_argument_group("Response")
    resp_group.add_argument("--temperature", type=float, default=0.6,
                            help="Response temperature (default: 0.6)")
    resp_group.add_argument("--max-tokens", type=int, default=512,
                            help="Max response tokens (default: 512)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    if args.speech:
        asyncio.run(run_speech_to_speech(args))
    elif args.interactive:
        asyncio.run(interactive_mode(args))
    elif args.text:
        user_text = " ".join(args.text)
        asyncio.run(run_text_to_audio(args, user_text))
    else:
        asyncio.run(run_text_to_audio(
            args,
            "Hello Jarvis! Please introduce yourself and tell me what you can help me with today."
        ))


if __name__ == "__main__":
    main()
