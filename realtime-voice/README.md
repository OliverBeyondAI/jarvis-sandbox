# Jarvis Realtime Voice — OpenAI GPT Realtime API Integration

A production-ready reference implementation of OpenAI's GPT Realtime API for the Jarvis voice assistant. This prototype demonstrates real-time speech-to-speech interaction with ~250–500ms first-response latency using a single unified model for STT, reasoning, and TTS.

---

## Architecture

### Current Jarvis Voice Stack (Legacy)

```
┌──────────┐     ┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  User's  │────▶│  Twilio Voice /  │────▶│  STT Engine  │────▶│  Claude /   │
│  Phone   │     │  ConversationRelay│    │  (Deepgram)  │     │  Bedrock    │
│          │◀────│                  │◀────│              │◀────│  LLM        │
└──────────┘     └─────────────────┘     └──────────────┘     └─────────────┘
                         │                                           │
                         │              ┌──────────────┐             │
                         └──────────────│  ElevenLabs  │◀────────────┘
                                        │  TTS         │
                                        └──────────────┘

  Latency: ~800–1500ms (STT → LLM → TTS pipeline)
  Services: 4 separate vendors
  Billing: per-minute (Twilio) + per-character (ElevenLabs) + per-token (Claude)
```

### Proposed GPT Realtime Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Jarvis Server (Python/aiohttp)               │
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │  HTTP Server  │    │  Token Manager   │    │  WebSocket Proxy   │  │
│  │  (Static UI)  │    │  (Ephemeral Keys)│    │  (Session Relay)   │  │
│  └──────┬───────┘    └────────┬─────────┘    └─────────┬──────────┘  │
│         │                     │                        │             │
└─────────┼─────────────────────┼────────────────────────┼─────────────┘
          │                     │                        │
          ▼                     ▼                        ▼
┌──────────────────┐   ┌───────────────┐   ┌────────────────────────────┐
│   Web Browser    │   │  OpenAI REST  │   │   OpenAI Realtime API      │
│                  │   │  /v1/realtime │   │   wss://api.openai.com     │
│  ┌────────────┐  │   │  /sessions    │   │                            │
│  │ WebAudio   │  │   └───────────────┘   │  ┌──────────────────────┐  │
│  │ Capture &  │  │                       │  │  Unified GPT-4o      │  │
│  │ Playback   │──┼───────────────────────┼─▶│  ┌────┐ ┌───┐ ┌───┐ │  │
│  └────────────┘  │    Persistent         │  │  │STT │ │LLM│ │TTS│ │  │
│  ┌────────────┐  │    WebSocket           │  │  └────┘ └───┘ └───┘ │  │
│  │ Waveform   │  │    (PCM16 Audio)      │  └──────────────────────┘  │
│  │ Visualizer │  │                       │                            │
│  └────────────┘  │                       │  Latency: ~250–500ms       │
│  ┌────────────┐  │                       │  Services: 1 vendor        │
│  │ Settings   │  │                       │  Billing: per-token        │
│  │ Panel      │  │                       │                            │
│  └────────────┘  │                       └────────────────────────────┘
└──────────────────┘

  Audio Format: PCM16 (24 kHz, 16-bit, mono) — browser
                G.711 μ-law/A-law — telephony (SIP trunking)
```

### Data Flow

```
  User speaks into mic
        │
        ▼
  ┌─────────────────┐
  │ Browser captures │
  │ audio via        │
  │ WebAudio API     │
  └────────┬────────┘
           │ Float32 → PCM16 → Base64
           ▼
  ┌─────────────────┐      ┌─────────────────────────────┐
  │ WebSocket sends  │─────▶│ OpenAI Realtime API          │
  │ input_audio_     │      │                              │
  │ buffer.append    │      │  1. VAD detects speech end   │
  └─────────────────┘      │  2. STT transcribes audio    │
                           │  3. LLM generates response   │
  ┌─────────────────┐      │  4. TTS synthesizes speech   │
  │ Browser receives │◀─────│  5. Streams audio chunks     │
  │ response.audio.  │      │                              │
  │ delta events     │      └─────────────────────────────┘
  └────────┬────────┘
           │ Base64 → PCM16 → Float32
           ▼
  ┌─────────────────┐
  │ WebAudio plays   │
  │ back with        │
  │ scheduling       │
  └─────────────────┘
```

---

## Project Structure

```
realtime-voice/
├── README.md              ← You are here
├── server.py              ← Backend: HTTP server + ephemeral token generation
├── realtime_client.py     ← CLI client: text-to-speech, speech-to-speech, interactive
├── index.html             ← Web UI: full-featured voice interface with settings
├── migration-guide.html   ← Detailed migration guide from legacy stack
├── research-notes.md      ← Technical reference for the GPT Realtime API
└── requirements.txt       ← Python dependencies
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to the Realtime API
- A modern browser (Chrome, Edge, Firefox) for the web UI
- A working microphone for voice interaction

### 1. Install Dependencies

```bash
cd realtime-voice
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Start the Server

```bash
python server.py
```

The server starts on `http://localhost:8080`. Open that URL in your browser to use the web UI.

### 4. (Optional) Use the CLI Client

The CLI client supports three modes:

```bash
# Text-to-speech: type text, hear spoken response
python realtime_client.py --mode text

# Speech-to-speech: speak into mic, hear response
python realtime_client.py --mode speech

# Interactive: continuous text chat with audio responses
python realtime_client.py --mode interactive
```

### Configuration

The web UI includes a settings panel where you can adjust:

| Setting | Options | Default |
|---------|---------|---------|
| Voice | alloy, ash, ballad, coral, echo, sage, shimmer, verse | coral |
| Model | gpt-4o-realtime-preview, gpt-4o-mini-realtime-preview | gpt-4o-mini-realtime-preview |
| VAD Mode | semantic (natural), server_vad (fast) | semantic |
| Temperature | 0.0 – 1.5 | 0.8 |
| Max Tokens | 100 – 4096 | 1024 |

Server-side defaults can be changed via environment variables or directly in `server.py`.

---

## Pros & Cons vs. Current Jarvis Voice Stack

### Advantages of GPT Realtime API

| Category | Benefit |
|----------|---------|
| **Latency** | ~250–500ms vs. ~800–1500ms. Single model eliminates inter-service round trips. |
| **Simplicity** | One vendor, one API, one billing model. Replaces Deepgram STT + Claude LLM + ElevenLabs TTS. |
| **Voice quality** | 8 built-in voices with natural inflection, whispering, laughter, and tone-following from instructions. |
| **Cost efficiency** | Token-based pricing. `gpt-4o-mini-realtime` costs ~$0.01–$0.03/min vs. stacked per-service billing. |
| **Telephony ready** | Native G.711 μ-law/A-law codec support — zero transcoding for SIP trunk integration. |
| **Interruption handling** | Built-in barge-in support with automatic audio truncation and transcript reconciliation. |
| **Semantic VAD** | Model-aware turn detection understands conversational pauses vs. mid-sentence breaks. |
| **Security** | Ephemeral token pattern keeps API keys server-side; tokens auto-expire. |
| **Streaming** | True bidirectional streaming — audio plays as it generates, no waiting for full response. |

### Disadvantages / Risks

| Category | Concern |
|----------|---------|
| **Vendor lock-in** | Entire voice pipeline depends on OpenAI. No fallback if their API is down or deprecated. |
| **Model control** | Cannot swap in Claude/Bedrock for reasoning — the LLM is baked into the realtime model. |
| **Voice customization** | Limited to 8 preset voices. ElevenLabs offers voice cloning and far more variety. |
| **Maturity** | Realtime API is newer; less battle-tested than the current multi-vendor stack at scale. |
| **Pricing uncertainty** | Token-based audio pricing may shift. No long-term pricing commitments from OpenAI. |
| **Debugging** | Single black-box model is harder to debug than a pipeline where each stage is independently observable. |
| **Compliance** | Audio data flows through OpenAI — may conflict with data residency or processing requirements. |
| **Function calling latency** | Tool use adds round-trip time within the realtime session; complex tool chains may negate latency gains. |

### Feature Comparison Matrix

```
Feature                     Legacy Stack          GPT Realtime
─────────────────────────────────────────────────────────────
First-response latency      800–1500ms            250–500ms
Vendors involved            4 (Twilio, Deepgram,  1 (OpenAI)
                            ElevenLabs, Bedrock)
Voice options               ElevenLabs library    8 built-in
Voice cloning               ✅ (ElevenLabs)        ❌
LLM flexibility             ✅ (any model)         ❌ (GPT-4o only)
Telephony support           ✅ (Twilio)            ✅ (SIP trunking)
Barge-in / interruption     Custom implementation Built-in
Turn detection              Manual / silence      Semantic VAD
Streaming audio             Chunked TTS           True bidirectional
Cost (per minute, est.)     $0.05–$0.15           $0.01–$0.11
WebSocket protocol          Multiple connections  Single persistent
Function calling            Via LLM               Built-in
Audio format (browser)      Varies                PCM16 24kHz
Audio format (telephony)    G.711 via Twilio      G.711 native
```

---

## Estimated Migration Effort

### Phase 1: Core Integration (1–2 weeks)

| Task | Effort | Notes |
|------|--------|-------|
| Deploy `server.py` to production infrastructure | 2–3 days | Containerize, add health checks, configure load balancing |
| Replace Deepgram STT + ElevenLabs TTS with Realtime API | 3–4 days | Remove two service integrations, wire up single WebSocket |
| Migrate system prompt and persona configuration | 1 day | Adapt Jarvis persona instructions for realtime model format |
| Implement ephemeral token rotation in production | 1 day | Add token refresh logic, monitor expiration |

### Phase 2: Telephony Migration (1–2 weeks)

| Task | Effort | Notes |
|------|--------|-------|
| Configure SIP trunk with OpenAI (or Twilio → OpenAI relay) | 3–4 days | Choose provider (Twilio, Telnyx), configure G.711 codec |
| Migrate ConversationRelay flows to Realtime API sessions | 3–4 days | Map existing call flows to realtime session events |
| Implement call recording and logging | 1–2 days | Capture audio deltas for compliance/debugging |

### Phase 3: Tool & Context Migration (1 week)

| Task | Effort | Notes |
|------|--------|-------|
| Port existing function calls / tool definitions | 2–3 days | Convert tool schemas to Realtime API format |
| Implement conversation context injection | 1–2 days | Feed user history / CRM data into session instructions |
| Handle edge cases: timeouts, reconnection, error recovery | 2 days | Production hardening |

### Phase 4: Testing & Rollout (1–2 weeks)

| Task | Effort | Notes |
|------|--------|-------|
| A/B test latency and voice quality vs. legacy stack | 3–4 days | Measure P50/P95 latency, user satisfaction scores |
| Load testing at production scale | 2–3 days | Validate concurrent session limits, WebSocket stability |
| Staged rollout (canary → 10% → 50% → 100%) | 3–5 days | Monitor error rates, cost per call, user feedback |
| Legacy stack decommission | 1–2 days | Remove Deepgram and ElevenLabs integrations |

### Total Estimated Effort

```
Phase 1: Core Integration ............ 1–2 weeks
Phase 2: Telephony Migration ......... 1–2 weeks
Phase 3: Tool & Context Migration .... 1 week
Phase 4: Testing & Rollout ........... 1–2 weeks
──────────────────────────────────────────────────
Total .............................. 4–7 weeks
Team size .......................... 1–2 engineers
```

### Risk Mitigations

- **Keep legacy stack operational** during migration as a fallback. Route traffic back if Realtime API degrades.
- **Start with non-critical flows** (e.g., internal testing, low-traffic routes) before migrating primary voice channels.
- **Monitor cost per call closely** — token-based pricing behaves differently from per-minute billing at scale.
- **Build an abstraction layer** over the voice provider so future migrations (or hybrid setups) require minimal code changes.

---

## Quick Reference

| Resource | Location |
|----------|----------|
| API docs | [OpenAI Realtime API Guide](https://platform.openai.com/docs/guides/realtime) |
| Research notes | [`research-notes.md`](research-notes.md) |
| Migration guide | [`migration-guide.html`](migration-guide.html) (open in browser) |
| Web UI | [`index.html`](index.html) via `python server.py` → `http://localhost:8080` |
| CLI client | [`realtime_client.py`](realtime_client.py) |

---

## License

Internal prototype — not for redistribution.
