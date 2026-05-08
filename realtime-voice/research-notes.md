# OpenAI Realtime API — Research Notes

> Researched: 2026-05-08
> Sources: OpenAI Python SDK (v2.35+), OpenAI API reference, openai-realtime-api-beta repo

---

## 1. SIP Calling Capability

OpenAI introduced **SIP (Session Initiation Protocol) connectivity** for the Realtime API, enabling direct phone-call integration without building a separate telephony bridge.

### How It Works
- OpenAI provisions a **SIP trunk endpoint** that can receive inbound calls or be connected via an outbound SIP INVITE from your telephony provider (e.g., Twilio, Vonage, Telnyx).
- The SIP trunk bridges the PSTN phone call directly into a Realtime API session — audio flows bidirectionally between the caller and the model.
- Audio codec support includes **G.711 μ-law (`g711_ulaw`)** and **G.711 A-law (`g711_alaw`)**, which are the standard telephony codecs. These are natively supported as `input_audio_format` and `output_audio_format` options in the session configuration, specifically to support telephony integrations.

### Integration Pattern
1. **Provision a phone number** via a telephony provider (Twilio, Telnyx, etc.).
2. **Configure SIP trunking** on the provider to route calls to the OpenAI SIP endpoint.
3. **Create a Realtime API session** (via REST) to obtain a session token / connection config.
4. **Bridge the call** — the telephony provider forwards the SIP INVITE, and the Realtime API session handles the conversation.

### Key Details
- G.711 codec support (`g711_ulaw`, `g711_alaw`) avoids transcoding overhead, keeping latency low.
- Works with server VAD or semantic VAD for natural turn-taking over phone.
- Function calling / tool use is fully available during phone calls, enabling IVR-like workflows.

---

## 2. WebSocket Connection Approach

The Realtime API uses a **persistent WebSocket connection** for bidirectional, event-driven communication.

### Connection Endpoint
```
wss://api.openai.com/v1/realtime?model=gpt-realtime
```

The base URL mirrors the REST API base (`https://api.openai.com`) with `wss://` replacing `https://`.

### Two Connection Methods

#### A. Direct WebSocket (Server-Side)
For server-side applications using your API key directly:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async with client.realtime.connect(model="gpt-realtime") as connection:
    await connection.session.update(
        session={"modalities": ["text", "audio"], "voice": "alloy"}
    )
    # Send and receive events...
    async for event in connection:
        print(event.type)
```

The SDK uses the `websockets` library under the hood and handles authentication automatically via the `Authorization: Bearer <API_KEY>` header.

#### B. Ephemeral Token (Client-Side / Browser)
For browser or mobile clients, create a short-lived token server-side, then connect from the client:

**Step 1 — Server creates an ephemeral token:**
```python
from openai import OpenAI
client = OpenAI()

session = client.beta.realtime.sessions.create(
    model="gpt-realtime",
    voice="alloy",
    modalities=["text", "audio"],
    instructions="You are a helpful assistant.",
    input_audio_format="pcm16",
    output_audio_format="pcm16",
)
ephemeral_token = session.client_secret.value
```

**Step 2 — Client connects using the ephemeral token:**
```javascript
const ws = new WebSocket(
  "wss://api.openai.com/v1/realtime?model=gpt-realtime",
  { headers: { "Authorization": `Bearer ${ephemeralToken}` } }
);
```

### Event Architecture
Communication follows a **client-event / server-event** model:

| Direction | Example Events |
|-----------|---------------|
| Client → Server | `session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`, `conversation.item.create`, `response.create` |
| Server → Client | `session.created`, `session.updated`, `response.audio.delta`, `response.audio_transcript.delta`, `response.done`, `error` |

### Audio Format
- **PCM16**: 16-bit PCM, 24 kHz sample rate, mono, little-endian (default for non-telephony)
- **G.711 μ-law / A-law**: 8 kHz, standard telephony codecs

### Turn Detection
Two modes for detecting when the user has finished speaking:

| Mode | Description |
|------|-------------|
| `server_vad` | Volume-based voice activity detection. Configurable `threshold` (0.0–1.0), `prefix_padding_ms`, `silence_duration_ms`. |
| `semantic_vad` | Uses a turn-detection model to estimate whether the user has finished speaking. Configurable `eagerness`: `low`, `medium`, `high`, `auto`. More natural but slightly higher latency. |

Turn detection can be disabled entirely (`null`), requiring manual `response.create` triggers.

---

## 3. Available Voices

The Realtime API offers **8 built-in voices** plus support for custom/arbitrary voice IDs:

| Voice | Character |
|-------|-----------|
| **alloy** | Neutral, balanced — good default |
| **ash** | Warm, conversational |
| **ballad** | Gentle, melodic |
| **coral** | Clear, friendly |
| **echo** | Smooth, resonant |
| **sage** | Calm, authoritative |
| **shimmer** | Bright, energetic |
| **verse** | Expressive, dynamic |

### Voice Behavior
- Voices have **natural inflection** and can laugh, whisper, and follow tone direction via instructions.
- Voice is set at session creation and is the **one parameter that cannot be changed** via `session.update` after initialization.
- **Speed control**: Adjustable between `0.25` (slow) and `1.5` (fast), default `1.0`. Can only be changed between model turns, not mid-response.
- The `voice` parameter also accepts arbitrary strings (for custom voice IDs), not just the 8 built-in options.

---

## 4. Authentication Method

### API Key Authentication
The primary authentication is via **Bearer token** in the WebSocket handshake:

```
Authorization: Bearer sk-...
```

This is handled automatically by the official SDKs when you provide your API key.

### Ephemeral Client Tokens
For client-side (browser/mobile) use, OpenAI provides a **session-based ephemeral token** flow:

1. **Server-side**: Call `POST /v1/realtime/sessions` with your API key to create a session and receive a `client_secret`.
2. **Client-side**: Use the `client_secret.value` as the Bearer token to open the WebSocket.
3. The token is **short-lived** (configurable expiry via `client_secret.expires_after`), preventing key exposure.

```python
# Server-side: create ephemeral token
session = client.beta.realtime.sessions.create(
    model="gpt-realtime",
    client_secret={"expires_after": {"anchor": "created", "minutes": 5}},
)
token = session.client_secret.value  # Send this to the client
```

### Organization & Project Scoping
Standard OpenAI headers apply:
- `OpenAI-Organization: org-...`
- `OpenAI-Project: proj-...`

### Azure OpenAI
The Realtime API is also available on Azure OpenAI Service, using Azure AD / managed identity authentication with the same WebSocket protocol.

---

## 5. Pricing

### Model Tiers and Token-Based Pricing

The Realtime API is priced per **token** (not per minute), since the model processes audio natively as tokens.

#### gpt-4o-realtime-preview (and `gpt-realtime` alias)
| Component | Price |
|-----------|-------|
| **Text input** | $5.00 / 1M tokens |
| **Text output** | $20.00 / 1M tokens |
| **Audio input** | $40.00 / 1M tokens |
| **Audio output** | $80.00 / 1M tokens |

#### gpt-4o-mini-realtime-preview
| Component | Price |
|-----------|-------|
| **Text input** | $0.60 / 1M tokens |
| **Text output** | $2.40 / 1M tokens |
| **Audio input** | $10.00 / 1M tokens |
| **Audio output** | $20.00 / 1M tokens |

### Audio Token Rate
- Approximately **1 second of audio ≈ 32–50 tokens** (varies by content density).
- A typical 1-minute conversation (30s user + 30s model) costs roughly:
  - **gpt-4o-realtime**: ~$0.06–$0.11
  - **gpt-4o-mini-realtime**: ~$0.01–$0.03

### Additional Costs
- **Input audio transcription** (Whisper/GPT-4o-transcribe): Billed separately per the Audio transcription API pricing.
- **Cached audio input tokens**: 50% discount (same as text cached tokens).
- No per-session or per-connection fee — you pay only for tokens consumed.

---

## 6. Supported Models

| Model ID | Notes |
|----------|-------|
| `gpt-realtime` | Latest stable alias (currently points to `gpt-realtime-2025-08-28`) |
| `gpt-realtime-2025-08-28` | Pinned version of the GA model |
| `gpt-realtime-1.5` | Next-gen model (added ~Feb 2026), improved quality and latency |
| `gpt-4o-realtime-preview` | Original preview model |
| `gpt-4o-realtime-preview-2024-10-01` | First preview snapshot |
| `gpt-4o-realtime-preview-2024-12-17` | December 2024 preview update |
| `gpt-4o-realtime-preview-2025-06-03` | June 2025 preview update |
| `gpt-4o-mini-realtime-preview` | Smaller, cheaper, faster model |
| `gpt-4o-mini-realtime-preview-2024-12-17` | Mini model snapshot |

The model is set at connection time and **cannot be changed** during a session.

---

## 7. Additional Capabilities

### Function Calling / Tool Use
- Full function-calling support with the same schema as Chat Completions.
- Tools are defined in `session.update` or at connection time.
- `tool_choice`: `auto`, `none`, `required`, or a specific function name.

### Input Audio Noise Reduction
- Built-in noise reduction filter that processes audio before VAD and model inference.
- Improves accuracy in noisy environments (e.g., phone calls, public spaces).
- Can be toggled on/off per session.

### Tracing
- Built-in tracing support for debugging and monitoring.
- Configurable `workflow_name`, `group_id`, and `metadata` for dashboard filtering.

### Multimodal Output
- Model can produce **text and audio simultaneously**.
- Text output is useful for moderation, logging, and accessibility.
- Audio is streamed faster than real-time to ensure smooth playback.

### Conversation History
- The API maintains a **server-side conversation state**.
- Items can be added, deleted, or truncated programmatically.
- Supports injecting system messages, user messages, and assistant messages mid-conversation.

---

## 8. Quick-Start Code Example

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()

    async with client.realtime.connect(model="gpt-realtime") as conn:
        # Configure session
        await conn.session.update(session={
            "modalities": ["text", "audio"],
            "voice": "coral",
            "instructions": "You are a friendly phone assistant.",
            "input_audio_format": "g711_ulaw",  # for telephony
            "output_audio_format": "g711_ulaw",
            "turn_detection": {"type": "semantic_vad", "eagerness": "medium"},
            "input_audio_noise_reduction": {"type": "near_field"},
            "temperature": 0.8,
        })

        # Send a text message
        await conn.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello, how can you help me today?"}],
        })
        await conn.response.create()

        # Process events
        async for event in conn:
            if event.type == "response.audio_transcript.done":
                print(f"Assistant said: {event.transcript}")
            elif event.type == "response.done":
                break
            elif event.type == "error":
                print(f"Error: {event.error.message}")
                break

asyncio.run(main())
```

---

## 9. Key Takeaways for Integration

1. **SIP/Telephony**: Use `g711_ulaw`/`g711_alaw` audio formats + a SIP trunk provider (Twilio, Telnyx) to connect phone calls directly to the Realtime API.
2. **WebSocket-first**: All communication is event-driven over a persistent WebSocket — no polling, no REST calls during a session.
3. **Ephemeral tokens for clients**: Never expose your API key in browsers; use the sessions endpoint to mint short-lived tokens.
4. **Semantic VAD**: Prefer `semantic_vad` over `server_vad` for more natural phone conversations — it understands pauses vs. actual turn endings.
5. **Cost control**: Use `gpt-4o-mini-realtime-preview` for cost-sensitive applications (4–5x cheaper). Set `max_response_output_tokens` to cap response length.
6. **Voice is immutable**: Choose the right voice at session creation — it cannot be changed mid-session.
