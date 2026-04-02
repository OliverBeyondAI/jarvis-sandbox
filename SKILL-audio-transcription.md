# SKILL: Audio Transcription

**Skill ID:** `audio-transcription`
**Version:** 1.0
**Last Updated:** 2026-04-02

---

## Trigger Conditions

This skill activates when:

- Oliver sends an audio file with one of the supported formats: `.m4a`, `.mp3`, `.wav`, `.opus`, `.ogg`
- Oliver sends a message like:
  - "transcribe this"
  - "what did they say"
  - "summarize this audio"
  - Any similar request referencing an attached or recently sent audio file

---

## Skill Steps

### Step 1: Receive Audio

- Accept the audio file URL or attachment from the incoming message (WhatsApp, or any other channel).
- Validate file format is supported (`.m4a`, `.mp3`, `.wav`, `.opus`, `.ogg`).
- Validate file size is under **25 MB** (Whisper API limit).
  - If the file exceeds 25 MB, respond to Oliver:
    > "This audio file is too large for transcription (max 25 MB). Can you compress it or split it into smaller parts?"

### Step 2: Transcribe with Whisper

- Call the **OpenAI Whisper API** (`whisper-1`) to get a full word-for-word transcription with timestamps.
- **Endpoint:** `https://api.openai.com/v1/audio/transcriptions`
- **Parameters:**
  - `model`: `whisper-1`
  - `response_format`: `verbose_json` (to get word-level timestamps)
  - `timestamp_granularities`: `["segment"]`
- **Authentication:** Bearer token via `OPENAI_API_KEY` environment variable.

### Step 3: Analyze with Claude

- Send the raw transcript to the **Claude API** to generate:
  - **Speaker identification** — label distinct speakers (Speaker 1, Speaker 2, etc.) based on context clues, tone shifts, and conversational patterns.
  - **Structured summary** containing:
    - Main Topics
    - Key Insights
    - Action Items
    - People / Companies Mentioned

### Step 4: Return Output

- Send **two outputs** back to Oliver in a single message (formatted for WhatsApp readability).

---

## Output Format

```
*Full Transcript*
[00:00] Speaker 1: ...
[00:30] Speaker 2: ...
[01:15] Speaker 1: ...

---

*Summary*
• Main Topics: ...
• Key Insights: ...
• Action Items: ...
• People Mentioned: ...
```

### Formatting Rules

- Use WhatsApp-compatible markdown: `*bold*` for headers, `•` for bullet points.
- Timestamps in `[MM:SS]` format.
- If the transcript is very long (>4000 chars), split the full transcript and summary into separate messages.

---

## Integration Notes

### New Tool Required

```
transcribe_audio(file_url: str) -> dict
```

- **Location:** Add to the Jarvis Fargate server tools module.
- **Returns:**
  ```json
  {
    "transcript": "Full timestamped transcript string",
    "summary": {
      "main_topics": ["..."],
      "key_insights": ["..."],
      "action_items": ["..."],
      "people_mentioned": ["..."]
    },
    "duration_seconds": 180,
    "speaker_count": 2
  }
  ```

### Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Authentication for Whisper API |

### Constraints

- **Max file size:** 25 MB (Whisper API hard limit)
- **Supported formats:** `.m4a`, `.mp3`, `.wav`, `.opus`, `.ogg`
- **Whisper API endpoint:** `https://api.openai.com/v1/audio/transcriptions`

---

## Error Handling

| Scenario | Response to Oliver |
|---|---|
| File too large (>25 MB) | "This file is too large for transcription (max 25 MB). Please compress or split it." |
| Unsupported format | "I can't transcribe this file type. Supported formats: m4a, mp3, wav, opus, ogg." |
| Whisper API failure | "Transcription failed — I'll retry in a moment." (auto-retry once, then escalate) |
| Empty/silent audio | "The audio appears to be empty or silent. Can you check the file?" |

---

## Future Enhancements

- **Auto-save transcripts** to DynamoDB linked to a project.
- **Auto-tag to project** — e.g., if Roman sends audio, auto-tag to the ophthalmology project.
- **Auto-forward summary** to relevant project notes automatically.
- **Language detection** — detect non-English audio and translate.
- **Real-time transcription** for voice notes as they arrive.
