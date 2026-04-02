"""
Jarvis Transcriber — FastAPI Backend Endpoint

POST /transcribe
Accepts multipart/form-data audio uploads, converts to base64,
sends to Claude via AWS Bedrock for transcription and summary,
and returns JSON with transcript and summary fields.
"""

import base64
import os
import boto3
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Jarvis Transcriber", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
SUPPORTED_FORMATS = {".m4a", ".mp3", ".wav", ".opus", ".ogg", ".webm", ".flac"}
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-20250514-v1:0")

# Media type mapping for Bedrock's audio content blocks
MEDIA_TYPE_MAP = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".opus": "audio/ogg",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


def get_bedrock_client():
    """Create a Bedrock Runtime client using default AWS credentials."""
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension from a filename."""
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


# ---------------------------------------------------------------------------
# Transcription prompt
# ---------------------------------------------------------------------------
TRANSCRIPTION_PROMPT = """You are an expert audio transcription and analysis assistant. Listen to the provided audio and produce two outputs:

1. **TRANSCRIPT** — A full, word-for-word transcription of the audio. If you detect multiple speakers, label them (Speaker 1, Speaker 2, etc.). Include timestamps in [MM:SS] format at natural breaks (every 30-60 seconds or at speaker changes).

2. **SUMMARY** — A structured analysis with these sections:
   - MAIN TOPICS: List the main topics discussed, one per line.
   - KEY INSIGHTS: Important insights, pain points, or notable statements.
   - ACTION ITEMS: Tasks, next steps, or commitments mentioned. If none, write "None identified."
   - PEOPLE & COMPANIES MENTIONED: Names of people or companies. If none, write "None mentioned."
   - BRIEF SUMMARY: A 2-3 sentence overall summary.

Return your response as valid JSON with this exact structure:
{
  "transcript": "The full timestamped transcript as a string",
  "summary": {
    "main_topics": ["topic 1", "topic 2"],
    "key_insights": ["insight 1", "insight 2"],
    "action_items": ["action 1", "action 2"],
    "people_mentioned": ["person 1", "company 1"],
    "brief_summary": "A 2-3 sentence summary."
  }
}

Return ONLY the JSON object. No markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Accept an audio file upload, send it to Claude on Bedrock for
    transcription and analysis, and return structured JSON.
    """

    # Validate file extension
    ext = get_file_extension(file.filename or "")
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        )

    # Read file contents
    contents = await file.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents) / (1024*1024):.1f} MB). Maximum size is 25 MB.",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Convert to base64 for Bedrock
    audio_b64 = base64.b64encode(contents).decode("utf-8")
    media_type = MEDIA_TYPE_MAP.get(ext, "audio/mpeg")

    # Build the Bedrock request
    bedrock = get_bedrock_client()

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": audio_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": TRANSCRIPTION_PROMPT,
                    },
                ],
            }
        ],
    }

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )
    except bedrock.exceptions.ValidationException as e:
        raise HTTPException(status_code=400, detail=f"Bedrock validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bedrock API error: {str(e)}")

    # Parse response
    response_body = json.loads(response["body"].read())
    assistant_text = ""
    for block in response_body.get("content", []):
        if block.get("type") == "text":
            assistant_text += block["text"]

    if not assistant_text:
        raise HTTPException(status_code=502, detail="Empty response from Bedrock.")

    # Parse the JSON from Claude's response
    try:
        # Strip markdown fences if present
        cleaned = assistant_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # If Claude didn't return valid JSON, wrap the raw text
        result = {
            "transcript": assistant_text,
            "summary": {
                "main_topics": [],
                "key_insights": [],
                "action_items": [],
                "people_mentioned": [],
                "brief_summary": "Summary could not be structured automatically.",
            },
        }

    # Ensure expected keys exist
    if "transcript" not in result:
        result["transcript"] = assistant_text
    if "summary" not in result:
        result["summary"] = {}

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "jarvis-transcriber"}


# ---------------------------------------------------------------------------
# Run with: uvicorn transcribe_endpoint:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
