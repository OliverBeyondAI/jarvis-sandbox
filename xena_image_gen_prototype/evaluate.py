#!/usr/bin/env python3
"""
Comparison Evaluation Script for Xena Image Generation Prototype.

Loads generated images from both OpenAI and Google (Gemini Imagen)
output directories, analyzes them, and produces a structured Markdown report
scoring each output on:
  - Prompt adherence
  - Visual quality
  - Text rendering accuracy
  - Brand consistency

Supports two modes:
  1. Automated (default) — uses image metrics (resolution, color analysis,
     file size) and metadata inspection to produce baseline scores.
  2. LLM-assisted (--llm) — sends images to Claude for visual evaluation
     with detailed reasoning.

Usage:
    python evaluate.py                       # Automated evaluation
    python evaluate.py --llm                 # LLM-assisted evaluation (requires ANTHROPIC_API_KEY)
    python evaluate.py --output report.md    # Custom output path
    python evaluate.py --use-case social_media  # Evaluate a single use case
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Optional

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully
# ---------------------------------------------------------------------------

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from config import (
    MARKETING_PROMPTS,
    OPENAI_OUTPUT_DIR,
    GEMINI_OUTPUT_DIR,
    OUTPUT_BASE,
    XENA_SMB_PROFILE,
    get_all_use_cases,
    get_prompts_by_use_case,
    sanitize_filename,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENAI_OUTPUT = Path(OPENAI_OUTPUT_DIR)
GEMINI_OUTPUT = Path(GEMINI_OUTPUT_DIR)
DEFAULT_REPORT_PATH = Path(OUTPUT_BASE) / "evaluation_report.md"

PROVIDERS = ["openai", "gemini"]
PROVIDER_LABELS = {"openai": "OpenAI GPT Image", "gemini": "Google Gemini Imagen"}

CRITERIA = [
    ("prompt_adherence", "Prompt Adherence", "How well the image matches the described scene, objects, composition, and mood."),
    ("visual_quality", "Visual Quality", "Technical quality: resolution clarity, artifact-free rendering, lighting, and overall polish."),
    ("text_rendering", "Text Rendering Accuracy", "Accuracy of any text/typography in the image (if applicable to the prompt)."),
    ("brand_consistency", "Brand Consistency", "Alignment with Greenleaf Wellness Studio brand: colors (#4A7C59, #F5E6CC, #2C3E50), tone (warm, inviting, nature-inspired), and aesthetic."),
]

SCORE_LABELS = {1: "Poor", 2: "Below Average", 3: "Average", 4: "Good", 5: "Excellent"}

# Brand colors in RGB for analysis
BRAND_COLORS_RGB = [
    (74, 124, 89),    # #4A7C59 — forest green
    (245, 230, 204),  # #F5E6CC — warm cream
    (44, 62, 80),     # #2C3E50 — dark blue-gray
    (255, 255, 255),  # #FFFFFF — white
]


# ---------------------------------------------------------------------------
# Image analysis helpers
# ---------------------------------------------------------------------------

def load_image_info(image_path: Path) -> dict | None:
    """Load basic image info using PIL (if available) or just file stats."""
    if not image_path.exists():
        return None

    info = {
        "path": str(image_path),
        "file_size_kb": round(image_path.stat().st_size / 1024, 1),
        "exists": True,
    }

    if HAS_PIL:
        try:
            with Image.open(image_path) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["mode"] = img.mode
                info["format"] = img.format
                info["megapixels"] = round((img.width * img.height) / 1_000_000, 2)
                info["dominant_colors"] = _extract_dominant_colors(img)
                info["brand_color_proximity"] = _compute_brand_color_proximity(img)
        except Exception as e:
            info["pil_error"] = str(e)

    return info


def _extract_dominant_colors(img: "Image.Image", n: int = 5) -> list[str]:
    """Extract top-N dominant colors as hex strings."""
    small = img.copy()
    small.thumbnail((100, 100))
    small = small.convert("RGB")
    colors = small.getcolors(maxcolors=10000)
    if not colors:
        return []
    colors.sort(key=lambda c: c[0], reverse=True)
    return [f"#{r:02x}{g:02x}{b:02x}" for count, (r, g, b) in colors[:n]]


def _color_distance(c1: tuple, c2: tuple) -> float:
    """Euclidean distance between two RGB tuples."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _compute_brand_color_proximity(img: "Image.Image") -> float:
    """Score 0-100 how close the image palette is to brand colors."""
    small = img.copy()
    small.thumbnail((50, 50))
    small = small.convert("RGB")
    pixels = list(small.getdata())

    if not pixels:
        return 0.0

    total_min_dist = 0.0
    for pixel in pixels:
        min_dist = min(_color_distance(pixel, bc) for bc in BRAND_COLORS_RGB)
        total_min_dist += min_dist

    avg_dist = total_min_dist / len(pixels)
    # Max possible distance is ~441 (sqrt(255^2*3)), normalize to 0-100
    score = max(0, 100 - (avg_dist / 441 * 100))
    return round(score, 1)


def load_metadata(meta_path: Path) -> dict | None:
    """Load JSON metadata sidecar file."""
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def image_to_base64(image_path: Path) -> str | None:
    """Read image file and return base64-encoded string."""
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Automated scoring heuristics
# ---------------------------------------------------------------------------

def auto_score_image(prompt_obj, image_info: dict | None, metadata: dict | None) -> dict:
    """Produce automated scores based on available metrics."""
    scores = {}
    notes = {}

    if image_info is None:
        for key, _, _ in CRITERIA:
            scores[key] = 0
            notes[key] = "Image not found — generation may have failed."
        return {"scores": scores, "notes": notes}

    # --- Prompt Adherence (heuristic: metadata check + resolution match) ---
    pa_score = 3  # baseline
    pa_note_parts = []
    if metadata:
        if metadata.get("generation", {}).get("error"):
            pa_score = 1
            pa_note_parts.append(f"Generation error: {metadata['generation']['error']}")
        else:
            pa_score = 3
            pa_note_parts.append("Image generated successfully from enhanced prompt.")
            revised = metadata.get("generation", {}).get("revised_prompt")
            if revised:
                pa_note_parts.append(f"Provider revised the prompt (may diverge from original intent).")
                pa_score = 3
    scores["prompt_adherence"] = pa_score
    notes["prompt_adherence"] = " ".join(pa_note_parts) if pa_note_parts else "Metadata unavailable for deeper analysis."

    # --- Visual Quality (resolution, file size as proxy) ---
    vq_score = 3
    vq_parts = []
    if "width" in image_info:
        mp = image_info["megapixels"]
        if mp >= 1.5:
            vq_score = 5
            vq_parts.append(f"High resolution ({image_info['width']}×{image_info['height']}, {mp} MP).")
        elif mp >= 0.5:
            vq_score = 4
            vq_parts.append(f"Good resolution ({image_info['width']}×{image_info['height']}, {mp} MP).")
        else:
            vq_score = 3
            vq_parts.append(f"Moderate resolution ({image_info['width']}×{image_info['height']}, {mp} MP).")

    fsize = image_info.get("file_size_kb", 0)
    if fsize > 500:
        vq_parts.append(f"Rich detail ({fsize:.0f} KB).")
    elif fsize > 100:
        vq_parts.append(f"Adequate detail ({fsize:.0f} KB).")
    elif fsize > 0:
        vq_parts.append(f"Low detail/possible compression ({fsize:.0f} KB).")
        vq_score = max(2, vq_score - 1)

    scores["visual_quality"] = vq_score
    notes["visual_quality"] = " ".join(vq_parts) if vq_parts else "Unable to assess without PIL."

    # --- Text Rendering (heuristic: most prompts don't require text) ---
    has_text_requirement = any(
        kw in prompt_obj.prompt.lower()
        for kw in ["text", "typography", "letter", "word", "sign", "banner text", "headline"]
    )
    if has_text_requirement:
        scores["text_rendering"] = 3
        notes["text_rendering"] = "Prompt includes text elements — requires visual inspection for accuracy."
    else:
        scores["text_rendering"] = 4
        notes["text_rendering"] = "No explicit text rendering required by this prompt (N/A — scored favorably)."

    # --- Brand Consistency (color proximity) ---
    bc_score = 3
    bc_parts = []
    if "brand_color_proximity" in image_info:
        prox = image_info["brand_color_proximity"]
        if prox >= 75:
            bc_score = 5
            bc_parts.append(f"Strong brand color alignment (proximity: {prox}%).")
        elif prox >= 60:
            bc_score = 4
            bc_parts.append(f"Good brand color alignment (proximity: {prox}%).")
        elif prox >= 45:
            bc_score = 3
            bc_parts.append(f"Moderate brand color alignment (proximity: {prox}%).")
        else:
            bc_score = 2
            bc_parts.append(f"Weak brand color alignment (proximity: {prox}%).")
    if "dominant_colors" in image_info:
        bc_parts.append(f"Dominant colors: {', '.join(image_info['dominant_colors'][:5])}.")

    scores["brand_consistency"] = bc_score
    notes["brand_consistency"] = " ".join(bc_parts) if bc_parts else "Color analysis unavailable."

    return {"scores": scores, "notes": notes}


# ---------------------------------------------------------------------------
# LLM-assisted evaluation
# ---------------------------------------------------------------------------

def llm_score_image(client, prompt_obj, image_path: Path) -> dict:
    """Use Claude to evaluate an image against the prompt and brand guidelines."""
    b64 = image_to_base64(image_path)
    if not b64:
        return {
            "scores": {k: 0 for k, _, _ in CRITERIA},
            "notes": {k: "Image not available for LLM evaluation." for k, _, _ in CRITERIA},
        }

    eval_prompt = dedent(f"""\
        You are an expert creative director evaluating AI-generated marketing images.

        ## Business Context
        - Brand: {XENA_SMB_PROFILE.business_name}
        - Industry: {XENA_SMB_PROFILE.industry}
        - Tagline: "{XENA_SMB_PROFILE.tagline}"
        - Brand colors: {', '.join(XENA_SMB_PROFILE.brand_colors)}
        - Tone: {XENA_SMB_PROFILE.tone}
        - Target audience: {XENA_SMB_PROFILE.target_audience}

        ## Original Prompt
        {prompt_obj.prompt}

        ## Style Notes
        {prompt_obj.style_notes}

        ## Scoring Criteria
        Score each criterion from 1 (Poor) to 5 (Excellent):

        1. **Prompt Adherence** — Does the image match the described scene, objects, composition, and mood?
        2. **Visual Quality** — Technical quality: clarity, lighting, artifacts, overall polish.
        3. **Text Rendering Accuracy** — If the image contains any text, is it legible and accurate? If no text is expected, score 4 and note N/A.
        4. **Brand Consistency** — Does the image align with the brand colors, tone, and aesthetic?

        Respond in this exact JSON format (no markdown fences):
        {{
            "prompt_adherence": {{"score": <1-5>, "note": "<brief reasoning>"}},
            "visual_quality": {{"score": <1-5>, "note": "<brief reasoning>"}},
            "text_rendering": {{"score": <1-5>, "note": "<brief reasoning>"}},
            "brand_consistency": {{"score": <1-5>, "note": "<brief reasoning>"}}
        }}
    """)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": eval_prompt},
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        # Handle potential markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)

        scores = {}
        notes = {}
        for key, _, _ in CRITERIA:
            entry = result.get(key, {})
            scores[key] = min(5, max(1, int(entry.get("score", 3))))
            notes[key] = entry.get("note", "")

        return {"scores": scores, "notes": notes}

    except Exception as e:
        return {
            "scores": {k: 0 for k, _, _ in CRITERIA},
            "notes": {k: f"LLM evaluation error: {e}" for k, _, _ in CRITERIA},
        }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_evaluation_entry(prompt_obj, provider: str, eval_result: dict, image_info: dict | None) -> dict:
    """Package evaluation data for a single prompt+provider pair."""
    return {
        "provider": provider,
        "provider_label": PROVIDER_LABELS[provider],
        "title": prompt_obj.title,
        "use_case": prompt_obj.use_case,
        "prompt": prompt_obj.prompt,
        "style_notes": prompt_obj.style_notes,
        "aspect_ratio": prompt_obj.aspect_ratio,
        "image_info": image_info,
        "scores": eval_result["scores"],
        "notes": eval_result["notes"],
        "total_score": sum(v for v in eval_result["scores"].values() if v > 0),
        "avg_score": round(
            sum(v for v in eval_result["scores"].values() if v > 0)
            / max(1, sum(1 for v in eval_result["scores"].values() if v > 0)),
            2,
        ),
    }


def generate_report(evaluations: list[dict], mode: str) -> str:
    """Generate the full Markdown evaluation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# Xena Image Generation — Comparison Evaluation Report")
    lines.append("")
    lines.append(f"> **Generated:** {now}  ")
    lines.append(f"> **Evaluation mode:** {'LLM-Assisted (Claude)' if mode == 'llm' else 'Automated Metrics'}  ")
    lines.append(f"> **Business profile:** {XENA_SMB_PROFILE.business_name} — {XENA_SMB_PROFILE.industry}  ")
    lines.append(f"> **Brand colors:** {' '.join(XENA_SMB_PROFILE.brand_colors)}  ")
    lines.append(f"> **Providers compared:** {', '.join(PROVIDER_LABELS.values())}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")

    provider_totals = {p: {"score_sum": 0, "count": 0, "missing": 0} for p in PROVIDERS}
    for ev in evaluations:
        p = ev["provider"]
        if ev["image_info"] is None:
            provider_totals[p]["missing"] += 1
        else:
            provider_totals[p]["score_sum"] += ev["avg_score"]
            provider_totals[p]["count"] += 1

    lines.append("| Provider | Images Found | Images Missing | Avg Score (1–5) |")
    lines.append("|----------|:------------:|:--------------:|:---------------:|")
    for p in PROVIDERS:
        t = provider_totals[p]
        avg = round(t["score_sum"] / max(1, t["count"]), 2)
        label = PROVIDER_LABELS[p]
        lines.append(f"| {label} | {t['count']} | {t['missing']} | **{avg}** |")
    lines.append("")

    # Determine winner per criterion
    lines.append("### Criterion-Level Comparison")
    lines.append("")
    lines.append("| Criterion | " + " | ".join(PROVIDER_LABELS[p] for p in PROVIDERS) + " | Leader |")
    lines.append("|-----------|" + "|".join(":---:" for _ in PROVIDERS) + "|--------|")

    for key, label, _ in CRITERIA:
        avgs = {}
        for p in PROVIDERS:
            vals = [ev["scores"][key] for ev in evaluations if ev["provider"] == p and ev["scores"][key] > 0]
            avgs[p] = round(sum(vals) / max(1, len(vals)), 2)
        best = max(avgs, key=lambda k: avgs[k])
        leader = PROVIDER_LABELS[best] if avgs[PROVIDERS[0]] != avgs[PROVIDERS[1]] else "Tie"
        row_vals = " | ".join(f"{avgs[p]}" for p in PROVIDERS)
        lines.append(f"| {label} | {row_vals} | {leader} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # --- Detailed Per-Prompt Evaluations ---
    lines.append("## Detailed Evaluations")
    lines.append("")

    # Group by prompt title
    prompt_titles = []
    seen = set()
    for ev in evaluations:
        if ev["title"] not in seen:
            prompt_titles.append(ev["title"])
            seen.add(ev["title"])

    for idx, title in enumerate(prompt_titles, 1):
        entries = [ev for ev in evaluations if ev["title"] == title]
        if not entries:
            continue

        ref = entries[0]
        lines.append(f"### {idx}. {title}")
        lines.append("")
        lines.append(f"- **Use case:** `{ref['use_case']}`")
        lines.append(f"- **Aspect ratio:** {ref['aspect_ratio']}")
        lines.append(f"- **Style notes:** {ref['style_notes']}")
        lines.append("")
        lines.append("<details>")
        lines.append(f"<summary>Prompt (click to expand)</summary>")
        lines.append("")
        lines.append(f"> {ref['prompt']}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

        # Side-by-side image references
        lines.append("#### Side-by-Side Comparison")
        lines.append("")
        lines.append("| | " + " | ".join(PROVIDER_LABELS[ev["provider"]] for ev in entries) + " |")
        lines.append("|---|" + "|".join("---" for _ in entries) + "|")

        # Image preview row
        img_cells = []
        for ev in entries:
            if ev["image_info"] and ev["image_info"].get("exists"):
                rel_path = os.path.relpath(ev["image_info"]["path"], Path(__file__).parent / "outputs")
                img_cells.append(f"![{ev['provider']}]({rel_path})")
            else:
                img_cells.append("*Image not generated*")
        lines.append("| **Preview** | " + " | ".join(img_cells) + " |")

        # Resolution row
        res_cells = []
        for ev in entries:
            if ev["image_info"] and "width" in ev["image_info"]:
                ii = ev["image_info"]
                res_cells.append(f"{ii['width']}×{ii['height']} ({ii['megapixels']} MP)")
            elif ev["image_info"]:
                res_cells.append(f"{ev['image_info']['file_size_kb']} KB")
            else:
                res_cells.append("—")
        lines.append("| **Resolution** | " + " | ".join(res_cells) + " |")

        # File size row
        size_cells = []
        for ev in entries:
            if ev["image_info"] and ev["image_info"].get("file_size_kb"):
                size_cells.append(f"{ev['image_info']['file_size_kb']} KB")
            else:
                size_cells.append("—")
        lines.append("| **File Size** | " + " | ".join(size_cells) + " |")

        lines.append("")

        # Score table
        lines.append("#### Scores")
        lines.append("")
        lines.append("| Criterion | " + " | ".join(PROVIDER_LABELS[ev["provider"]] for ev in entries) + " |")
        lines.append("|-----------|" + "|".join(":---:" for _ in entries) + "|")

        for key, label, _ in CRITERIA:
            cells = []
            for ev in entries:
                s = ev["scores"][key]
                if s == 0:
                    cells.append("N/A")
                else:
                    emoji = "★" * s + "☆" * (5 - s)
                    cells.append(f"{emoji} ({s}/5)")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        # Total row
        total_cells = []
        for ev in entries:
            total_cells.append(f"**{ev['total_score']}/20** (avg {ev['avg_score']})")
        lines.append(f"| **Total** | " + " | ".join(total_cells) + " |")
        lines.append("")

        # Notes
        lines.append("#### Notes")
        lines.append("")
        for ev in entries:
            lines.append(f"**{PROVIDER_LABELS[ev['provider']]}:**")
            for key, label, _ in CRITERIA:
                note = ev["notes"].get(key, "—")
                if note:
                    lines.append(f"- *{label}:* {note}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # --- Scoring Rubric ---
    lines.append("## Scoring Rubric")
    lines.append("")
    lines.append("| Score | Label | Description |")
    lines.append("|:-----:|-------|-------------|")
    for score, label in SCORE_LABELS.items():
        lines.append(f"| {score} | {label} | {'★' * score + '☆' * (5 - score)} |")
    lines.append("")

    lines.append("### Criteria Definitions")
    lines.append("")
    for _, label, desc in CRITERIA:
        lines.append(f"- **{label}:** {desc}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `evaluate.py` — Xena Image Generation Prototype*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _html_escape(text: str) -> str:
    """Minimal HTML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def generate_html_report(evaluations: list[dict], mode: str) -> str:
    """Generate a self-contained HTML comparison report from real evaluation data."""
    from config import OPENAI_IMAGE_MODEL, GOOGLE_IMAGE_MODEL

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Compute provider-level stats
    provider_stats: dict[str, dict] = {}
    for p in PROVIDERS:
        entries = [e for e in evaluations if e["provider"] == p]
        found = [e for e in entries if e["image_info"] is not None]
        missing = len(entries) - len(found)
        avg = round(sum(e["avg_score"] for e in found) / max(1, len(found)), 2) if found else 0
        provider_stats[p] = {"found": len(found), "missing": missing, "avg": avg, "total": len(entries)}

    # Compute per-criterion averages
    criterion_avgs: dict[str, dict[str, float]] = {}
    for key, label, _ in CRITERIA:
        criterion_avgs[key] = {}
        for p in PROVIDERS:
            vals = [e["scores"][key] for e in evaluations if e["provider"] == p and e["scores"][key] > 0]
            criterion_avgs[key][p] = round(sum(vals) / max(1, len(vals)), 2) if vals else 0

    # Compute per-use-case averages
    use_cases = sorted(set(e["use_case"] for e in evaluations))
    uc_labels = {
        "social_media": "Social Media Posts",
        "blog_header": "Blog Headers",
        "product_announcement": "Product Announcements",
        "email_marketing": "Email Marketing",
    }

    uc_stats: dict[str, dict[str, float]] = {}
    for uc in use_cases:
        uc_stats[uc] = {}
        for p in PROVIDERS:
            entries = [e for e in evaluations if e["use_case"] == uc and e["provider"] == p and e["image_info"] is not None]
            uc_stats[uc][p] = round(sum(e["avg_score"] for e in entries) / max(1, len(entries)), 2) if entries else 0

    has_data = any(provider_stats[p]["found"] > 0 for p in PROVIDERS)
    best_provider = max(PROVIDERS, key=lambda p: provider_stats[p]["avg"]) if has_data else None
    model_names = {"openai": OPENAI_IMAGE_MODEL, "gemini": GOOGLE_IMAGE_MODEL}

    def winner_badge(p1_val: float, p2_val: float) -> str:
        if p1_val == 0 and p2_val == 0:
            return '<span class="winner-badge tie">No Data</span>'
        if p1_val == p2_val:
            return '<span class="winner-badge tie">Tie</span>'
        winner = PROVIDERS[0] if p1_val > p2_val else PROVIDERS[1]
        return f'<span class="winner-badge {winner}">{PROVIDER_LABELS[winner]}</span>'

    # Build the HTML
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Xena Image Gen — OpenAI vs Gemini Imagen Evaluation Report</title>
    <style>
        :root {{
            --brand-green: #4A7C59;
            --brand-cream: #F5E6CC;
            --brand-dark: #2C3E50;
            --brand-white: #FFFFFF;
            --openai-color: #10A37F;
            --gemini-color: #4285F4;
            --tie-color: #8B5CF6;
            --bg: #F8F9FA;
            --card-bg: #FFFFFF;
            --border: #E2E8F0;
            --text: #1A202C;
            --text-muted: #64748B;
            --warning: #F59E0B;
            --radius: 12px;
            --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
            --shadow-lg: 0 4px 24px rgba(0,0,0,0.10);
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg); color: var(--text); line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}
        .header {{
            background: linear-gradient(135deg, var(--brand-dark) 0%, #1a2a38 100%);
            color: var(--brand-white); padding: 48px 24px 40px; text-align: center;
        }}
        .header-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
            border-radius: 20px; padding: 6px 14px; font-size: 12px; font-weight: 600;
            letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 20px;
        }}
        .header h1 {{ font-size: clamp(28px, 4vw, 42px); font-weight: 700; letter-spacing: -0.5px; margin-bottom: 10px; }}
        .header p {{ font-size: 16px; opacity: 0.75; max-width: 600px; margin: 0 auto; }}
        .header-meta {{ display: flex; justify-content: center; gap: 24px; margin-top: 24px; flex-wrap: wrap; }}
        .header-meta span {{ font-size: 13px; opacity: 0.6; }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 0 20px; }}
        .section {{ margin: 40px 0; }}
        .section-title {{ font-size: 22px; font-weight: 700; color: var(--brand-dark); margin-bottom: 6px; }}
        .section-subtitle {{ color: var(--text-muted); font-size: 14px; margin-bottom: 20px; }}
        .card {{
            background: var(--card-bg); border: 1px solid var(--border);
            border-radius: var(--radius); box-shadow: var(--shadow); padding: 28px; margin-bottom: 20px;
        }}
        .notice {{
            background: #FFFBEB; border: 1px solid #FDE68A; border-radius: var(--radius);
            padding: 20px 24px; margin-bottom: 24px; font-size: 14px; line-height: 1.7;
        }}
        .notice strong {{ color: #92400E; }}
        .verdict-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
        .verdict-card {{
            background: var(--card-bg); border: 2px solid var(--border);
            border-radius: var(--radius); box-shadow: var(--shadow); padding: 28px; text-align: center;
            position: relative; transition: transform 0.15s ease;
        }}
        .verdict-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-lg); }}
        .verdict-card.recommended {{
            border-color: var(--brand-green);
            background: linear-gradient(180deg, rgba(74,124,89,0.04) 0%, var(--card-bg) 100%);
        }}
        .verdict-card.recommended::after {{
            content: 'LEADER'; position: absolute; top: -12px; left: 50%; transform: translateX(-50%);
            background: var(--brand-green); color: white; font-size: 10px; font-weight: 700;
            letter-spacing: 1px; padding: 4px 12px; border-radius: 10px;
        }}
        .verdict-logo {{
            width: 48px; height: 48px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 14px; margin: 0 auto 14px; color: white;
        }}
        .verdict-logo.openai {{ background: var(--openai-color); }}
        .verdict-logo.gemini {{ background: var(--gemini-color); }}
        .verdict-name {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; }}
        .verdict-model {{ font-size: 13px; color: var(--text-muted); margin-bottom: 16px; }}
        .verdict-score {{ font-size: 48px; font-weight: 800; letter-spacing: -2px; line-height: 1; margin-bottom: 4px; }}
        .verdict-score-label {{ font-size: 13px; color: var(--text-muted); }}
        .verdict-score.openai {{ color: var(--openai-color); }}
        .verdict-score.gemini {{ color: var(--gemini-color); }}
        .verdict-detail {{ font-size: 12px; color: var(--text-muted); margin-top: 8px; }}
        .ranking-table {{
            width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px;
        }}
        .ranking-table thead th {{
            background: var(--bg); padding: 12px 16px; text-align: left; font-weight: 600;
            font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
            color: var(--text-muted); border-bottom: 2px solid var(--border);
        }}
        .ranking-table thead th:first-child {{ border-radius: 8px 0 0 0; }}
        .ranking-table thead th:last-child {{ border-radius: 0 8px 0 0; }}
        .ranking-table tbody td {{
            padding: 14px 16px; border-bottom: 1px solid var(--border); vertical-align: middle;
        }}
        .ranking-table tbody tr:last-child td {{ border-bottom: none; }}
        .score-bar {{ display: flex; align-items: center; gap: 8px; }}
        .score-bar-track {{
            flex: 1; height: 8px; background: var(--bg);
            border-radius: 4px; overflow: hidden; min-width: 60px;
        }}
        .score-bar-fill {{ height: 100%; border-radius: 4px; }}
        .score-bar-fill.openai {{ background: var(--openai-color); }}
        .score-bar-fill.gemini {{ background: var(--gemini-color); }}
        .score-bar-value {{ font-weight: 700; font-size: 13px; min-width: 36px; text-align: right; }}
        .winner-badge {{
            display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px;
            border-radius: 12px; font-size: 11px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.3px;
        }}
        .winner-badge.openai {{ background: rgba(16,163,127,0.1); color: var(--openai-color); }}
        .winner-badge.gemini {{ background: rgba(66,133,244,0.1); color: var(--gemini-color); }}
        .winner-badge.tie {{ background: rgba(139,92,246,0.1); color: var(--tie-color); }}
        .usecase-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 20px; }}
        .usecase-card {{
            background: var(--card-bg); border: 1px solid var(--border);
            border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden;
        }}
        .usecase-header {{ padding: 20px 24px 14px; display: flex; align-items: center; gap: 12px; }}
        .usecase-title {{ font-size: 16px; font-weight: 700; }}
        .usecase-prompts {{ font-size: 12px; color: var(--text-muted); }}
        .usecase-body {{ padding: 0 24px 20px; }}
        .usecase-row {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13px;
        }}
        .usecase-row:last-child {{ border-bottom: none; }}
        .usecase-row-label {{ color: var(--text-muted); font-weight: 500; }}
        .usecase-row-value {{ font-weight: 700; }}
        .prompt-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }}
        .prompt-table thead th {{
            background: var(--bg); padding: 10px 14px; text-align: left; font-weight: 600;
            font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
            color: var(--text-muted); border-bottom: 2px solid var(--border);
        }}
        .prompt-table tbody td {{ padding: 12px 14px; border-bottom: 1px solid var(--border); }}
        .prompt-table tbody tr:last-child td {{ border-bottom: none; }}
        .footer {{
            text-align: center; padding: 40px 24px; color: var(--text-muted);
            font-size: 13px; border-top: 1px solid var(--border); margin-top: 40px;
        }}
        @media (max-width: 768px) {{
            .verdict-grid {{ grid-template-columns: 1fr; }}
            .usecase-grid {{ grid-template-columns: 1fr; }}
            .header {{ padding: 36px 16px 32px; }}
        }}
    </style>
</head>
<body>

<div class="header">
    <div class="header-badge">Xena Image Generation Prototype</div>
    <h1>OpenAI vs Gemini Imagen</h1>
    <p>Evaluation report comparing image generation models for SMB marketing workflows</p>
    <div class="header-meta">
        <span>{now}</span>
        <span>{len(evaluations) // 2} Marketing Prompts</span>
        <span>{len(use_cases)} Use Cases</span>
        <span>{_html_escape(XENA_SMB_PROFILE.business_name)}</span>
        <span>Mode: {'LLM-Assisted' if mode == 'llm' else 'Automated Metrics'}</span>
    </div>
</div>

<div class="container">
""")

    # Status notice
    if not has_data:
        html_parts.append("""
    <div class="section">
        <div class="notice">
            <strong>No images were found.</strong> The generation pipeline has not been run yet.
            To populate this report with real data, configure your API keys in <code>.env</code> and run:<br>
            <code>python generate_openai.py && python generate_gemini.py && python evaluate.py</code>
        </div>
    </div>
""")

    # Overall Verdict
    html_parts.append(f"""
    <div class="section">
        <div class="section-title">Overall Verdict</div>
        <p class="section-subtitle">Composite ranking based on {'empirical evaluation' if has_data else 'available data'} across all dimensions</p>

        <div class="verdict-grid">
""")
    for p in PROVIDERS:
        s = provider_stats[p]
        recommended = "recommended" if (best_provider == p and has_data) else ""
        html_parts.append(f"""            <div class="verdict-card {recommended}">
                <div class="verdict-logo {p}">{'OAI' if p == 'openai' else 'G'}</div>
                <div class="verdict-name">{PROVIDER_LABELS[p]}</div>
                <div class="verdict-model">{model_names[p]}</div>
                <div class="verdict-score {p}">{s['avg']}</div>
                <div class="verdict-score-label">avg score (1&ndash;5 scale)</div>
                <div class="verdict-detail">{s['found']} images evaluated, {s['missing']} missing</div>
            </div>
""")
    html_parts.append("        </div>\n    </div>\n")

    # Criterion-Level Ranking
    html_parts.append("""
    <div class="section">
        <div class="section-title">Criterion-Level Ranking</div>
        <p class="section-subtitle">Head-to-head comparison across evaluation dimensions</p>

        <div class="card" style="overflow-x:auto;">
            <table class="ranking-table">
                <thead>
                    <tr>
                        <th style="min-width:180px;">Criterion</th>
""")
    for p in PROVIDERS:
        html_parts.append(f'                        <th style="min-width:200px;">{PROVIDER_LABELS[p]}</th>\n')
    html_parts.append("                        <th style=\"min-width:100px;\">Winner</th>\n")
    html_parts.append("                    </tr>\n                </thead>\n                <tbody>\n")

    for key, label, desc in CRITERIA:
        oai_val = criterion_avgs[key]["openai"]
        gem_val = criterion_avgs[key]["gemini"]
        html_parts.append(f"""                    <tr>
                        <td><strong>{label}</strong><br><span style="font-size:11px;color:var(--text-muted);">{_html_escape(desc[:80])}</span></td>
""")
        for p in PROVIDERS:
            val = criterion_avgs[key][p]
            pct = int(val / 5 * 100) if val > 0 else 0
            color_var = f"var(--{p}-color)"
            html_parts.append(f"""                        <td>
                            <div class="score-bar">
                                <div class="score-bar-track"><div class="score-bar-fill {p}" style="width:{pct}%;"></div></div>
                                <span class="score-bar-value" style="color:{color_var};">{val}</span>
                            </div>
                        </td>
""")
        html_parts.append(f"                        <td>{winner_badge(oai_val, gem_val)}</td>\n")
        html_parts.append("                    </tr>\n")

    html_parts.append("                </tbody>\n            </table>\n        </div>\n    </div>\n")

    # Use Case Breakdown
    html_parts.append("""
    <div class="section">
        <div class="section-title">Use Case Breakdown</div>
        <p class="section-subtitle">Performance across SMB marketing categories</p>
        <div class="usecase-grid">
""")
    for uc in use_cases:
        uc_label = uc_labels.get(uc, uc.replace("_", " ").title())
        prompt_count = len([e for e in evaluations if e["use_case"] == uc and e["provider"] == PROVIDERS[0]])
        oai_val = uc_stats[uc].get("openai", 0)
        gem_val = uc_stats[uc].get("gemini", 0)
        html_parts.append(f"""            <div class="usecase-card">
                <div class="usecase-header">
                    <div><div class="usecase-title">{uc_label}</div>
                    <div class="usecase-prompts">{prompt_count} prompts</div></div>
                </div>
                <div class="usecase-body">
                    <div class="usecase-row">
                        <span class="usecase-row-label">{PROVIDER_LABELS['openai']}</span>
                        <span class="usecase-row-value" style="color:var(--openai-color);">{oai_val} / 5</span>
                    </div>
                    <div class="usecase-row">
                        <span class="usecase-row-label">{PROVIDER_LABELS['gemini']}</span>
                        <span class="usecase-row-value" style="color:var(--gemini-color);">{gem_val} / 5</span>
                    </div>
                    <div class="usecase-row">
                        <span class="usecase-row-label">Winner</span>
                        {winner_badge(oai_val, gem_val)}
                    </div>
                </div>
            </div>
""")
    html_parts.append("        </div>\n    </div>\n")

    # Per-Prompt Detail Table
    html_parts.append("""
    <div class="section">
        <div class="section-title">Per-Prompt Results</div>
        <p class="section-subtitle">Individual scores for each marketing prompt</p>

        <div class="card" style="overflow-x:auto;">
            <table class="prompt-table">
                <thead>
                    <tr>
                        <th>Prompt</th>
                        <th>Use Case</th>
                        <th>Provider</th>
                        <th>Status</th>
                        <th>Prompt Adh.</th>
                        <th>Visual Qual.</th>
                        <th>Text Rend.</th>
                        <th>Brand Cons.</th>
                        <th>Avg</th>
                    </tr>
                </thead>
                <tbody>
""")
    for ev in evaluations:
        status = "Found" if ev["image_info"] is not None else "Missing"
        status_style = "" if ev["image_info"] else "color:var(--warning);font-weight:600;"
        scores = ev["scores"]
        cells = []
        for key, _, _ in CRITERIA:
            s = scores[key]
            cells.append(f"{s}/5" if s > 0 else "&mdash;")
        avg_display = f"{ev['avg_score']}" if ev["image_info"] else "&mdash;"
        html_parts.append(f"""                    <tr>
                        <td>{_html_escape(ev['title'])}</td>
                        <td><code>{ev['use_case']}</code></td>
                        <td>{PROVIDER_LABELS[ev['provider']]}</td>
                        <td style="{status_style}">{status}</td>
                        <td>{cells[0]}</td>
                        <td>{cells[1]}</td>
                        <td>{cells[2]}</td>
                        <td>{cells[3]}</td>
                        <td><strong>{avg_display}</strong></td>
                    </tr>
""")
    html_parts.append("                </tbody>\n            </table>\n        </div>\n    </div>\n")

    # Methodology
    mode_desc = "LLM-Assisted (Claude vision)" if mode == "llm" else "Automated metrics (resolution, file size, color proximity)"
    html_parts.append(f"""
    <div class="section">
        <div class="section-title">Methodology</div>
        <div class="card">
            <div style="font-size:13px; color:var(--text-muted); line-height:1.8;">
                <p style="margin-bottom:12px;">
                    <strong>Evaluation mode:</strong> {mode_desc}
                </p>
                <p style="margin-bottom:12px;">
                    <strong>Scoring:</strong> Each image is scored 1&ndash;5 on four criteria: prompt adherence,
                    visual quality, text rendering accuracy, and brand consistency. A score of 0 indicates the
                    image was not found (generation not run or failed).
                </p>
                <p style="margin-bottom:12px;">
                    <strong>Models:</strong> OpenAI <code>{model_names['openai']}</code> and
                    Google <code>{model_names['gemini']}</code>.
                </p>
                <p>
                    <strong>Data source:</strong> All scores in this report are derived from {'actual generated images' if has_data else 'the evaluation pipeline (no images were generated yet)'}. {'Run the generation scripts with valid API keys to populate scores.' if not has_data else ''}
                </p>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">How to Run</div>
        <div class="card">
            <div style="font-size:13px; color:var(--text-muted); line-height:1.8;">
                <p style="margin-bottom:8px;"><strong>1.</strong> Set API keys in <code>.env</code>:</p>
                <pre style="background:var(--bg);padding:12px;border-radius:8px;margin-bottom:12px;font-size:12px;">OPENAI_API_KEY=sk-...
GOOGLE_GENAI_API_KEY=...</pre>
                <p style="margin-bottom:8px;"><strong>2.</strong> Generate images:</p>
                <pre style="background:var(--bg);padding:12px;border-radius:8px;margin-bottom:12px;font-size:12px;">python generate_openai.py
python generate_gemini.py</pre>
                <p style="margin-bottom:8px;"><strong>3.</strong> Evaluate and produce reports:</p>
                <pre style="background:var(--bg);padding:12px;border-radius:8px;margin-bottom:12px;font-size:12px;">python evaluate.py          # automated metrics
python evaluate.py --llm    # LLM-assisted (requires ANTHROPIC_API_KEY)</pre>
            </div>
        </div>
    </div>
""")

    html_parts.append(f"""</div>

<div class="footer">
    <p><strong>Xena Image Generation Prototype</strong> &mdash; Evaluation Report</p>
    <p style="margin-top:4px;">Generated {now} &middot; {PROVIDER_LABELS['openai']} vs {PROVIDER_LABELS['gemini']} &middot; {len(evaluations) // 2} prompts &middot; {len(use_cases)} use cases</p>
</div>

</body>
</html>""")

    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_evaluation(prompts, mode: str = "auto", output_path: Path = DEFAULT_REPORT_PATH):
    """Run full evaluation pipeline and write report."""
    print("=" * 60)
    print("Xena Image Generation — Comparison Evaluation")
    print(f"Business: {XENA_SMB_PROFILE.business_name}")
    print(f"Mode:     {'LLM-Assisted (Claude)' if mode == 'llm' else 'Automated Metrics'}")
    print(f"Prompts:  {len(prompts)}")
    print(f"Output:   {output_path}")
    print("=" * 60)

    # Initialize LLM client if needed
    llm_client = None
    if mode == "llm":
        if not HAS_ANTHROPIC:
            print("\nERROR: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\nERROR: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        llm_client = anthropic.Anthropic(api_key=api_key)

    evaluations = []
    provider_dirs = {"openai": OPENAI_OUTPUT, "gemini": GEMINI_OUTPUT}

    for i, prompt_obj in enumerate(prompts, 1):
        filename_stem = sanitize_filename(prompt_obj.title)
        print(f"\n[{i}/{len(prompts)}] {prompt_obj.title}")

        for provider in PROVIDERS:
            out_dir = provider_dirs[provider]
            image_path = out_dir / f"{filename_stem}.png"
            meta_path = out_dir / f"{filename_stem}_metadata.json"

            image_info = load_image_info(image_path)
            metadata = load_metadata(meta_path)

            status = "found" if (image_info and image_info.get("exists")) else "MISSING"
            print(f"  {PROVIDER_LABELS[provider]:30s} [{status}]", end="")

            if mode == "llm" and llm_client and image_info:
                print(" — evaluating with Claude...", end="")
                eval_result = llm_score_image(llm_client, prompt_obj, image_path)
            else:
                eval_result = auto_score_image(prompt_obj, image_info, metadata)

            entry = build_evaluation_entry(prompt_obj, provider, eval_result, image_info)
            evaluations.append(entry)

            if image_info:
                print(f"  avg={entry['avg_score']}")
            else:
                print()

    # Generate report
    print(f"\nGenerating report...")
    report_md = generate_report(evaluations, mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_md)

    # Save raw evaluation data as JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(evaluations, f, indent=2, default=str)

    # Generate HTML report from real data
    html_report = generate_html_report(evaluations, mode)
    html_path = output_path.parent / "evaluation_report.html"
    with open(html_path, "w") as f:
        f.write(html_report)

    print(f"\nReport saved to: {output_path}")
    print(f"HTML report: {html_path}")
    print(f"Raw data saved to: {json_path}")

    # Print quick summary
    print("\n" + "=" * 60)
    print("Quick Summary:")
    for provider in PROVIDERS:
        entries = [e for e in evaluations if e["provider"] == provider and e["image_info"]]
        if entries:
            avg = round(sum(e["avg_score"] for e in entries) / len(entries), 2)
            print(f"  {PROVIDER_LABELS[provider]:30s}  avg={avg}/5  images={len(entries)}")
        else:
            print(f"  {PROVIDER_LABELS[provider]:30s}  no images found")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare generated images from OpenAI and Gemini"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use Claude for visual evaluation (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Output path for Markdown report (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--use-case",
        choices=get_all_use_cases(),
        help="Evaluate only a specific use case",
    )
    parser.add_argument(
        "--list-use-cases",
        action="store_true",
        help="List available use case categories and exit",
    )
    args = parser.parse_args()

    if args.list_use_cases:
        print("Available use cases:")
        for uc in get_all_use_cases():
            count = len(get_prompts_by_use_case(uc))
            print(f"  {uc} ({count} prompts)")
        return

    prompts = get_prompts_by_use_case(args.use_case) if args.use_case else MARKETING_PROMPTS
    if not prompts:
        print(f"No prompts found for use case: {args.use_case}")
        sys.exit(1)

    mode = "llm" if args.llm else "auto"
    run_evaluation(prompts, mode=mode, output_path=args.output)


if __name__ == "__main__":
    main()
