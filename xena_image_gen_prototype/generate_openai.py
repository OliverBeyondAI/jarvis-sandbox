#!/usr/bin/env python3
"""
OpenAI Image Generator for Xena Marketing Prototype.

Generates marketing images for each prompt defined in config.py using the
OpenAI image generation API (model configured in config.py), saving images
and metadata to outputs/openai/.

Usage:
    python generate_openai.py                  # Generate all prompts
    python generate_openai.py --use-case social_media  # Filter by use case
    python generate_openai.py --dry-run        # Preview prompts without generating
"""

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    OPENAI_IMAGE_MODEL,
    OPENAI_OUTPUT_DIR,
    MARKETING_PROMPTS,
    XENA_SMB_PROFILE,
    build_enhanced_prompt,
    get_all_use_cases,
    get_prompts_by_use_case,
    sanitize_filename,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(OPENAI_OUTPUT_DIR)

# Map aspect ratios to supported sizes
ASPECT_RATIO_TO_SIZE = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "3:1": "1792x1024",   # closest supported size
    "2:1": "1792x1024",   # closest supported size
}

DEFAULT_QUALITY = "hd"
DEFAULT_STYLE = "vivid"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_metadata(output_dir: Path, filename_stem: str, prompt_obj, params: dict, generation_result: dict):
    """Save generation metadata as a JSON sidecar file."""
    metadata = {
        "title": prompt_obj.title,
        "use_case": prompt_obj.use_case,
        "prompt_original": prompt_obj.prompt,
        "prompt_enhanced": params.get("prompt", prompt_obj.prompt),
        "style_notes": prompt_obj.style_notes,
        "aspect_ratio_requested": prompt_obj.aspect_ratio,
        "model": OPENAI_IMAGE_MODEL,
        "parameters": {
            "size": params["size"],
            "quality": params["quality"],
            "style": params["style"],
        },
        "generation": {
            "revised_prompt": generation_result.get("revised_prompt"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_file": f"{filename_stem}.png",
        },
        "business_profile": {
            "name": XENA_SMB_PROFILE.business_name,
            "industry": XENA_SMB_PROFILE.industry,
            "tagline": XENA_SMB_PROFILE.tagline,
        },
    }

    meta_path = output_dir / f"{filename_stem}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(client: OpenAI, prompt_obj, output_dir: Path) -> dict:
    """Generate a single image via the OpenAI API and save it with metadata."""
    filename_stem = sanitize_filename(prompt_obj.title)
    image_path = output_dir / f"{filename_stem}.png"

    size = ASPECT_RATIO_TO_SIZE.get(prompt_obj.aspect_ratio, "1024x1024")
    enhanced_prompt = build_enhanced_prompt(prompt_obj)

    params = {
        "prompt": enhanced_prompt,
        "size": size,
        "quality": DEFAULT_QUALITY,
        "style": DEFAULT_STYLE,
    }

    print(f"  Generating: {prompt_obj.title}")
    print(f"    Model: {OPENAI_IMAGE_MODEL} | Size: {size} | Quality: {DEFAULT_QUALITY} | Style: {DEFAULT_STYLE}")

    try:
        response = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=enhanced_prompt,
            n=1,
            size=size,
            quality=DEFAULT_QUALITY,
            style=DEFAULT_STYLE,
            response_format="b64_json",
        )

        # Decode and save image
        image_data = base64.b64decode(response.data[0].b64_json)
        with open(image_path, "wb") as f:
            f.write(image_data)

        revised_prompt = response.data[0].revised_prompt or ""
        generation_result = {"revised_prompt": revised_prompt}

        meta_path = save_metadata(output_dir, filename_stem, prompt_obj, params, generation_result)

        print(f"    Saved: {image_path.name}")
        print(f"    Metadata: {meta_path.name}")
        if revised_prompt:
            print(f"    Revised prompt: {revised_prompt[:100]}...")

        return {
            "status": "success",
            "title": prompt_obj.title,
            "image_path": str(image_path),
            "metadata_path": str(meta_path),
        }

    except Exception as e:
        print(f"    ERROR: {e}")
        generation_result = {"revised_prompt": None, "error": str(e)}
        save_metadata(output_dir, filename_stem, prompt_obj, params, generation_result)
        return {
            "status": "error",
            "title": prompt_obj.title,
            "error": str(e),
        }


def run_generation(prompts, dry_run: bool = False):
    """Run image generation for a list of prompts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Xena Image Generator — OpenAI ({OPENAI_IMAGE_MODEL})")
    print(f"Business: {XENA_SMB_PROFILE.business_name}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"Prompts:  {len(prompts)}")
    print(f"Mode:     {'DRY RUN' if dry_run else 'LIVE GENERATION'}")
    print("=" * 60)

    if dry_run:
        for i, p in enumerate(prompts, 1):
            size = ASPECT_RATIO_TO_SIZE.get(p.aspect_ratio, "1024x1024")
            print(f"\n[{i}/{len(prompts)}] {p.title}")
            print(f"  Use case:    {p.use_case}")
            print(f"  Aspect:      {p.aspect_ratio} -> {size}")
            print(f"  Style notes: {p.style_notes}")
            print(f"  Prompt:      {p.prompt[:120]}...")
        print(f"\nDry run complete. {len(prompts)} prompts would be generated.")
        return

    if not OPENAI_API_KEY:
        print("\nERROR: OPENAI_API_KEY is not set.")
        print("Set it in your environment or in a .env file at the project root.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []
    start_time = time.time()

    for i, prompt_obj in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}]")
        result = generate_image(client, prompt_obj, OUTPUT_DIR)
        results.append(result)

        # Rate-limit courtesy pause between requests
        if i < len(prompts):
            time.sleep(1)

    elapsed = time.time() - start_time
    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "error")

    # Save run summary
    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "model": OPENAI_IMAGE_MODEL,
        "total_prompts": len(prompts),
        "successes": successes,
        "failures": failures,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    summary_path = OUTPUT_DIR / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Generation complete in {elapsed:.1f}s")
    print(f"  Success: {successes}/{len(prompts)}")
    print(f"  Failed:  {failures}/{len(prompts)}")
    print(f"  Summary: {summary_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=f"Generate marketing images using OpenAI ({OPENAI_IMAGE_MODEL})"
    )
    parser.add_argument(
        "--use-case",
        choices=get_all_use_cases(),
        help="Only generate images for a specific use case",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without making API calls",
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

    run_generation(prompts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
