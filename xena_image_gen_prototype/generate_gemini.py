#!/usr/bin/env python3
"""
Google Gemini Imagen Image Generator for Xena Marketing Prototype.

Generates marketing images for each prompt defined in config.py using the
Google Gemini Imagen API, saving images and metadata to outputs/gemini/.

Usage:
    python generate_gemini.py                  # Generate all prompts
    python generate_gemini.py --use-case social_media  # Filter by use case
    python generate_gemini.py --dry-run        # Preview prompts without generating
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

from config import (
    GEMINI_OUTPUT_DIR,
    GOOGLE_GENAI_API_KEY,
    GOOGLE_IMAGE_MODEL,
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

OUTPUT_DIR = Path(GEMINI_OUTPUT_DIR)

# Map aspect ratios to Imagen supported ratios
ASPECT_RATIO_MAP = {
    "1:1": "1:1",
    "16:9": "16:9",
    "3:1": "16:9",   # closest supported ratio
    "2:1": "16:9",   # closest supported ratio
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_metadata(
    output_dir: Path,
    filename_stem: str,
    prompt_obj,
    params: dict,
    generation_result: dict,
):
    """Save generation metadata as a JSON sidecar file."""
    metadata = {
        "title": prompt_obj.title,
        "use_case": prompt_obj.use_case,
        "prompt_original": prompt_obj.prompt,
        "prompt_enhanced": params.get("prompt", prompt_obj.prompt),
        "style_notes": prompt_obj.style_notes,
        "aspect_ratio_requested": prompt_obj.aspect_ratio,
        "aspect_ratio_mapped": params.get("aspect_ratio"),
        "model": GOOGLE_IMAGE_MODEL,
        "parameters": {
            "aspect_ratio": params.get("aspect_ratio"),
            "number_of_images": params.get("number_of_images", 1),
        },
        "generation": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_file": f"{filename_stem}.png",
            "mime_type": generation_result.get("mime_type"),
        },
        "business_profile": {
            "name": XENA_SMB_PROFILE.business_name,
            "industry": XENA_SMB_PROFILE.industry,
            "tagline": XENA_SMB_PROFILE.tagline,
        },
    }

    if "error" in generation_result:
        metadata["generation"]["error"] = generation_result["error"]

    meta_path = output_dir / f"{filename_stem}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(client: genai.Client, prompt_obj, output_dir: Path) -> dict:
    """Generate a single image via Gemini Imagen and save it with metadata."""
    filename_stem = sanitize_filename(prompt_obj.title)
    image_path = output_dir / f"{filename_stem}.png"

    aspect_ratio = ASPECT_RATIO_MAP.get(prompt_obj.aspect_ratio, "1:1")
    enhanced_prompt = build_enhanced_prompt(prompt_obj)

    params = {
        "prompt": enhanced_prompt,
        "aspect_ratio": aspect_ratio,
        "number_of_images": 1,
    }

    print(f"  Generating: {prompt_obj.title}")
    print(f"    Model: {GOOGLE_IMAGE_MODEL} | Aspect ratio: {aspect_ratio}")

    try:
        response = client.models.generate_images(
            model=GOOGLE_IMAGE_MODEL,
            prompt=enhanced_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            ),
        )

        if not response.generated_images:
            raise RuntimeError("No images returned by the API")

        image = response.generated_images[0].image
        image_bytes = image.image_bytes
        mime_type = getattr(image, "mime_type", "image/png")

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        generation_result = {"mime_type": mime_type}

        meta_path = save_metadata(
            output_dir, filename_stem, prompt_obj, params, generation_result
        )

        print(f"    Saved: {image_path.name}")
        print(f"    Metadata: {meta_path.name}")

        return {
            "status": "success",
            "title": prompt_obj.title,
            "image_path": str(image_path),
            "metadata_path": str(meta_path),
        }

    except Exception as e:
        print(f"    ERROR: {e}")
        generation_result = {"mime_type": None, "error": str(e)}
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
    print("Xena Image Generator — Google Gemini Imagen")
    print(f"Business: {XENA_SMB_PROFILE.business_name}")
    print(f"Model:    {GOOGLE_IMAGE_MODEL}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"Prompts:  {len(prompts)}")
    print(f"Mode:     {'DRY RUN' if dry_run else 'LIVE GENERATION'}")
    print("=" * 60)

    if dry_run:
        for i, p in enumerate(prompts, 1):
            ratio = ASPECT_RATIO_MAP.get(p.aspect_ratio, "1:1")
            print(f"\n[{i}/{len(prompts)}] {p.title}")
            print(f"  Use case:    {p.use_case}")
            print(f"  Aspect:      {p.aspect_ratio} -> {ratio}")
            print(f"  Style notes: {p.style_notes}")
            print(f"  Prompt:      {p.prompt[:120]}...")
        print(f"\nDry run complete. {len(prompts)} prompts would be generated.")
        return

    if not GOOGLE_GENAI_API_KEY:
        print("\nERROR: GOOGLE_GENAI_API_KEY is not set.")
        print("Set it in your environment or in a .env file at the project root.")
        sys.exit(1)

    client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)
    results = []
    start_time = time.time()

    for i, prompt_obj in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}]")
        result = generate_image(client, prompt_obj, OUTPUT_DIR)
        results.append(result)

        if i < len(prompts):
            time.sleep(2)

    elapsed = time.time() - start_time
    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "error")

    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "model": GOOGLE_IMAGE_MODEL,
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
        description="Generate marketing images using Google Gemini Imagen"
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

    prompts = (
        get_prompts_by_use_case(args.use_case) if args.use_case else MARKETING_PROMPTS
    )
    if not prompts:
        print(f"No prompts found for use case: {args.use_case}")
        sys.exit(1)

    run_generation(prompts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
