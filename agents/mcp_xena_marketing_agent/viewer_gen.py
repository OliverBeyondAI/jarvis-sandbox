"""
HTML Campaign Viewer Generator.

Generates a self-contained HTML file from campaign result JSON
by injecting data into the viewer.html template.
"""

from __future__ import annotations

import json
from pathlib import Path


_TEMPLATE_PATH = Path(__file__).parent / "viewer.html"


def generate_html_viewer(result_json: dict, output_path: str) -> str:
    """Generate a polished HTML viewer for the campaign result.

    Args:
        result_json: Campaign result dictionary (from agent run).
        output_path: Path where the HTML file will be saved.

    Returns:
        Absolute path to the generated HTML file.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    result_str = json.dumps(result_json, indent=2, default=str)
    html = template.replace("/* __CAMPAIGN_DATA__ */", result_str)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path.resolve())
