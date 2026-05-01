#!/usr/bin/env python3
"""
Structured Output Formatter — Jarvis-Style Research Summaries

Compiles gathered research into polished summaries with key findings,
source comparisons, and actionable recommendations.

Supports three output formats:
  - terminal: Rich ANSI-colored output for interactive CLI use
  - markdown: Clean markdown for reports and documentation
  - json: Machine-readable structured JSON
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .agent import SummaryResult


# ---------------------------------------------------------------------------
# ANSI color helpers (no external dependencies)
# ---------------------------------------------------------------------------

class _C:
    """ANSI escape codes for terminal styling."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Jarvis palette
    CYAN = "\033[38;5;45m"
    BLUE = "\033[38;5;33m"
    GREEN = "\033[38;5;42m"
    YELLOW = "\033[38;5;220m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[38;5;196m"
    MAGENTA = "\033[38;5;177m"
    WHITE = "\033[38;5;255m"
    GRAY = "\033[38;5;245m"
    DARK_GRAY = "\033[38;5;238m"

    # Backgrounds
    BG_BLUE = "\033[48;5;17m"
    BG_DARK = "\033[48;5;233m"


def _styled(text: str, *styles: str) -> str:
    """Apply ANSI styles to text."""
    return "".join(styles) + text + _C.RESET


# ---------------------------------------------------------------------------
# Terminal Formatter
# ---------------------------------------------------------------------------

_WIDTH = 72
_DIVIDER = "━" * _WIDTH
_THIN_DIVIDER = "─" * _WIDTH


def format_terminal(result: SummaryResult) -> str:
    """Format a SummaryResult as a rich ANSI-colored terminal output."""
    lines: list[str] = []
    _a = lines.append  # shorthand

    # ── Header ──
    _a("")
    _a(_styled(_DIVIDER, _C.CYAN))
    _a(_styled("  ◆  JARVIS RESEARCH BRIEFING", _C.BOLD, _C.CYAN))
    _a(_styled(_DIVIDER, _C.CYAN))
    _a(_styled(f"  Generated: {result.timestamp}", _C.GRAY))
    _a(_styled(f"  Sources analyzed: {len(result.sources)}", _C.GRAY))
    _a("")

    # ── Sources ──
    if result.sources:
        for i, source in enumerate(result.sources, 1):
            title = source.get("title", "Untitled Source")
            url = source.get("url", source.get("source", "N/A"))

            _a(_styled(_THIN_DIVIDER, _C.DARK_GRAY))
            _a(f"  {_styled(f'SOURCE {i}', _C.BOLD, _C.BLUE)}  {_styled(title, _C.BOLD, _C.WHITE)}")
            _a(f"  {_styled(url, _C.DIM, _C.CYAN)}")
            _a("")

            # Key Findings
            findings = source.get("key_findings", [])
            if findings:
                _a(f"  {_styled('Key Findings', _C.BOLD, _C.GREEN)}")
                for j, finding in enumerate(findings, 1):
                    wrapped = textwrap.fill(
                        finding, width=_WIDTH - 8,
                        initial_indent=f"    {_styled('▸', _C.GREEN)} ",
                        subsequent_indent="      ",
                    )
                    _a(wrapped)
                _a("")

            # Methodology
            methodology = source.get("methodology", "")
            if methodology:
                _a(f"  {_styled('Methodology', _C.BOLD, _C.YELLOW)}")
                wrapped = textwrap.fill(
                    methodology, width=_WIDTH - 6,
                    initial_indent="    ", subsequent_indent="    ",
                )
                _a(wrapped)
                _a("")

            # Relevance
            relevance = source.get("relevance", "")
            if relevance:
                _a(f"  {_styled('Relevance', _C.BOLD, _C.MAGENTA)}")
                wrapped = textwrap.fill(
                    relevance, width=_WIDTH - 6,
                    initial_indent="    ", subsequent_indent="    ",
                )
                _a(wrapped)
                _a("")

    # ── Cross-Source Synthesis ──
    if result.synthesis:
        _a(_styled(_THIN_DIVIDER, _C.DARK_GRAY))
        _a(f"  {_styled('◈  CROSS-SOURCE SYNTHESIS', _C.BOLD, _C.ORANGE)}")
        _a("")
        wrapped = textwrap.fill(
            result.synthesis, width=_WIDTH - 6,
            initial_indent="    ", subsequent_indent="    ",
        )
        _a(wrapped)
        _a("")

    # ── Key Takeaways ──
    if result.key_takeaways:
        _a(_styled(_THIN_DIVIDER, _C.DARK_GRAY))
        _a(f"  {_styled('◈  KEY TAKEAWAYS', _C.BOLD, _C.GREEN)}")
        _a("")
        for i, takeaway in enumerate(result.key_takeaways, 1):
            num_style = _styled(f"  {i}.", _C.BOLD, _C.GREEN)
            wrapped = textwrap.fill(
                takeaway, width=_WIDTH - 8,
                initial_indent=f"  {num_style} ",
                subsequent_indent="      ",
            )
            _a(wrapped)
        _a("")

    # ── Suggested Follow-Up ──
    if result.follow_up:
        _a(_styled(_THIN_DIVIDER, _C.DARK_GRAY))
        _a(f"  {_styled('◈  RECOMMENDED FOLLOW-UP', _C.BOLD, _C.YELLOW)}")
        _a("")
        for q in result.follow_up:
            wrapped = textwrap.fill(
                q, width=_WIDTH - 8,
                initial_indent=f"    {_styled('→', _C.YELLOW)} ",
                subsequent_indent="      ",
            )
            _a(wrapped)
        _a("")

    # ── Raw fallback ──
    if not result.sources and result.raw_response:
        _a(_styled(_THIN_DIVIDER, _C.DARK_GRAY))
        _a(f"  {_styled('Raw Agent Output:', _C.BOLD, _C.RED)}")
        _a("")
        for line in result.raw_response.splitlines():
            _a(f"    {line}")
        _a("")

    # ── Footer ──
    _a(_styled(_DIVIDER, _C.CYAN))
    _a(_styled("  ◆  End of Jarvis Research Briefing", _C.DIM, _C.CYAN))
    _a(_styled(_DIVIDER, _C.CYAN))
    _a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown Formatter
# ---------------------------------------------------------------------------

def format_markdown(result: SummaryResult) -> str:
    """Format a SummaryResult as clean Markdown."""
    lines: list[str] = []
    _a = lines.append

    _a("# Jarvis Research Briefing")
    _a("")
    _a(f"**Generated:** {result.timestamp}  ")
    _a(f"**Sources analyzed:** {len(result.sources)}")
    _a("")
    _a("---")

    if result.sources:
        for i, source in enumerate(result.sources, 1):
            title = source.get("title", "Untitled Source")
            url = source.get("url", source.get("source", "N/A"))

            _a("")
            _a(f"## Source {i}: {title}")
            _a("")
            _a(f"**URL:** [{url}]({url})")
            _a("")

            findings = source.get("key_findings", [])
            if findings:
                _a("### Key Findings")
                _a("")
                for finding in findings:
                    _a(f"- {finding}")
                _a("")

            methodology = source.get("methodology", "")
            if methodology:
                _a(f"**Methodology:** {methodology}")
                _a("")

            relevance = source.get("relevance", "")
            if relevance:
                _a(f"**Relevance:** {relevance}")
                _a("")

            _a("---")

    if result.synthesis:
        _a("")
        _a("## Cross-Source Synthesis")
        _a("")
        _a(result.synthesis)
        _a("")

    if result.key_takeaways:
        _a("## Key Takeaways")
        _a("")
        for i, takeaway in enumerate(result.key_takeaways, 1):
            _a(f"{i}. {takeaway}")
        _a("")

    if result.follow_up:
        _a("## Recommended Follow-Up")
        _a("")
        for q in result.follow_up:
            _a(f"- {q}")
        _a("")

    if not result.sources and result.raw_response:
        _a("## Raw Output")
        _a("")
        _a("```")
        _a(result.raw_response)
        _a("```")
        _a("")

    _a("---")
    _a("")
    _a("*Generated by Jarvis Research Summarizer Agent*")
    _a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

def format_json(result: SummaryResult, pretty: bool = True) -> str:
    """Format a SummaryResult as structured JSON."""
    output = {
        "meta": {
            "generator": "Jarvis Research Summarizer Agent",
            "timestamp": result.timestamp,
            "source_count": len(result.sources),
        },
        "sources": result.sources,
        "cross_source_synthesis": result.synthesis,
        "key_takeaways": result.key_takeaways,
        "suggested_follow_up": result.follow_up,
    }
    if not result.sources and result.raw_response:
        output["raw_response"] = result.raw_response

    if pretty:
        return json.dumps(output, indent=2, ensure_ascii=False, default=str)
    return json.dumps(output, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Unified format dispatcher
# ---------------------------------------------------------------------------

FORMATS = {
    "terminal": format_terminal,
    "markdown": format_markdown,
    "json": format_json,
}


def format_output(result: SummaryResult, fmt: str = "terminal", **kwargs: Any) -> str:
    """
    Format a SummaryResult in the specified format.

    Args:
        result: The SummaryResult to format.
        fmt: Output format — "terminal", "markdown", or "json".
        **kwargs: Additional keyword arguments passed to the formatter.

    Returns:
        Formatted string output.
    """
    formatter = FORMATS.get(fmt)
    if formatter is None:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {', '.join(FORMATS)}")
    return formatter(result, **kwargs)
