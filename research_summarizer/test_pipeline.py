#!/usr/bin/env python3
"""
End-to-End Pipeline Test — Research Summarizer Agent

Tests the full pipeline: agent → structured result → formatter output.

Two modes:
  1. Offline (default): Uses mock data to validate the formatter and CLI
     without requiring API keys or network access.
  2. Live (--live): Runs the real agent against sample URLs (requires
     ANTHROPIC_API_KEY and optionally TAVILY_API_KEY).

Usage:
    python -m research_summarizer.test_pipeline          # offline tests
    python -m research_summarizer.test_pipeline --live    # live end-to-end
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap

from .agent import SummaryResult
from .formatter import format_terminal, format_markdown, format_json, format_output


# ---------------------------------------------------------------------------
# Sample data for offline testing
# ---------------------------------------------------------------------------

SAMPLE_RESULT = SummaryResult(
    sources=[
        {
            "title": "Scaling Laws for Neural Language Models",
            "url": "https://arxiv.org/abs/2001.08361",
            "key_findings": [
                "Model performance scales as a power law with model size, dataset size, and compute budget.",
                "Larger models are significantly more sample-efficient than smaller models.",
                "Optimal compute allocation favors scaling model size over training tokens.",
                "Performance improvements are smooth and predictable across seven orders of magnitude.",
            ],
            "methodology": "Empirical study training transformer language models ranging from 768 to 1.5 billion parameters on WebText2 dataset, measuring cross-entropy loss.",
            "relevance": "Foundational work establishing the predictable scaling behavior that motivated the development of GPT-3 and subsequent large language models.",
        },
        {
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "key_findings": [
                "The Transformer architecture relies entirely on self-attention, eliminating recurrence and convolutions.",
                "Multi-head attention allows the model to attend to information from different representation subspaces.",
                "Achieves state-of-the-art BLEU scores on English-to-German and English-to-French translation.",
                "Training is significantly more parallelizable than recurrent architectures, reducing training time.",
                "Positional encodings enable the model to leverage sequence order without recurrence.",
            ],
            "methodology": "Novel neural network architecture evaluated on WMT 2014 machine translation benchmarks with comparison to existing RNN and CNN baselines.",
            "relevance": "Introduced the Transformer — the foundational architecture behind all modern large language models including GPT, Claude, and Gemini.",
        },
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "url": "https://arxiv.org/abs/2212.08073",
            "key_findings": [
                "AI systems can be trained to be helpful, harmless, and honest using AI-generated feedback rather than solely human feedback.",
                "Constitutional AI (CAI) uses a set of principles to guide self-critique and revision.",
                "RLHF from AI feedback (RLAIF) can match or exceed human-feedback-trained models on harmlessness.",
            ],
            "methodology": "Two-phase approach: (1) supervised learning with AI-revised responses, (2) RLAIF using AI-generated preference labels based on constitutional principles.",
            "relevance": "Demonstrates a scalable alternative to human feedback for AI alignment, reducing reliance on costly human annotation while improving safety properties.",
        },
    ],
    synthesis=(
        "These three papers trace the arc of modern LLM development: the Transformer "
        "architecture (2017) provided the foundation, scaling laws (2020) showed that "
        "bigger models predictably improve, and Constitutional AI (2022) addressed the "
        "alignment challenge that arises when deploying these powerful models. Together, "
        "they illustrate how architecture, scale, and safety form the three pillars of "
        "frontier AI development. A notable tension exists between the scaling laws' "
        "emphasis on raw capability and CAI's focus on controlled, safe behavior."
    ),
    key_takeaways=[
        "The Transformer architecture is the universal backbone of frontier AI — its self-attention mechanism enables the parallelism that makes scaling feasible.",
        "Scaling laws provide a predictive framework: given a compute budget, optimal model and data size can be calculated in advance.",
        "Alignment techniques like Constitutional AI are essential complements to scaling — capability without safety leads to unpredictable and potentially harmful behavior.",
        "The field is converging on a paradigm where architecture, scale, and alignment are co-designed rather than addressed in isolation.",
    ],
    follow_up=[
        "How do the original scaling laws hold up for models trained beyond 100B parameters (e.g., Chinchilla-optimal scaling)?",
        "What are the limitations of Constitutional AI when applied to multimodal models?",
        "How does mixture-of-experts architecture change the scaling dynamics described in the original scaling laws paper?",
        "What role does synthetic data play in extending the scaling curves when natural data is exhausted?",
    ],
    raw_response="",
    timestamp="2026-05-01T10:30:00.000000",
)


# ---------------------------------------------------------------------------
# Offline tests
# ---------------------------------------------------------------------------

def test_terminal_format() -> bool:
    """Test that terminal formatting produces expected ANSI output."""
    output = format_terminal(SAMPLE_RESULT)

    checks = [
        ("header", "JARVIS RESEARCH BRIEFING" in output),
        ("source count", "Sources analyzed: 3" in output),
        ("source 1 title", "Scaling Laws" in output),
        ("source 2 title", "Attention Is All You Need" in output),
        ("source 3 title", "Constitutional AI" in output),
        ("key findings marker", "Key Findings" in output),
        ("synthesis section", "CROSS-SOURCE SYNTHESIS" in output),
        ("takeaways section", "KEY TAKEAWAYS" in output),
        ("follow-up section", "RECOMMENDED FOLLOW-UP" in output),
        ("footer", "End of Jarvis Research Briefing" in output),
        ("non-empty", len(output) > 500),
    ]

    return _run_checks("Terminal Format", checks)


def test_markdown_format() -> bool:
    """Test that markdown formatting produces valid markdown."""
    output = format_markdown(SAMPLE_RESULT)

    checks = [
        ("h1 header", output.startswith("# Jarvis Research Briefing") ),
        ("h2 source headers", "## Source 1:" in output),
        ("h3 findings", "### Key Findings" in output),
        ("bullet points", "- " in output),
        ("numbered takeaways", "1. " in output),
        ("synthesis section", "## Cross-Source Synthesis" in output),
        ("follow-up section", "## Recommended Follow-Up" in output),
        ("URL links", "](https://arxiv.org" in output),
        ("footer", "Jarvis Research Summarizer Agent" in output),
        ("no ANSI codes", "\033[" not in output),
    ]

    return _run_checks("Markdown Format", checks)


def test_json_format() -> bool:
    """Test that JSON formatting produces valid, parseable JSON."""
    output = format_json(SAMPLE_RESULT)

    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        _print_result("JSON Format", False, f"Invalid JSON: {e}")
        return False

    checks = [
        ("has meta", "meta" in data),
        ("has sources", "sources" in data and len(data["sources"]) == 3),
        ("has synthesis", bool(data.get("cross_source_synthesis"))),
        ("has takeaways", len(data.get("key_takeaways", [])) == 4),
        ("has follow-up", len(data.get("suggested_follow_up", [])) == 4),
        ("meta generator", data["meta"]["generator"] == "Jarvis Research Summarizer Agent"),
        ("meta source_count", data["meta"]["source_count"] == 3),
        ("source has title", data["sources"][0].get("title") == "Scaling Laws for Neural Language Models"),
        ("source has url", "arxiv.org" in data["sources"][0].get("url", "")),
        ("source has findings", len(data["sources"][0].get("key_findings", [])) >= 3),
    ]

    return _run_checks("JSON Format", checks)


def test_format_dispatcher() -> bool:
    """Test the unified format_output dispatcher."""
    checks = []

    # Valid formats
    for fmt in ("terminal", "markdown", "json"):
        try:
            out = format_output(SAMPLE_RESULT, fmt=fmt)
            checks.append((f"dispatch {fmt}", len(out) > 100))
        except Exception as e:
            checks.append((f"dispatch {fmt}", False))

    # Invalid format
    try:
        format_output(SAMPLE_RESULT, fmt="invalid")
        checks.append(("reject invalid format", False))
    except ValueError:
        checks.append(("reject invalid format", True))

    return _run_checks("Format Dispatcher", checks)


def test_empty_result() -> bool:
    """Test formatting with an empty/minimal SummaryResult."""
    empty = SummaryResult(raw_response="Agent returned raw text only.")

    checks = []
    for fmt in ("terminal", "markdown", "json"):
        try:
            out = format_output(empty, fmt=fmt)
            checks.append((f"empty {fmt} non-empty output", len(out) > 50))
            if fmt == "terminal":
                checks.append(("empty terminal shows raw", "raw" in out.lower() or "Raw" in out))
            elif fmt == "json":
                data = json.loads(out)
                checks.append(("empty json has raw_response", "raw_response" in data))
        except Exception as e:
            checks.append((f"empty {fmt} no crash", False))

    return _run_checks("Empty Result Handling", checks)


# ---------------------------------------------------------------------------
# Live end-to-end test
# ---------------------------------------------------------------------------

SAMPLE_URLS = [
    "https://arxiv.org/abs/2001.08361",
    "https://arxiv.org/abs/1706.03762",
]


async def test_live_pipeline() -> bool:
    """Run the real agent against sample URLs and validate output."""
    from .agent import ResearchSummarizerAgent

    print("\n  Live Pipeline Test")
    print("  " + "-" * 50)
    print(f"  URLs: {', '.join(SAMPLE_URLS)}")
    print("  Running agent (this may take 30-60 seconds)...\n")

    agent = ResearchSummarizerAgent(verbose=True)

    try:
        result = await agent.summarize(
            SAMPLE_URLS,
            topic="frontier AI model architectures and scaling",
        )
    except Exception as e:
        _print_result("Live Agent Run", False, str(e))
        return False

    checks = [
        ("has sources", len(result.sources) > 0),
        ("has synthesis", bool(result.synthesis)),
        ("has takeaways", len(result.key_takeaways) > 0),
        ("has timestamp", bool(result.timestamp)),
    ]

    agent_ok = _run_checks("Live Agent Output", checks)

    # Test all formatters on the live result
    fmt_checks = []
    for fmt in ("terminal", "markdown", "json"):
        try:
            out = format_output(result, fmt=fmt)
            fmt_checks.append((f"live {fmt} format", len(out) > 200))
        except Exception as e:
            fmt_checks.append((f"live {fmt} format", False))

    fmt_ok = _run_checks("Live Formatter Output", fmt_checks)

    # Print the terminal output as a demo
    print("\n" + "=" * 60)
    print("  DEMO: Terminal Output from Live Run")
    print("=" * 60)
    print(format_terminal(result))

    return agent_ok and fmt_ok


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

def _run_checks(suite: str, checks: list[tuple[str, bool]]) -> bool:
    """Run a list of (name, passed) checks and print results."""
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    all_ok = passed == total

    status = "\033[38;5;42m PASS \033[0m" if all_ok else "\033[38;5;196m FAIL \033[0m"
    print(f"  [{status}] {suite} ({passed}/{total})")

    for name, ok in checks:
        icon = "\033[38;5;42m✓\033[0m" if ok else "\033[38;5;196m✗\033[0m"
        print(f"    {icon} {name}")

    return all_ok


def _print_result(suite: str, ok: bool, detail: str = "") -> None:
    """Print a single test result."""
    status = "\033[38;5;42m PASS \033[0m" if ok else "\033[38;5;196m FAIL \033[0m"
    msg = f"  [{status}] {suite}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Research Summarizer — Pipeline Tests")
    parser.add_argument(
        "--live", action="store_true",
        help="Run live end-to-end test against real URLs (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Print a demo of all output formats using sample data",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Research Summarizer — Pipeline Tests")
    print("=" * 60 + "\n")

    if args.demo:
        print("=" * 60)
        print("  DEMO: Terminal Format")
        print("=" * 60)
        print(format_terminal(SAMPLE_RESULT))

        print("=" * 60)
        print("  DEMO: Markdown Format")
        print("=" * 60)
        print(format_markdown(SAMPLE_RESULT))

        print("=" * 60)
        print("  DEMO: JSON Format")
        print("=" * 60)
        print(format_json(SAMPLE_RESULT))
        return

    # Run offline tests
    results = [
        test_terminal_format(),
        test_markdown_format(),
        test_json_format(),
        test_format_dispatcher(),
        test_empty_result(),
    ]

    # Run live test if requested
    if args.live:
        live_ok = asyncio.run(test_live_pipeline())
        results.append(live_ok)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"  \033[38;5;42mAll {total} test suites passed.\033[0m")
    else:
        print(f"  \033[38;5;196m{total - passed}/{total} test suites failed.\033[0m")
    print(f"{'=' * 60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
