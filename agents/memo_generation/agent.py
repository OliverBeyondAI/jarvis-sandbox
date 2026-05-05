"""
Memo Generation Agent — Takes synthesis findings from Agent 2 and produces
a polished internal memo suitable for stakeholder distribution.

The agent uses Claude to transform structured synthesis data into a
well-formatted, narrative memo with executive summary, key findings,
recommendations, and appendix.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import anthropic

from ..synthesis.models import SynthesisReport
from ..trend_research.models import ResearchReport
from .config import MemoConfig
from .models import ArtifactBundle, InternalMemo, MemoAudience, MemoSection
from .storage import MemoStorage


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

MEMO_AGENT_SYSTEM = """You are an expert internal communications writer specializing in technology strategy memos for healthcare companies.

Your task: Transform structured synthesis data (research findings mapped to product applications) into a polished internal memo that stakeholders can read and act on.

## Memo Structure

Produce a memo with these sections:

1. **TL;DR** — 2-3 sentences capturing the most important takeaway. A busy executive should be able to read just this and know what to do.

2. **Context & Methodology** — Brief overview of how the research was conducted and what was analyzed. Keep to 2-3 sentences.

3. **Key Findings** — The most important discoveries, organized by theme. Use bullet points. Include data points and specifics.

4. **Strategic Opportunities** — Top 3-5 opportunities ranked by impact. For each:
   - What it is (one line)
   - Why it matters (clinical/business impact)
   - Platform fit (OphthoFlow, Xena, or both)
   - Effort estimate and timeline
   - Recommended next step

5. **Quick Wins** — Opportunities that can be captured in <1 month with high confidence. These are low-hanging fruit.

6. **Moonshots** — High-impact, high-effort bets worth investigating. Include why the risk is justified.

7. **Risk Assessment** — Key risks to be aware of, with mitigation strategies.

8. **Recommended Next Steps** — Concrete, prioritized actions. Who should do what by when.

9. **Appendix: Trend Details** — For each trend analyzed, provide a brief summary paragraph covering relevance, maturity, and top application ideas.

## Writing Style

- Professional but accessible — avoid jargon unless it's industry-standard
- Lead with "so what" — why should the reader care?
- Use concrete numbers, timelines, and names whenever available
- Active voice, present tense for current state, future tense for recommendations
- Bold key terms and use bullet points for scanability
- Keep the total memo to 1500-2500 words (excluding appendix)

## Output Format

Return your memo as valid Markdown. Use proper heading levels (## for sections, ### for subsections).
Start with a YAML-style header block:

```
---
TO: Product & Engineering Leadership
FROM: AI Research Pipeline
DATE: {date}
RE: {subject}
CLASSIFICATION: Internal — Do Not Distribute
---
```

Then the full memo body in Markdown."""


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

MEMO_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
    --color-primary: #1a365d;
    --color-accent: #2b6cb0;
    --color-bg: #ffffff;
    --color-surface: #f7fafc;
    --color-border: #e2e8f0;
    --color-text: #2d3748;
    --color-text-muted: #718096;
    --color-success: #38a169;
    --color-warning: #d69e2e;
    --color-danger: #e53e3e;
    --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'SF Mono', 'Fira Code', monospace;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: var(--font-sans);
    color: var(--color-text);
    background: var(--color-bg);
    line-height: 1.7;
    padding: 2rem;
    max-width: 900px;
    margin: 0 auto;
}}
.memo-header {{
    border-bottom: 3px solid var(--color-primary);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}}
.memo-header h1 {{
    color: var(--color-primary);
    font-size: 1.75rem;
    margin-bottom: 0.5rem;
}}
.memo-meta {{
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.25rem 1rem;
    font-size: 0.9rem;
    color: var(--color-text-muted);
    margin-top: 1rem;
}}
.memo-meta dt {{
    font-weight: 600;
    color: var(--color-text);
}}
.classification {{
    display: inline-block;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.75rem;
}}
.tldr {{
    background: var(--color-surface);
    border-left: 4px solid var(--color-accent);
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 1.05rem;
}}
.tldr strong {{
    color: var(--color-primary);
}}
h2 {{
    color: var(--color-primary);
    font-size: 1.3rem;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--color-border);
}}
h3 {{
    color: var(--color-accent);
    font-size: 1.1rem;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}}
p {{ margin-bottom: 1rem; }}
ul, ol {{
    margin-bottom: 1rem;
    padding-left: 1.5rem;
}}
li {{ margin-bottom: 0.5rem; }}
strong {{ color: var(--color-primary); }}
.opportunity-card {{
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}}
.opportunity-card h4 {{
    color: var(--color-accent);
    margin-bottom: 0.5rem;
}}
.tag {{
    display: inline-block;
    background: var(--color-accent);
    color: white;
    border-radius: 3px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.25rem;
}}
.tag--effort {{
    background: var(--color-warning);
}}
.tag--risk {{
    background: var(--color-danger);
}}
.appendix {{
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 2px solid var(--color-border);
}}
.appendix h2 {{
    color: var(--color-text-muted);
}}
.footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--color-border);
    font-size: 0.8rem;
    color: var(--color-text-muted);
    text-align: center;
}}
@media (max-width: 640px) {{
    body {{ padding: 1rem; }}
    .memo-meta {{ grid-template-columns: 1fr; }}
    h2 {{ font-size: 1.15rem; }}
}}
</style>
</head>
<body>
{body}
<div class="footer">
    Generated by AI Research Pipeline &mdash; Agent 3 (Memo Generation) &mdash; {date}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------

class MemoGenerationAgent:
    """
    Agentic memo generation loop that uses Claude to transform synthesis
    data into a polished internal memo.
    """

    def __init__(self, config: MemoConfig | None = None) -> None:
        self.config = config or MemoConfig.from_env()
        self._client: anthropic.Anthropic | None = None
        self._storage: MemoStorage | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    @property
    def storage(self) -> MemoStorage:
        if self._storage is None:
            self._storage = MemoStorage(self.config)
        return self._storage

    async def generate_memo(
        self,
        synthesis_report: SynthesisReport,
        audience: MemoAudience = MemoAudience.PRODUCT,
    ) -> str:
        """
        Generate a formatted memo from a synthesis report.

        Args:
            synthesis_report: The SynthesisReport from Agent 2.
            audience: Target audience for tone/detail adjustments.

        Returns:
            Rendered memo as Markdown string.
        """
        print(f"\n[memo] Generating memo from: {synthesis_report.title}", file=sys.stderr)
        print(f"[memo] Target audience: {audience.value}", file=sys.stderr)

        # Serialize synthesis for the prompt
        synthesis_json = json.dumps(
            synthesis_report.model_dump(mode="json"),
            indent=2,
            default=str,
        )

        user_message = (
            f"## Synthesis Report to Convert into Memo\n\n"
            f"Target audience: **{audience.value}** (Product & Engineering Leadership)\n\n"
            f"Transform the following synthesis report into a polished internal memo "
            f"following the structure in your instructions.\n\n"
            f"```json\n{synthesis_json}\n```\n\n"
            f"## Requirements\n\n"
            f"1. Write the full memo in Markdown format.\n"
            f"2. Include the YAML header block (TO, FROM, DATE, RE, CLASSIFICATION).\n"
            f"3. Start with a compelling TL;DR.\n"
            f"4. Highlight the top strategic opportunities with clear next steps.\n"
            f"5. Make it actionable — readers should know exactly what to do after reading.\n"
            f"6. Include the appendix with trend details.\n"
            f"7. Keep the main body to 1500-2500 words."
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        turn = 0
        max_turns = self.config.max_agent_turns

        while turn < max_turns:
            turn += 1
            print(f"[memo] Turn {turn}/{max_turns}", file=sys.stderr)

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=MEMO_AGENT_SYSTEM,
                messages=messages,
            )

            text_blocks = [b.text for b in response.content if b.type == "text"]
            final_text = "\n".join(text_blocks)

            if response.stop_reason == "end_turn":
                print(f"[memo] Memo generated ({len(final_text)} chars)", file=sys.stderr)
                return final_text

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": "Please complete the memo in full Markdown format.",
            })

        print(f"[memo] Max turns ({max_turns}) reached", file=sys.stderr)
        return final_text if 'final_text' in dir() else ""

    def markdown_to_html(self, markdown_content: str, title: str = "Internal Memo") -> str:
        """
        Convert markdown memo to styled HTML.

        Uses a simple conversion approach that handles the most common
        Markdown patterns without external dependencies.
        """
        import re

        # Remove YAML front matter if present
        body = markdown_content
        if body.startswith("---"):
            end_idx = body.find("---", 3)
            if end_idx != -1:
                front_matter = body[3:end_idx].strip()
                body = body[end_idx + 3:].strip()

                # Parse front matter for header
                meta = {}
                for line in front_matter.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        meta[key.strip()] = val.strip()

                # Build header HTML
                header_html = '<div class="memo-header">\n'
                header_html += f'<h1>{meta.get("RE", title)}</h1>\n'
                header_html += '<dl class="memo-meta">\n'
                for key in ["TO", "FROM", "DATE"]:
                    if key in meta:
                        header_html += f"<dt>{key}:</dt><dd>{meta[key]}</dd>\n"
                header_html += "</dl>\n"
                classification = meta.get("CLASSIFICATION", "Internal")
                header_html += f'<span class="classification">{classification}</span>\n'
                header_html += "</div>\n"
            else:
                header_html = f'<div class="memo-header"><h1>{title}</h1></div>\n'
        else:
            header_html = f'<div class="memo-header"><h1>{title}</h1></div>\n'

        # Convert markdown to HTML (simple but effective conversion)
        html_body = self._convert_md_to_html(body)

        # Assemble final HTML
        date_str = asyncio.get_event_loop().time() if False else ""
        from datetime import datetime, timezone
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        full_body = header_html + html_body
        return MEMO_HTML_TEMPLATE.format(
            title=title,
            body=full_body,
            date=date_str,
        )

    def _convert_md_to_html(self, md: str) -> str:
        """Simple Markdown-to-HTML converter for memo content."""
        import re

        lines = md.split("\n")
        html_lines: list[str] = []
        in_list = False
        in_ol = False
        list_type = ""

        i = 0
        while i < len(lines):
            line = lines[i]

            # Headings
            if line.startswith("## "):
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                text = self._inline_format(line[3:])
                # Check for TL;DR section
                if "tl;dr" in line.lower() or "tldr" in line.lower():
                    html_lines.append(f'<h2>{text}</h2>')
                    # Wrap next paragraph(s) in tldr class
                    i += 1
                    # Skip blank lines after heading
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    tldr_content = []
                    while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
                        tldr_content.append(lines[i])
                        i += 1
                    if tldr_content:
                        html_lines.append(f'<div class="tldr">{self._inline_format(" ".join(tldr_content))}</div>')
                    continue
                else:
                    html_lines.append(f"<h2>{text}</h2>")
            elif line.startswith("### "):
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                text = self._inline_format(line[4:])
                html_lines.append(f"<h3>{text}</h3>")
            elif line.startswith("#### "):
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                text = self._inline_format(line[5:])
                html_lines.append(f"<h4>{text}</h4>")

            # Unordered list items
            elif re.match(r"^[\-\*]\s+", line):
                if not in_list or list_type != "ul":
                    if in_list:
                        html_lines.append(f"</{list_type}>")
                    html_lines.append("<ul>")
                    in_list = True
                    list_type = "ul"
                text = self._inline_format(re.sub(r"^[\-\*]\s+", "", line))
                html_lines.append(f"<li>{text}</li>")

            # Ordered list items
            elif re.match(r"^\d+\.\s+", line):
                if not in_list or list_type != "ol":
                    if in_list:
                        html_lines.append(f"</{list_type}>")
                    html_lines.append("<ol>")
                    in_list = True
                    list_type = "ol"
                text = self._inline_format(re.sub(r"^\d+\.\s+", "", line))
                html_lines.append(f"<li>{text}</li>")

            # Empty line
            elif not line.strip():
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                html_lines.append("")

            # Paragraph
            else:
                if in_list:
                    html_lines.append(f"</{list_type}>")
                    in_list = False
                text = self._inline_format(line)
                html_lines.append(f"<p>{text}</p>")

            i += 1

        if in_list:
            html_lines.append(f"</{list_type}>")

        return "\n".join(html_lines)

    def _inline_format(self, text: str) -> str:
        """Apply inline Markdown formatting (bold, italic, code)."""
        import re

        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        # Italic
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        # Inline code
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        return text

    async def run_full_pipeline(
        self,
        research_report: ResearchReport,
        synthesis_report: SynthesisReport,
        audience: MemoAudience = MemoAudience.PRODUCT,
    ) -> ArtifactBundle:
        """
        Full pipeline: generate memo, convert to HTML, store all artifacts in S3.

        Args:
            research_report: The ResearchReport from Agent 1.
            synthesis_report: The SynthesisReport from Agent 2.
            audience: Target audience for the memo.

        Returns:
            ArtifactBundle with paths to all stored artifacts.
        """
        # Step 1: Generate memo markdown
        memo_markdown = await self.generate_memo(synthesis_report, audience)

        # Step 2: Convert to HTML
        memo_html = self.markdown_to_html(
            memo_markdown,
            title=f"Memo: {synthesis_report.title}",
        )

        # Step 3: Store all outputs
        bundle = await self.storage.store_all_outputs(
            research_data=research_report.model_dump(mode="json"),
            synthesis_data=synthesis_report.model_dump(mode="json"),
            memo_markdown=memo_markdown,
            memo_html=memo_html,
            memo_title=synthesis_report.title,
        )

        print(f"[memo] Pipeline complete. Bundle ID: {bundle.bundle_id}", file=sys.stderr)
        return bundle


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def generate_memo(
    synthesis_report: SynthesisReport,
    config: MemoConfig | None = None,
    audience: MemoAudience = MemoAudience.PRODUCT,
) -> str:
    """
    Convenience function to generate a memo from a synthesis report.

    Returns:
        Rendered memo as Markdown string.
    """
    agent = MemoGenerationAgent(config)
    return await agent.generate_memo(synthesis_report, audience)


async def run_memo_pipeline(
    research_report: ResearchReport,
    synthesis_report: SynthesisReport,
    config: MemoConfig | None = None,
    audience: MemoAudience = MemoAudience.PRODUCT,
) -> ArtifactBundle:
    """
    Convenience function to run the full memo pipeline with S3 storage.

    Returns:
        ArtifactBundle with paths to all stored artifacts.
    """
    agent = MemoGenerationAgent(config)
    return await agent.run_full_pipeline(research_report, synthesis_report, audience)
