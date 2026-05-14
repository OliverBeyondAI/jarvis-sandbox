#!/usr/bin/env python3
"""
HTML Trace Viewer Generator — creates a polished, self-contained HTML page
that visualizes the agent's extended thinking, tool calls, and final response.

Can be used as a module (from viewer import generate_html_viewer) or standalone:
    python viewer.py trace.json           # generates trace_viewer.html
    python viewer.py trace.json out.html  # custom output path
"""

from __future__ import annotations

import json
import html
import sys
from pathlib import Path
from typing import Any


def generate_html_viewer(trace: dict[str, Any]) -> str:
    """Generate a self-contained HTML page visualizing an agent trace."""

    thinking_blocks_html = ""
    for i, tb in enumerate(trace.get("thinking_blocks", []), 1):
        escaped = html.escape(tb["text"])
        thinking_blocks_html += f"""
        <div class="thinking-block">
            <div class="block-header">
                <span class="block-label">Thinking Block {i}</span>
                <span class="block-meta">Turn {tb['turn']} &middot; budget: {tb['budget_tokens']:,} tokens</span>
            </div>
            <div class="thinking-content" id="thinking-{i}">
                <div class="thinking-preview">{escaped[:400]}{'...' if len(escaped) > 400 else ''}</div>
                <div class="thinking-full" style="display:none">{escaped}</div>
            </div>
            <button class="expand-btn" onclick="toggleThinking({i})">Show full thinking</button>
        </div>"""

    tool_calls_html = ""
    for i, tc in enumerate(trace.get("tool_calls", []), 1):
        input_json = html.escape(json.dumps(tc["input"], indent=2))
        try:
            result_parsed = json.loads(tc["result"])
            result_formatted = html.escape(json.dumps(result_parsed, indent=2))
        except (json.JSONDecodeError, TypeError):
            result_formatted = html.escape(tc["result"])

        tool_calls_html += f"""
        <div class="tool-call">
            <div class="tool-header">
                <span class="tool-name">{html.escape(tc['name'])}</span>
                <span class="tool-meta">Turn {tc['turn']} &middot; {tc.get('duration_ms', 0):.0f}ms</span>
            </div>
            <div class="tool-io">
                <div class="tool-section">
                    <div class="tool-section-label">Input</div>
                    <pre class="tool-json">{input_json}</pre>
                </div>
                <div class="tool-section">
                    <div class="tool-section-label">Result</div>
                    <pre class="tool-json result-json" id="result-{i}">
                        <span class="result-preview">{result_formatted[:500]}{'...' if len(result_formatted) > 500 else ''}</span>
                        <span class="result-full" style="display:none">{result_formatted}</span>
                    </pre>
                    {'<button class="expand-btn small" onclick="toggleResult(' + str(i) + ')">Show full result</button>' if len(result_formatted) > 500 else ''}
                </div>
            </div>
        </div>"""

    tools_discovered = trace.get("tools_discovered", [])
    tools_pills = " ".join(
        f'<span class="tool-pill">{html.escape(t["name"])}</span>'
        for t in tools_discovered
    )

    final_response = html.escape(trace.get("final_response", ""))
    # Convert markdown-like formatting
    final_html = final_response.replace("\n\n", "</p><p>").replace("\n", "<br>")
    final_html = f"<p>{final_html}</p>"

    # Build timeline
    events = []
    for tb in trace.get("thinking_blocks", []):
        events.append(("thinking", tb["turn"], f"Extended thinking ({len(tb['text'])} chars)"))
    for tc in trace.get("tool_calls", []):
        events.append(("tool", tc["turn"], f"{tc['name']}({json.dumps(tc['input'])[:50]})"))
    events.sort(key=lambda e: (e[1], 0 if e[0] == "thinking" else 1))

    timeline_html = ""
    for etype, turn, label in events:
        icon = "brain" if etype == "thinking" else "wrench"
        css_class = "thinking" if etype == "thinking" else "tool"
        timeline_html += f"""
        <div class="timeline-event {css_class}">
            <div class="timeline-icon {css_class}">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    {'<path d="M8 1a5 5 0 0 0-5 5c0 1.5.7 2.8 1.7 3.7.5.5.8 1.1.8 1.8v.5c0 .6.4 1 1 1h3c.6 0 1-.4 1-1v-.5c0-.7.3-1.3.8-1.8A5 5 0 0 0 8 1zm-1.5 13c0 .3.1.5.2.7.2.2.4.3.7.3h1.2c.3 0 .5-.1.7-.3.1-.2.2-.4.2-.7v-.5h-3v.5z"/>' if etype == "thinking" else '<<path d="M11.5 1.3l3.2 3.2c.4.4.4 1 0 1.4L6.3 14.3c-.2.2-.4.3-.7.3H2.4c-.6 0-1-.4-1-1v-3.2c0-.3.1-.5.3-.7L10.1 1.3c.4-.4 1-.4 1.4 0zM3.4 10.8v1.8h1.8l7.5-7.5-1.8-1.8-7.5 7.5z"/>'}
                </svg>
            </div>
            <div class="timeline-content">
                <span class="timeline-turn">Turn {turn}</span>
                <span class="timeline-label">{html.escape(label)}</span>
            </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weather Thinking Agent — Trace Viewer</title>
<style>
:root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface-2: #232733;
    --border: #2d3148;
    --text: #e4e6f0;
    --text-dim: #8b8fa8;
    --accent: #6c8aff;
    --accent-glow: rgba(108, 138, 255, 0.15);
    --thinking: #c084fc;
    --thinking-bg: rgba(192, 132, 252, 0.08);
    --thinking-border: rgba(192, 132, 252, 0.25);
    --tool: #34d399;
    --tool-bg: rgba(52, 211, 153, 0.08);
    --tool-border: rgba(52, 211, 153, 0.25);
    --response: #60a5fa;
    --response-bg: rgba(96, 165, 250, 0.08);
    --warning: #fbbf24;
    --danger: #f87171;
    --radius: 10px;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
}}

.container {{
    max-width: 960px;
    margin: 0 auto;
    padding: 40px 24px 80px;
}}

/* Header */
.header {{
    text-align: center;
    margin-bottom: 48px;
    padding-bottom: 32px;
    border-bottom: 1px solid var(--border);
}}

.header h1 {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, var(--accent), var(--thinking));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.header .subtitle {{
    color: var(--text-dim);
    font-size: 15px;
}}

.header .model-badge {{
    display: inline-block;
    margin-top: 12px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    background: var(--accent-glow);
    color: var(--accent);
    border: 1px solid rgba(108, 138, 255, 0.3);
}}

/* Query card */
.query-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 32px;
}}

.query-card .label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 8px;
}}

.query-card .prompt {{
    font-size: 16px;
    line-height: 1.7;
    color: var(--text);
}}

/* Stats grid */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 32px;
}}

.stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    text-align: center;
}}

.stat-value {{
    font-size: 24px;
    font-weight: 700;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
}}

.stat-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-top: 4px;
}}

/* Tools discovered */
.tools-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 32px;
    padding: 16px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}}

.tools-bar .label {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
}}

.tool-pill {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 13px;
    font-weight: 500;
    font-family: var(--mono);
    background: var(--tool-bg);
    color: var(--tool);
    border: 1px solid var(--tool-border);
}}

/* Section headers */
.section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 40px 0 20px;
    font-size: 18px;
    font-weight: 600;
}}

.section-header .icon {{
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}

.section-header .icon.thinking {{
    background: var(--thinking-bg);
    color: var(--thinking);
    border: 1px solid var(--thinking-border);
}}

.section-header .icon.tool {{
    background: var(--tool-bg);
    color: var(--tool);
    border: 1px solid var(--tool-border);
}}

.section-header .icon.response {{
    background: var(--response-bg);
    color: var(--response);
    border: 1px solid rgba(96, 165, 250, 0.25);
}}

.section-header .icon.timeline {{
    background: var(--accent-glow);
    color: var(--accent);
    border: 1px solid rgba(108, 138, 255, 0.3);
}}

/* Timeline */
.timeline {{
    position: relative;
    padding-left: 32px;
    margin-bottom: 32px;
}}

.timeline::before {{
    content: '';
    position: absolute;
    left: 11px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border);
}}

.timeline-event {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    position: relative;
}}

.timeline-icon {{
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    position: absolute;
    left: -32px;
    z-index: 1;
}}

.timeline-icon.thinking {{
    background: var(--thinking-bg);
    color: var(--thinking);
    border: 2px solid var(--thinking);
}}

.timeline-icon.tool {{
    background: var(--tool-bg);
    color: var(--tool);
    border: 2px solid var(--tool);
}}

.timeline-content {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
}}

.timeline-turn {{
    font-weight: 600;
    font-size: 12px;
    color: var(--text-dim);
    background: var(--surface-2);
    padding: 2px 8px;
    border-radius: 4px;
    font-variant-numeric: tabular-nums;
}}

.timeline-label {{
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
}}

/* Thinking blocks */
.thinking-block {{
    background: var(--surface);
    border: 1px solid var(--thinking-border);
    border-left: 3px solid var(--thinking);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
}}

.block-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}

.block-label {{
    font-weight: 600;
    font-size: 14px;
    color: var(--thinking);
}}

.block-meta {{
    font-size: 12px;
    color: var(--text-dim);
}}

.thinking-content {{
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.7;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow: hidden;
    position: relative;
}}

.thinking-content.expanded {{
    max-height: none;
    overflow: visible;
}}

.thinking-content:not(.expanded)::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 80px;
    background: linear-gradient(transparent, var(--surface));
    pointer-events: none;
}}

.expand-btn {{
    display: inline-block;
    margin-top: 12px;
    padding: 6px 16px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    background: var(--surface-2);
    color: var(--text-dim);
    border: 1px solid var(--border);
    transition: all 0.15s;
}}

.expand-btn:hover {{
    color: var(--text);
    border-color: var(--text-dim);
}}

.expand-btn.small {{
    margin-top: 8px;
    padding: 4px 12px;
    font-size: 11px;
}}

/* Tool calls */
.tool-call {{
    background: var(--surface);
    border: 1px solid var(--tool-border);
    border-left: 3px solid var(--tool);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
}}

.tool-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}}

.tool-name {{
    font-family: var(--mono);
    font-weight: 600;
    font-size: 15px;
    color: var(--tool);
}}

.tool-meta {{
    font-size: 12px;
    color: var(--text-dim);
}}

.tool-io {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}}

@media (max-width: 640px) {{
    .tool-io {{
        grid-template-columns: 1fr;
    }}
}}

.tool-section {{
    min-width: 0;
}}

.tool-section-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-bottom: 6px;
}}

.tool-json {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.6;
    color: var(--text);
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 240px;
    overflow-y: auto;
}}

/* Final response */
.response-card {{
    background: var(--surface);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-left: 3px solid var(--response);
    border-radius: var(--radius);
    padding: 28px;
    margin-bottom: 32px;
}}

.response-card p {{
    margin-bottom: 12px;
    font-size: 15px;
    line-height: 1.8;
}}

.response-card p:last-child {{
    margin-bottom: 0;
}}

/* Footer */
.footer {{
    text-align: center;
    padding-top: 32px;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 13px;
}}

.footer a {{
    color: var(--accent);
    text-decoration: none;
}}
</style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <h1>Weather Thinking Agent</h1>
        <div class="subtitle">Claude Opus 4.7 &middot; Extended Thinking &middot; MCP Tool Use</div>
        <div class="model-badge">{html.escape(trace.get('model', 'unknown'))}</div>
    </div>

    <!-- Query -->
    <div class="query-card">
        <div class="label">User Query</div>
        <div class="prompt">{html.escape(trace.get('prompt', ''))}</div>
    </div>

    <!-- Stats -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(trace.get('thinking_blocks', []))}</div>
            <div class="stat-label">Thinking Blocks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(trace.get('tool_calls', []))}</div>
            <div class="stat-label">Tool Calls</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{trace.get('total_turns', 0)}</div>
            <div class="stat-label">Turns</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{trace.get('budget_tokens', 0):,}</div>
            <div class="stat-label">Think Budget</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{trace.get('total_input_tokens', 0):,}</div>
            <div class="stat-label">Input Tokens</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{trace.get('wall_time_s', 0):.1f}s</div>
            <div class="stat-label">Wall Time</div>
        </div>
    </div>

    <!-- Tools discovered -->
    <div class="tools-bar">
        <span class="label">MCP Tools:</span>
        {tools_pills}
    </div>

    <!-- Timeline -->
    <div class="section-header">
        <div class="icon timeline">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm0 14.5a6.5 6.5 0 1 1 0-13 6.5 6.5 0 0 1 0 13zM8.5 4h-1v5l4.3 2.5.5-.8-3.8-2.2V4z"/>
            </svg>
        </div>
        Execution Timeline
    </div>
    <div class="timeline">
        {timeline_html}
    </div>

    <!-- Thinking blocks -->
    <div class="section-header">
        <div class="icon thinking">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 1a5 5 0 0 0-5 5c0 1.5.7 2.8 1.7 3.7.5.5.8 1.1.8 1.8v.5c0 .6.4 1 1 1h3c.6 0 1-.4 1-1v-.5c0-.7.3-1.3.8-1.8A5 5 0 0 0 8 1zm-1.5 13c0 .3.1.5.2.7.2.2.4.3.7.3h1.2c.3 0 .5-.1.7-.3.1-.2.2-.4.2-.7v-.5h-3v.5z"/>
            </svg>
        </div>
        Extended Thinking
    </div>
    {thinking_blocks_html if thinking_blocks_html else '<p style="color: var(--text-dim); font-size: 14px; margin-bottom: 24px;">No thinking blocks captured.</p>'}

    <!-- Tool calls -->
    <div class="section-header">
        <div class="icon tool">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M11.5 1.3l3.2 3.2c.4.4.4 1 0 1.4L6.3 14.3c-.2.2-.4.3-.7.3H2.4c-.6 0-1-.4-1-1v-3.2c0-.3.1-.5.3-.7L10.1 1.3c.4-.4 1-.4 1.4 0zM3.4 10.8v1.8h1.8l7.5-7.5-1.8-1.8-7.5 7.5z"/>
            </svg>
        </div>
        MCP Tool Calls
    </div>
    {tool_calls_html if tool_calls_html else '<p style="color: var(--text-dim); font-size: 14px; margin-bottom: 24px;">No tool calls made.</p>'}

    <!-- Final response -->
    <div class="section-header">
        <div class="icon response">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M1 2.5A1.5 1.5 0 0 1 2.5 1h11A1.5 1.5 0 0 1 15 2.5v9A1.5 1.5 0 0 1 13.5 13H5l-3 3v-3H2.5A1.5 1.5 0 0 1 1 11.5v-9zM4 5h8v1H4V5zm0 3h6v1H4V8z"/>
            </svg>
        </div>
        Final Response
    </div>
    <div class="response-card">
        {final_html}
    </div>

    <!-- Footer -->
    <div class="footer">
        Generated by Weather Thinking Agent &middot; {html.escape(trace.get('timestamp', '')[:19])} UTC
    </div>

</div>

<script>
function toggleThinking(id) {{
    const el = document.getElementById('thinking-' + id);
    const btn = el.parentElement.querySelector('.expand-btn');
    if (el.classList.contains('expanded')) {{
        el.classList.remove('expanded');
        el.querySelector('.thinking-preview').style.display = '';
        el.querySelector('.thinking-full').style.display = 'none';
        btn.textContent = 'Show full thinking';
    }} else {{
        el.classList.add('expanded');
        el.querySelector('.thinking-preview').style.display = 'none';
        el.querySelector('.thinking-full').style.display = '';
        btn.textContent = 'Collapse';
    }}
}}

function toggleResult(id) {{
    const el = document.getElementById('result-' + id);
    const btn = el.parentElement.querySelector('.expand-btn');
    const preview = el.querySelector('.result-preview');
    const full = el.querySelector('.result-full');
    if (full.style.display === 'none') {{
        preview.style.display = 'none';
        full.style.display = '';
        btn.textContent = 'Collapse';
    }} else {{
        preview.style.display = '';
        full.style.display = 'none';
        btn.textContent = 'Show full result';
    }}
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI: generate HTML from a trace JSON file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <trace.json> [output.html]")
        sys.exit(1)

    trace_path = Path(sys.argv[1])
    if not trace_path.exists():
        print(f"Error: {trace_path} not found")
        sys.exit(1)

    trace_data = json.loads(trace_path.read_text())
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else trace_path.with_suffix(".html")

    html_content = generate_html_viewer(trace_data)
    out_path.write_text(html_content)
    print(f"HTML viewer saved to: {out_path}")
