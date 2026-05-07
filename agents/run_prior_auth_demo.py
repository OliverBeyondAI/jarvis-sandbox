#!/usr/bin/env python3
"""
Prior Authorization Agent — End-to-End Demo
=============================================

Launches the simulated Prior Authorization Portal, runs the AI agent against it
to extract required fields and submit a prior authorization request, and displays
the agent's decision-making process and results in a richly formatted terminal UI.

Usage:
    python agents/run_prior_auth_demo.py                       # Default: specialty case
    python agents/run_prior_auth_demo.py --case controlled     # Controlled substance case
    python agents/run_prior_auth_demo.py --case gene_therapy   # Gene therapy case
    python agents/run_prior_auth_demo.py --all                 # Run all 5 cases sequentially
    python agents/run_prior_auth_demo.py --list                # List available cases
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import textwrap
import threading
import time
from datetime import datetime
from http.server import HTTPServer
from pathlib import Path

# Ensure the agents package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.prior_auth_agent import (
    SAMPLE_CASES,
    AgentResult,
    FormNavigator,
    PriorAuthAgent,
)
from agents.prior_auth_portal.server import PARequestHandler

# ---------------------------------------------------------------------------
# Terminal Colors & Formatting
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes for terminal output."""
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    ITALIC   = "\033[3m"
    ULINE    = "\033[4m"

    BLACK    = "\033[30m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"

    BG_BLACK   = "\033[40m"
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"
    BG_WHITE   = "\033[47m"

    # 256-color
    GRAY     = "\033[38;5;245m"
    ORANGE   = "\033[38;5;208m"
    PINK     = "\033[38;5;213m"
    TEAL     = "\033[38;5;43m"
    LIME     = "\033[38;5;118m"


def no_color():
    """Disable colors if terminal doesn't support them."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        for attr in dir(C):
            if not attr.startswith("_"):
                setattr(C, attr, "")


no_color()

W = 72  # Output width


def banner(text: str, color: str = C.CYAN) -> str:
    """Create a centered banner with box-drawing characters."""
    inner = f" {text} "
    pad = max(0, W - len(inner) - 2)
    left = pad // 2
    right = pad - left
    return (
        f"{color}{C.BOLD}"
        f"\u2554{'═' * W}\u2557\n"
        f"\u2551{'─' * left}{inner}{'─' * right}\u2551\n"
        f"\u255A{'═' * W}\u255D"
        f"{C.RESET}"
    )


def section(title: str, color: str = C.BLUE) -> str:
    pad = W - len(title) - 3
    return f"\n{color}{C.BOLD}── {title} {'─' * max(0, pad)}{C.RESET}"


def kv(key: str, value: str, key_color: str = C.CYAN, val_color: str = C.WHITE) -> str:
    return f"  {key_color}{key:<20}{C.RESET} {val_color}{value}{C.RESET}"


def bullet(text: str, indent: int = 2, color: str = C.GREEN) -> str:
    return f"{' ' * indent}{color}●{C.RESET} {text}"


def wrap_text(text: str, indent: int = 4, width: int = W - 4) -> str:
    lines = textwrap.wrap(text, width=width)
    prefix = " " * indent
    return "\n".join(f"{prefix}{line}" for line in lines)


def step_badge(step: int, name: str) -> str:
    colors = {1: C.CYAN, 2: C.MAGENTA, 3: C.YELLOW, 4: C.TEAL, 5: C.GREEN}
    c = colors.get(step, C.WHITE)
    return f"  {c}{C.BOLD}[Step {step}]{C.RESET} {C.BOLD}{name}{C.RESET}"


def progress_bar(current: int, total: int, width: int = 40, label: str = "") -> str:
    filled = int(width * current / total)
    bar = f"{C.GREEN}{'█' * filled}{C.DIM}{'░' * (width - filled)}{C.RESET}"
    pct = f"{100 * current // total}%"
    return f"  {bar} {pct} {C.DIM}{label}{C.RESET}"


# ---------------------------------------------------------------------------
# Portal Server Management
# ---------------------------------------------------------------------------

class PortalServer:
    """Manages the Prior Authorization Portal HTTP server in a background thread."""

    def __init__(self, port: int = 8080):
        self.port = port
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start the portal server. Returns True if started successfully."""
        try:
            self.server = HTTPServer(("localhost", self.port), PARequestHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            time.sleep(0.3)  # Brief pause for startup
            return True
        except OSError as e:
            if "Address already in use" in str(e):
                return True  # Already running
            return False

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()


# ---------------------------------------------------------------------------
# Enhanced Agent Runner with Live Logging
# ---------------------------------------------------------------------------

class DemoAgentRunner:
    """
    Wraps PriorAuthAgent with enhanced terminal output showing the agent's
    decision-making process in real time.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.step_names = {
            1: "Patient Information",
            2: "Medication Details",
            3: "Clinical Details",
            4: "Supporting Documents",
            5: "Review & Submit",
        }

    def print_case_overview(self, case_data: dict, case_name: str):
        """Print a formatted overview of the case before running."""
        print(section("CASE OVERVIEW", C.MAGENTA))

        desc = case_data.get("description", case_name)
        print(kv("Case:", desc, C.YELLOW))
        print()

        # Patient
        p = case_data.get("patient", {})
        print(f"  {C.BOLD}{C.CYAN}Patient{C.RESET}")
        print(kv("  Name:", f"{p.get('firstName', '')} {p.get('lastName', '')}"))
        print(kv("  DOB:", p.get("dob", "N/A")))
        print(kv("  Member ID:", p.get("memberId", "N/A")))
        plan_labels = {
            "gold_ppo": "Gold PPO", "silver_hmo": "Silver HMO",
            "bronze_hdhp": "Bronze HDHP", "platinum_ppo": "Platinum PPO",
            "medicaid": "Medicaid Managed Care", "medicare_advantage": "Medicare Advantage",
        }
        print(kv("  Plan:", plan_labels.get(p.get("insurancePlan", ""), p.get("insurancePlan", ""))))
        print()

        # Provider
        prov = case_data.get("provider", {})
        print(f"  {C.BOLD}{C.CYAN}Provider{C.RESET}")
        print(kv("  Name:", prov.get("name", "N/A")))
        print(kv("  NPI:", prov.get("npi", "N/A")))
        print(kv("  Phone:", prov.get("phone", "N/A")))
        print()

        # Medication
        m = case_data.get("medication", {})
        cat_labels = {
            "specialty": "Specialty / Biologic", "controlled": "Controlled Substance",
            "brand_nonpreferred": "Brand Name (Non-Preferred)",
            "compound": "Compounded Medication", "gene_therapy": "Gene / Cell Therapy",
        }
        print(f"  {C.BOLD}{C.CYAN}Medication{C.RESET}")
        print(kv("  Name:", m.get("name", "N/A")))
        print(kv("  Category:", cat_labels.get(m.get("category", ""), m.get("category", ""))))
        print(kv("  Dosage:", m.get("dosage", "N/A")))
        print()

        # Clinical summary
        cl = case_data.get("clinical", {})
        print(f"  {C.BOLD}{C.CYAN}Clinical{C.RESET}")
        print(kv("  Primary Dx:", f"{cl.get('icd10Primary', '')} — {cl.get('diagDescription', '')}"))
        urgency_labels = {
            "routine": "Routine (5-7 business days)",
            "urgent": "Urgent (24-72 hours)",
            "emergency": "Emergency (immediate)",
        }
        urg = cl.get("urgency", "routine")
        urg_color = {
            "routine": C.GREEN, "urgent": C.YELLOW, "emergency": C.RED,
        }.get(urg, C.WHITE)
        print(f"  {C.CYAN}{'Urgency:':<20}{C.RESET} {urg_color}{C.BOLD}{urgency_labels.get(urg, urg)}{C.RESET}")

    def print_agent_process(self, result: AgentResult):
        """Print a detailed breakdown of the agent's decision-making process."""
        print(section("AGENT DECISION-MAKING PROCESS", C.YELLOW))

        # Group tool calls by step
        current_step = 1
        step_calls: dict[int, list] = {i: [] for i in range(1, 6)}
        tool_sequence = []

        for tc in result.tool_calls:
            name = tc["name"]
            inp = tc["input"]

            if name == "click_button" and inp.get("button") == "next":
                current_step = min(current_step + 1, 5)
            elif name == "click_button" and inp.get("button") == "back":
                current_step = max(current_step - 1, 1)

            step_calls[current_step].append(tc)
            tool_sequence.append((current_step, tc))

        # Print each step's activity
        for step in range(1, 6):
            calls = step_calls[step]
            if not calls:
                continue

            name = self.step_names.get(step, f"Step {step}")
            print(step_badge(step, name))

            fills = [c for c in calls if c["name"] == "fill_field"]
            reads = [c for c in calls if c["name"] == "read_page"]
            navs = [c for c in calls if c["name"] == "click_button"]
            uploads = [c for c in calls if c["name"] == "upload_file"]
            checks = [c for c in calls if c["name"] == "get_form_state"]

            if reads:
                print(f"    {C.DIM}↳ Read page to discover {len(fills)} fields{C.RESET}")

            for fill in fills:
                fid = fill["input"].get("field_id", "?")
                val = fill["input"].get("value", "")
                # Truncate long values
                if len(val) > 55:
                    val = val[:52] + "..."
                print(f"    {C.GREEN}✎{C.RESET} {C.BOLD}{fid}{C.RESET} {C.DIM}←{C.RESET} {C.WHITE}\"{val}\"{C.RESET}")

            for up in uploads:
                cat = up["input"].get("category", "?")
                fname = up["input"].get("filename", "?")
                print(f"    {C.TEAL}📎{C.RESET} Upload {C.DIM}[{cat}]{C.RESET} {fname}")

            for nav in navs:
                btn = nav["input"].get("button", "?")
                if btn == "submit":
                    print(f"    {C.GREEN}{C.BOLD}⏎ Submit form{C.RESET}")
                elif btn == "next":
                    print(f"    {C.BLUE}→ Navigate to next step{C.RESET}")
                elif btn == "back":
                    print(f"    {C.YELLOW}← Navigate back{C.RESET}")

            print()

    def print_result(self, result: AgentResult, case_data: dict):
        """Print the final result with formatted output."""
        if result.success:
            print(banner("SUBMISSION SUCCESSFUL", C.GREEN))
        else:
            print(banner("SUBMISSION INCOMPLETE", C.RED))

        print()
        print(section("RESULT SUMMARY"))

        status_color = C.GREEN if result.success else C.RED
        status_text = "APPROVED FOR REVIEW" if result.success else "INCOMPLETE"
        print(kv("Status:", f"{status_color}{C.BOLD}{status_text}{C.RESET}"))

        if result.reference_number:
            print(kv("Reference #:", f"{C.BOLD}{result.reference_number}{C.RESET}"))

        print(kv("Agent Turns:", str(result.turns)))
        print(kv("Tool Calls:", str(len(result.tool_calls))))
        print(kv("Timestamp:", result.timestamp[:19].replace("T", " ")))

        # Error summary
        if result.errors:
            print()
            err_color = C.YELLOW if result.success else C.RED
            print(f"  {err_color}{C.BOLD}Errors encountered ({len(result.errors)}):{C.RESET}")
            for err in result.errors[:8]:
                print(f"    {C.RED}✗{C.RESET} {err}")
        else:
            print(kv("Errors:", f"{C.GREEN}None{C.RESET}"))

        # Form data summary
        if result.success and result.form_data:
            print(section("SUBMITTED FORM DATA"))

            fd = result.form_data
            pat = fd.get("patient", {})
            prov = fd.get("provider", {})
            med = fd.get("medication", {})
            clin = fd.get("clinical", {})
            docs = fd.get("documents", {})

            print(f"\n  {C.BOLD}{C.CYAN}Patient{C.RESET}")
            print(kv("  Name:", f"{pat.get('firstName', '')} {pat.get('lastName', '')}"))
            print(kv("  DOB:", pat.get("dob", "")))
            print(kv("  Member ID:", pat.get("memberId", "")))
            print(kv("  Plan:", pat.get("insurancePlan", "")))
            if pat.get("groupNumber"):
                print(kv("  Group #:", pat["groupNumber"]))

            print(f"\n  {C.BOLD}{C.CYAN}Provider{C.RESET}")
            print(kv("  Name:", prov.get("name", "")))
            print(kv("  NPI:", prov.get("npi", "")))
            print(kv("  Phone:", prov.get("phone", "")))

            print(f"\n  {C.BOLD}{C.CYAN}Medication{C.RESET}")
            print(kv("  Name:", med.get("name", "")))
            print(kv("  Category:", med.get("category", "")))
            print(kv("  NDC Code:", med.get("ndcCode", "")))
            print(kv("  Dosage:", med.get("dosage", "")))
            print(kv("  Frequency:", med.get("frequency", "")))
            print(kv("  Route:", med.get("route", "")))
            print(kv("  Duration:", med.get("duration", "")))

            print(f"\n  {C.BOLD}{C.CYAN}Clinical{C.RESET}")
            print(kv("  Primary Dx:", f"{clin.get('icd10Primary', '')} — {clin.get('diagDescription', '')}"))
            if clin.get("icd10Secondary"):
                print(kv("  Secondary Dx:", f"{clin.get('icd10Secondary', '')} — {clin.get('diagSecondary', '')}"))
            print(kv("  Urgency:", clin.get("urgency", "")))

            rationale = clin.get("clinicalRationale", "")
            if rationale:
                print(f"\n  {C.BOLD}{C.CYAN}Clinical Rationale{C.RESET}")
                print(wrap_text(rationale))

            # Documents
            clin_docs = docs.get("clinical", [])
            lab_docs = docs.get("lab", [])
            letter_docs = docs.get("letter", [])
            total_docs = len(clin_docs) + len(lab_docs) + len(letter_docs)

            if total_docs > 0:
                print(f"\n  {C.BOLD}{C.CYAN}Documents ({total_docs} uploaded){C.RESET}")
                for d in clin_docs:
                    print(f"    {C.TEAL}📋{C.RESET} [Clinical] {d}")
                for d in lab_docs:
                    print(f"    {C.TEAL}🧪{C.RESET} [Lab] {d}")
                for d in letter_docs:
                    print(f"    {C.TEAL}📝{C.RESET} [Letter] {d}")

            if docs.get("additionalNotes"):
                print(f"\n  {C.BOLD}{C.CYAN}Additional Notes{C.RESET}")
                print(wrap_text(docs["additionalNotes"]))

    def print_stats(self, result: AgentResult):
        """Print performance statistics."""
        print(section("PERFORMANCE METRICS"))

        # Tool call breakdown
        tool_counts: dict[str, int] = {}
        for tc in result.tool_calls:
            n = tc["name"]
            tool_counts[n] = tool_counts.get(n, 0) + 1

        print(f"\n  {C.BOLD}Tool Call Breakdown:{C.RESET}")
        tool_icons = {
            "read_page": "👁",
            "fill_field": "✎",
            "click_button": "⏎",
            "upload_file": "📎",
            "get_form_state": "📊",
        }
        max_count = max(tool_counts.values()) if tool_counts else 1
        for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            icon = tool_icons.get(name, "·")
            bar_len = int(30 * count / max_count)
            bar = f"{C.CYAN}{'█' * bar_len}{C.RESET}"
            print(f"    {icon} {name:<18} {bar} {C.BOLD}{count}{C.RESET}")

        print()
        total = len(result.tool_calls)
        print(kv("Total Tool Calls:", str(total)))
        print(kv("Agent Turns:", str(result.turns)))

        efficiency = total / max(result.turns, 1)
        print(kv("Calls/Turn:", f"{efficiency:.1f}"))

        err_rate = len(result.errors) / max(total, 1) * 100
        err_color = C.GREEN if err_rate < 5 else (C.YELLOW if err_rate < 15 else C.RED)
        print(kv("Error Rate:", f"{err_color}{err_rate:.1f}%{C.RESET}"))


# ---------------------------------------------------------------------------
# Main Demo Runner
# ---------------------------------------------------------------------------

async def run_single_demo(
    case_name: str,
    case_data: dict,
    runner: DemoAgentRunner,
    submit_to_server: bool = False,
    portal_url: str = "http://localhost:8080",
    model: str | None = None,
) -> AgentResult:
    """Run a single demo case and display results."""
    print()
    print(banner(f"PRIOR AUTHORIZATION AGENT DEMO", C.CYAN))
    print()

    # Show case overview
    runner.print_case_overview(case_data, case_name)

    # Animated launch
    print(section("LAUNCHING AGENT", C.GREEN))
    steps_label = ["Initializing agent...", "Connecting to Claude API...", "Starting form navigation..."]
    for i, label in enumerate(steps_label):
        print(progress_bar(i + 1, len(steps_label), label=label))
        await asyncio.sleep(0.15)

    print()
    print(f"  {C.BOLD}{C.GREEN}▶ Agent is running...{C.RESET}")
    print(f"  {C.DIM}(Agent logs appear below as it navigates the form){C.RESET}")
    print()

    # Run the agent
    agent_kwargs: dict = dict(verbose=True, submit=True)
    if model:
        agent_kwargs["model"] = model

    agent = PriorAuthAgent(**agent_kwargs)
    start_time = time.monotonic()
    result = await agent.run(case_data=case_data, case_name=case_name)
    elapsed = time.monotonic() - start_time

    print()

    # Optionally submit to server
    if submit_to_server and result.success and result.form_data:
        print(f"  {C.TEAL}Submitting to portal server at {portal_url}...{C.RESET}")
        from agents.prior_auth_agent import submit_to_server as post_to_server
        server_result = await post_to_server(result.form_data, portal_url)
        if "error" not in server_result:
            print(f"  {C.GREEN}✓ Server accepted submission{C.RESET}")
        else:
            print(f"  {C.YELLOW}⚠ Server submission failed: {server_result['error']}{C.RESET}")
        print()

    # Display results
    runner.print_agent_process(result)
    runner.print_result(result, case_data)
    runner.print_stats(result)

    # Elapsed time
    print(kv("Wall Time:", f"{elapsed:.1f}s"))

    # Footer
    print()
    print(f"{C.DIM}{'─' * W}{C.RESET}")
    print(f"{C.DIM}  Demo completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print()

    return result


async def run_all_cases(runner: DemoAgentRunner, **kwargs) -> dict[str, AgentResult]:
    """Run all sample cases sequentially."""
    results: dict[str, AgentResult] = {}

    print()
    print(banner("RUNNING ALL SAMPLE CASES (5 total)", C.MAGENTA))
    print()

    for i, (name, case_data) in enumerate(SAMPLE_CASES.items(), 1):
        print(f"\n{C.BOLD}{C.MAGENTA}━━━ Case {i}/5: {name} ━━━{C.RESET}\n")
        result = await run_single_demo(name, case_data, runner, **kwargs)
        results[name] = result

    # Summary table
    print()
    print(banner("ALL CASES — SUMMARY", C.CYAN))
    print()

    hdr = f"  {C.BOLD}{'Case':<22} {'Status':<12} {'Ref #':<18} {'Turns':<7} {'Calls':<7} {'Errors':<7}{C.RESET}"
    print(hdr)
    print(f"  {'─' * 70}")

    for name, r in results.items():
        status_color = C.GREEN if r.success else C.RED
        status = f"{status_color}{'OK' if r.success else 'FAIL'}{C.RESET}"
        ref = r.reference_number or "—"
        print(f"  {name:<22} {status:<21} {ref:<18} {r.turns:<7} {len(r.tool_calls):<7} {len(r.errors):<7}")

    successes = sum(1 for r in results.values() if r.success)
    total = len(results)
    rate_color = C.GREEN if successes == total else (C.YELLOW if successes > 0 else C.RED)
    print(f"\n  {C.BOLD}Success Rate: {rate_color}{successes}/{total} ({100 * successes // total}%){C.RESET}")
    print()

    return results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prior Authorization Agent — End-to-End Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python agents/run_prior_auth_demo.py                       # Specialty case
              python agents/run_prior_auth_demo.py --case controlled     # Controlled substance
              python agents/run_prior_auth_demo.py --case gene_therapy   # Gene therapy ($2.2M!)
              python agents/run_prior_auth_demo.py --all                 # All 5 cases
              python agents/run_prior_auth_demo.py --list                # List cases
              python agents/run_prior_auth_demo.py --with-server         # Also start HTTP portal
        """),
    )
    parser.add_argument(
        "--case", "-c",
        choices=list(SAMPLE_CASES.keys()),
        default="specialty",
        help="Sample case to run (default: specialty)",
    )
    parser.add_argument("--all", "-a", action="store_true", help="Run all 5 sample cases sequentially")
    parser.add_argument("--list", "-l", action="store_true", help="List available cases and exit")
    parser.add_argument(
        "--with-server", action="store_true",
        help="Start the portal HTTP server and POST results to it",
    )
    parser.add_argument("--port", type=int, default=8080, help="Portal server port (default: 8080)")
    parser.add_argument("--model", "-m", help="Override the Claude model to use")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    if args.no_color:
        for attr in dir(C):
            if not attr.startswith("_"):
                setattr(C, attr, "")

    if args.list:
        print()
        print(banner("AVAILABLE SAMPLE CASES", C.CYAN))
        print()
        for name, case in SAMPLE_CASES.items():
            p = case["patient"]
            m = case["medication"]
            cl = case["clinical"]
            urgency = cl.get("urgency", "routine")
            urg_icon = {"routine": "🟢", "urgent": "🟡", "emergency": "🔴"}.get(urgency, "⚪")
            print(f"  {C.BOLD}{C.CYAN}{name:<22}{C.RESET} {case['description']}")
            print(f"  {'':<22} Patient:    {p['firstName']} {p['lastName']}")
            print(f"  {'':<22} Medication: {m['name']}")
            print(f"  {'':<22} Urgency:    {urg_icon} {urgency}")
            print()
        return

    # Start portal server if requested
    portal = None
    if args.with_server:
        portal = PortalServer(port=args.port)
        ok = portal.start()
        if ok:
            print(f"{C.GREEN}✓{C.RESET} Portal server running at {C.ULINE}http://localhost:{args.port}{C.RESET}")
        else:
            print(f"{C.RED}✗{C.RESET} Failed to start portal server on port {args.port}")
            sys.exit(1)

    runner = DemoAgentRunner(verbose=True)

    extra_kwargs: dict = {}
    if args.model:
        extra_kwargs["model"] = args.model
    if args.with_server:
        extra_kwargs["submit_to_server"] = True
        extra_kwargs["portal_url"] = f"http://localhost:{args.port}"

    try:
        if args.all:
            asyncio.run(run_all_cases(runner, **extra_kwargs))
        else:
            case_data = SAMPLE_CASES[args.case]
            asyncio.run(run_single_demo(args.case, case_data, runner, **extra_kwargs))
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Demo interrupted.{C.RESET}")
    finally:
        if portal:
            portal.stop()


if __name__ == "__main__":
    main()
