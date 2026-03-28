"""
GitHub PR Status Checker for Jarvis Task Runner

Adds PR status checking to the task management system:
- Checks GitHub PR status (open/merged/closed) via GitHub API
- Stores pr_status field on DynamoDB task records
- Filters merged/closed PRs from default list_tasks response

Integration: Import and call from task_runner.py's heartbeat or list flow.

Usage:
    from pr_status_checker import check_and_update_pr_statuses, list_tasks

    # Update PR statuses for all completed tasks
    check_and_update_pr_statuses(dynamodb_table)

    # List active tasks (default: queued/running/completed with open PRs)
    tasks = list_tasks(dynamodb_table)

    # List all tasks including archived and merged/closed PRs
    tasks = list_tasks(dynamodb_table, include_closed=True)

    # Filter by specific status
    tasks = list_tasks(dynamodb_table, status="archived")

Requires:
    - GITHUB_TOKEN environment variable (PAT with repo scope)
    - boto3 (already in task_runner.py)
    - urllib.request (stdlib, no new dependencies)
"""

import json
import logging
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
GITHUB_API_BASE = "https://api.github.com"

# PR status values stored in DynamoDB
PR_STATUS_OPEN = "open"
PR_STATUS_MERGED = "merged"
PR_STATUS_CLOSED = "closed"
PR_STATUS_UNKNOWN = "unknown"


def parse_pr_url(pr_url: str) -> tuple:
    """
    Extract owner, repo, and PR number from a GitHub PR URL.

    Supports formats:
        https://github.com/owner/repo/pull/123
        https://github.com/owner/repo/pull/123/files

    Returns:
        (owner, repo, pr_number) or None if URL is invalid.
    """
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url
    )
    if match:
        return match.group(1), match.group(2), int(match.group(3))
    return None


def get_pr_status(pr_url: str) -> str:
    """
    Check the status of a GitHub PR via the GitHub REST API.

    Args:
        pr_url: Full GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)

    Returns:
        One of: "open", "merged", "closed", "unknown"
    """
    parsed = parse_pr_url(pr_url)
    if not parsed:
        logger.warning(f"Could not parse PR URL: {pr_url}")
        return PR_STATUS_UNKNOWN

    owner, repo, pr_number = parsed
    api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "jarvis-task-runner",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        if data.get("merged"):
            return PR_STATUS_MERGED
        elif data.get("state") == "closed":
            return PR_STATUS_CLOSED
        elif data.get("state") == "open":
            return PR_STATUS_OPEN
        else:
            return PR_STATUS_UNKNOWN

    except urllib.error.HTTPError as e:
        logger.error(f"GitHub API error for {pr_url}: {e.code} {e.reason}")
        return PR_STATUS_UNKNOWN
    except urllib.error.URLError as e:
        logger.error(f"Network error checking PR status for {pr_url}: {e.reason}")
        return PR_STATUS_UNKNOWN
    except Exception as e:
        logger.error(f"Unexpected error checking PR status for {pr_url}: {e}")
        return PR_STATUS_UNKNOWN


def check_and_update_pr_statuses(table) -> dict:
    """
    Scan completed tasks with PR URLs and update their pr_status in DynamoDB.

    Only checks tasks that:
    - Have status="completed"
    - Have a non-empty pr_url
    - Either have no pr_status yet, or have pr_status="open" (re-check open PRs)

    Args:
        table: boto3 DynamoDB Table resource for jarvis-tasks

    Returns:
        dict with counts: {"checked": N, "updated": N, "errors": N}
    """
    stats = {"checked": 0, "updated": 0, "errors": 0}

    try:
        # Scan for completed tasks with PR URLs
        response = table.scan(
            FilterExpression="attribute_exists(pr_url) AND #s = :completed",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":completed": "completed"},
        )
        tasks = response.get("Items", [])

        # Handle pagination
        while "LastEvaluatedKey" in response:
            response = table.scan(
                FilterExpression="attribute_exists(pr_url) AND #s = :completed",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":completed": "completed"},
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            tasks.extend(response.get("Items", []))

    except Exception as e:
        logger.error(f"Failed to scan tasks table: {e}")
        return stats

    for task in tasks:
        pr_url = task.get("pr_url", "").strip()
        if not pr_url:
            continue

        current_pr_status = task.get("pr_status", "")

        # Skip tasks with a terminal pr_status (merged/closed) — no need to re-check
        if current_pr_status in (PR_STATUS_MERGED, PR_STATUS_CLOSED):
            continue

        stats["checked"] += 1
        new_status = get_pr_status(pr_url)

        if new_status == PR_STATUS_UNKNOWN:
            stats["errors"] += 1
            continue

        # Update DynamoDB if status changed or was never set
        if new_status != current_pr_status:
            try:
                table.update_item(
                    Key={"task_id": task["task_id"]},
                    UpdateExpression="SET pr_status = :ps, pr_status_updated_at = :ts",
                    ExpressionAttributeValues={
                        ":ps": new_status,
                        ":ts": datetime.now(timezone.utc).isoformat(),
                    },
                )
                stats["updated"] += 1
                logger.info(
                    f"Task {task['task_id']}: PR status "
                    f"{current_pr_status or '(none)'} -> {new_status}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to update pr_status for task {task['task_id']}: {e}"
                )
                stats["errors"] += 1

    logger.info(
        f"PR status check complete: {stats['checked']} checked, "
        f"{stats['updated']} updated, {stats['errors']} errors"
    )
    return stats


def list_tasks(
    table,
    include_closed: bool = False,
    status: str = None,
) -> list:
    """
    List tasks from DynamoDB with sensible defaults for an active task view.

    Default behavior:
        Only returns tasks with status in {queued, running, completed}.
        For completed tasks, only those with open PRs (or no PR) are shown —
        merged/closed PR tasks are hidden since they need no further action.
        This keeps the default view focused on work that still needs attention.

    Args:
        table: boto3 DynamoDB Table resource for jarvis-tasks.
        include_closed: If True, return ALL tasks regardless of status or
                        PR state. Includes archived, failed, cancelled tasks
                        and tasks with merged/closed PRs. Useful for
                        dashboards or auditing.
        status: Optional single-status filter (e.g., "running", "archived").
                When set, only tasks matching this exact status are returned.
                Setting status="archived" returns archived tasks even without
                include_closed=True.

    Returns:
        List of task dicts sorted by created_at descending.
    """
    # Default visible statuses: queued, running, completed
    # Failed, cancelled, and archived are hidden unless explicitly requested.
    DEFAULT_VISIBLE_STATUSES = {"queued", "running", "completed"}

    try:
        scan_kwargs = {}

        # If a specific status filter is provided, push it down to DynamoDB
        if status:
            scan_kwargs["FilterExpression"] = "#s = :status_val"
            scan_kwargs["ExpressionAttributeNames"] = {"#s": "status"}
            scan_kwargs["ExpressionAttributeValues"] = {":status_val": status}

        response = table.scan(**scan_kwargs)
        tasks = response.get("Items", [])

        # Handle pagination
        while "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = table.scan(**scan_kwargs)
            tasks.extend(response.get("Items", []))

    except Exception as e:
        logger.error(f"Failed to scan tasks table: {e}")
        return []

    # Apply default visibility filters when no explicit status or include_closed
    if not status and not include_closed:
        # Step 1: Only show queued/running/completed tasks
        tasks = [
            t for t in tasks
            if t.get("status") in DEFAULT_VISIBLE_STATUSES
        ]
        # Step 2: For completed tasks, hide those with merged/closed PRs
        # (they need no further action). Tasks with open PRs or no PR are kept.
        tasks = [
            t for t in tasks
            if t.get("pr_status") not in (PR_STATUS_MERGED, PR_STATUS_CLOSED)
        ]

    # Sort by created_at descending (most recent first)
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)

    return tasks


# ---------------------------------------------------------------------------
# Integration helpers — drop these into the existing task_runner.py
# ---------------------------------------------------------------------------

def integrate_with_heartbeat(table):
    """
    Call this from the heartbeat function in task_runner.py to periodically
    refresh PR statuses. Example integration:

        # In send_heartbeat() or the heartbeat loop:
        from pr_status_checker import integrate_with_heartbeat
        integrate_with_heartbeat(tasks_table)
    """
    logger.info("Heartbeat: checking PR statuses for completed tasks...")
    stats = check_and_update_pr_statuses(table)
    return stats


def integrate_with_briefing(table) -> str:
    """
    Generate a PR status summary for the daily briefing. Returns a formatted
    string block that can be appended to the existing briefing output.

    Example integration in generate_briefing():
        from pr_status_checker import integrate_with_briefing
        pr_summary = integrate_with_briefing(tasks_table)
        briefing_text += pr_summary
    """
    # First refresh statuses
    check_and_update_pr_statuses(table)

    # Get all completed tasks with PR URLs (include_closed to see merged/closed PRs too)
    all_tasks = list_tasks(table, include_closed=True, status="completed")
    pr_tasks = [t for t in all_tasks if t.get("pr_url")]

    if not pr_tasks:
        return "\n**PR Status:** No PRs to report.\n"

    open_prs = [t for t in pr_tasks if t.get("pr_status") == PR_STATUS_OPEN]
    merged_prs = [t for t in pr_tasks if t.get("pr_status") == PR_STATUS_MERGED]
    closed_prs = [t for t in pr_tasks if t.get("pr_status") == PR_STATUS_CLOSED]

    lines = ["\n**PR Status Summary:**"]
    lines.append(f"  - Open: {len(open_prs)} | Merged: {len(merged_prs)} | Closed: {len(closed_prs)}")

    if open_prs:
        lines.append("\n  **Awaiting review:**")
        for t in open_prs:
            desc = t.get("description", "")[:60]
            lines.append(f"  - [{t['task_id']}]({t.get('pr_url', '')}) — {desc}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entrypoint for manual checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import boto3
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    table_name = os.environ.get("TASKS_TABLE_NAME", "jarvis-tasks")
    dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    table = dynamodb.Table(table_name)

    print(f"Checking PR statuses in table: {table_name}")
    stats = check_and_update_pr_statuses(table)
    print(f"\nResults: {stats}")

    print("\n--- Active Tasks (default view: queued/running/completed with open PRs) ---")
    active = list_tasks(table)
    for t in active:
        pr_status = t.get("pr_status", "n/a")
        print(f"  {t['task_id']}  [{t.get('status')}]  pr={pr_status}  {t.get('description', '')[:50]}")

    print(f"\nTotal active tasks: {len(active)}")
    all_tasks = list_tasks(table, include_closed=True)
    print(f"Total tasks (including all statuses): {len(all_tasks)}")
