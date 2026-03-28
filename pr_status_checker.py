"""
GitHub PR Status Checker for Jarvis Task Runner

Checks GitHub PR status (open/merged/closed) via the GitHub REST API and
stores the result as a `pr_status` field on DynamoDB task records.

This module is imported by task_runner.py — the authoritative list_tasks()
lives there and calls check_and_update_pr_statuses() automatically so that
PR filtering is never based on stale data.

Usage:
    from pr_status_checker import check_and_update_pr_statuses, get_pr_status

    # Update PR statuses for all completed tasks
    check_and_update_pr_statuses(dynamodb_table)

    # Check a single PR
    status = get_pr_status("https://github.com/owner/repo/pull/123")

Requires:
    - GITHUB_TOKEN environment variable (PAT with repo scope)
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



# ---------------------------------------------------------------------------
# CLI entrypoint for manual PR status checks
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
