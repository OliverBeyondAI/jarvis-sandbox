"""
Jarvis Task Runner — Core task management system.

Manages task lifecycle in DynamoDB with statuses:
    queued -> running -> completed
                     -> failed
                     -> cancelled
                     -> archived

The `archived` status is a terminal state for tasks that have been
manually closed. Archived tasks are hidden from the default list view.

Usage:
    from task_runner import close_task, list_tasks, TASK_STATUS_ARCHIVED

    # Archive a completed/cancelled/failed task
    close_task(table, "task-abc123")

    # List active tasks (auto-refreshes PR statuses, hides merged/closed PRs)
    tasks = list_tasks(table)

    # List all tasks including archived and merged/closed PRs
    tasks = list_tasks(table, include_closed=True)

    # Filter by a specific status
    tasks = list_tasks(table, status="archived")
"""

import logging
import os
from datetime import datetime, timezone

from pr_status_checker import (
    check_and_update_pr_statuses,
    PR_STATUS_MERGED,
    PR_STATUS_CLOSED,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task status constants
# ---------------------------------------------------------------------------
TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_CANCELLED = "cancelled"
TASK_STATUS_ARCHIVED = "archived"

# Statuses that can be archived (terminal states only)
ARCHIVABLE_STATUSES = {
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_CANCELLED,
}

# Statuses shown in the default list view (active work only)
# Completed tasks are included because they may have open PRs awaiting review.
# Failed, cancelled, and archived tasks are hidden by default.
DEFAULT_VISIBLE_STATUSES = {
    TASK_STATUS_QUEUED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_COMPLETED,
}

TASKS_TABLE_NAME = os.environ.get("TASKS_TABLE_NAME", "jarvis-tasks")


# ---------------------------------------------------------------------------
# close_task — archive a task
# ---------------------------------------------------------------------------

def close_task(table, task_id: str, reason: str = "") -> dict:
    """
    Archive a task by setting its status to 'archived' in DynamoDB.

    Only tasks in a terminal state (completed, failed, cancelled) can be
    archived. Running or queued tasks must be cancelled first.

    Args:
        table: boto3 DynamoDB Table resource for jarvis-tasks.
        task_id: The ID of the task to archive.
        reason: Optional reason for archiving.

    Returns:
        dict with keys:
            - success (bool)
            - message (str)
            - task_id (str)
            - previous_status (str | None)
    """
    # Fetch the task first to validate it exists and check its status
    try:
        response = table.get_item(Key={"task_id": task_id})
    except Exception as e:
        logger.error(f"Failed to fetch task {task_id}: {e}")
        return {
            "success": False,
            "message": f"DynamoDB error: {e}",
            "task_id": task_id,
            "previous_status": None,
        }

    task = response.get("Item")
    if not task:
        return {
            "success": False,
            "message": f"Task not found: {task_id}",
            "task_id": task_id,
            "previous_status": None,
        }

    current_status = task.get("status", "")

    # Already archived — no-op success
    if current_status == TASK_STATUS_ARCHIVED:
        return {
            "success": True,
            "message": "Task is already archived.",
            "task_id": task_id,
            "previous_status": TASK_STATUS_ARCHIVED,
        }

    # Only terminal statuses can be archived
    if current_status not in ARCHIVABLE_STATUSES:
        return {
            "success": False,
            "message": (
                f"Cannot archive task with status '{current_status}'. "
                f"Only tasks with status {sorted(ARCHIVABLE_STATUSES)} can be archived. "
                f"Cancel the task first if it is still running or queued."
            ),
            "task_id": task_id,
            "previous_status": current_status,
        }

    # Perform the update
    now = datetime.now(timezone.utc).isoformat()
    update_expr = "SET #s = :archived, archived_at = :ts, previous_status = :prev"
    expr_values = {
        ":archived": TASK_STATUS_ARCHIVED,
        ":ts": now,
        ":prev": current_status,
    }

    if reason:
        update_expr += ", archive_reason = :reason"
        expr_values[":reason"] = reason

    try:
        table.update_item(
            Key={"task_id": task_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=expr_values,
        )
    except Exception as e:
        logger.error(f"Failed to archive task {task_id}: {e}")
        return {
            "success": False,
            "message": f"DynamoDB update error: {e}",
            "task_id": task_id,
            "previous_status": current_status,
        }

    logger.info(f"Task {task_id} archived (was: {current_status})")
    return {
        "success": True,
        "message": f"Task archived successfully (was: {current_status}).",
        "task_id": task_id,
        "previous_status": current_status,
    }


# ---------------------------------------------------------------------------
# list_tasks — query tasks with archive filtering
# ---------------------------------------------------------------------------

def list_tasks(
    table,
    include_closed: bool = False,
    status: str = None,
) -> list:
    """
    List tasks from DynamoDB with sensible defaults for an active task view.

    Default behavior:
        - Automatically refreshes PR statuses for completed tasks via the
          GitHub API (so filtering is never based on stale data).
        - Only returns tasks with status in {queued, running, completed}.
        - For completed tasks, hides those whose PR has been merged or closed
          (they need no further action).
        This keeps the default view focused on work that still needs attention.

    Args:
        table: boto3 DynamoDB Table resource for jarvis-tasks.
        include_closed: If True, return ALL tasks regardless of status or
                        PR state. Includes archived, failed, cancelled tasks
                        and tasks with merged/closed PRs. Useful for
                        dashboards or auditing.
        status: Optional single-status filter (e.g., "running", "archived").
                When set, only tasks matching this exact status are returned.
                Setting status="archived" bypasses the default filter so
                archived tasks are returned even without include_closed.

    Returns:
        List of task dicts sorted by created_at descending.
    """
    # Refresh PR statuses before filtering so we never act on stale data.
    # This is a no-op for tasks whose PR is already in a terminal state
    # (merged/closed) — only open/unknown PRs are re-checked.
    if not include_closed:
        try:
            check_and_update_pr_statuses(table)
        except Exception as e:
            logger.warning(f"PR status refresh failed (listing with cached data): {e}")

    try:
        scan_kwargs = {}

        # If a specific status filter is provided, push it down to DynamoDB
        if status:
            scan_kwargs["FilterExpression"] = "#s = :status_val"
            scan_kwargs["ExpressionAttributeNames"] = {"#s": "status"}
            scan_kwargs["ExpressionAttributeValues"] = {":status_val": status}

        response = table.scan(**scan_kwargs)
        tasks = response.get("Items", [])

        while "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = table.scan(**scan_kwargs)
            tasks.extend(response.get("Items", []))

    except Exception as e:
        logger.error(f"Failed to scan tasks table: {e}")
        return []

    # Apply default visibility filters when no explicit overrides are set.
    if not status and not include_closed:
        # Only show queued/running/completed tasks
        tasks = [
            t for t in tasks
            if t.get("status") in DEFAULT_VISIBLE_STATUSES
        ]
        # For completed tasks, hide those with merged/closed PRs
        # (they need no further action). Tasks with open PRs or no PR are kept.
        tasks = [
            t for t in tasks
            if t.get("pr_status") not in (PR_STATUS_MERGED, PR_STATUS_CLOSED)
        ]

    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return tasks


# ---------------------------------------------------------------------------
# bulk_archive — archive multiple tasks at once
# ---------------------------------------------------------------------------

def bulk_archive(table, task_ids: list, reason: str = "") -> dict:
    """
    Archive multiple tasks. Skips tasks that cannot be archived.

    Args:
        table: boto3 DynamoDB Table resource.
        task_ids: List of task IDs to archive.
        reason: Optional reason applied to all.

    Returns:
        dict with counts: {"archived": N, "skipped": N, "errors": N, "details": [...]}
    """
    stats = {"archived": 0, "skipped": 0, "errors": 0, "details": []}

    for task_id in task_ids:
        result = close_task(table, task_id, reason=reason)
        stats["details"].append(result)
        if result["success"]:
            stats["archived"] += 1
        elif "Cannot archive" in result["message"] or "not found" in result["message"]:
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    logger.info(
        f"Bulk archive: {stats['archived']} archived, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import boto3
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Jarvis Task Runner — task management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # close/archive command
    close_parser = subparsers.add_parser("close", help="Archive a task")
    close_parser.add_argument("task_id", help="Task ID to archive")
    close_parser.add_argument("--reason", default="", help="Reason for archiving")

    # list command
    list_parser = subparsers.add_parser("list", help="List tasks")
    list_parser.add_argument(
        "--include-closed", action="store_true",
        help="Show all tasks including archived, failed, and cancelled",
    )
    list_parser.add_argument(
        "--status", default=None,
        help="Filter by status (queued, running, completed, failed, cancelled, archived)",
    )

    args = parser.parse_args()

    table_name = os.environ.get("TASKS_TABLE_NAME", "jarvis-tasks")
    dynamodb = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    table = dynamodb.Table(table_name)

    if args.command == "close":
        result = close_task(table, args.task_id, reason=args.reason)
        if result["success"]:
            print(f"OK: {result['message']}")
        else:
            print(f"FAILED: {result['message']}")

    elif args.command == "list":
        tasks = list_tasks(table, include_closed=args.include_closed, status=args.status)

        if not tasks:
            print("No tasks found.")
        else:
            for t in tasks:
                status_str = t.get("status", "?")
                desc = t.get("description", "")[:60]
                pr_status = t.get("pr_status", "")
                extra = ""
                if status_str == TASK_STATUS_ARCHIVED:
                    extra = f"  (archived: {t.get('archived_at', '')})"
                elif pr_status:
                    extra = f"  (pr: {pr_status})"
                print(f"  {t['task_id']}  [{status_str}]  {desc}{extra}")
            print(f"\nTotal: {len(tasks)} task(s)")

    else:
        parser.print_help()
