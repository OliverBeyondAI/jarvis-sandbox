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
from pathlib import Path
from zoneinfo import ZoneInfo

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

# ---------------------------------------------------------------------------
# Briefing schedule constants
# ---------------------------------------------------------------------------
TIMEZONE = ZoneInfo("America/New_York")
MORNING_BRIEFING_HOUR = 6   # 6 AM Eastern
EVENING_SUMMARY_HOUR = 21   # 9 PM Eastern

# Marker file directory — lightweight local tracking of sent briefings
BRIEFING_MARKER_DIR = Path(os.environ.get(
    "BRIEFING_MARKER_DIR",
    Path.home() / ".jarvis" / "briefing-markers",
))

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
# Briefing catch-up mechanism
# ---------------------------------------------------------------------------
#
# If the 6 AM ET morning-briefing window is missed (e.g. the machine was
# asleep), the catch-up logic sends the briefing as soon as the worker
# comes back online — but only if it hasn't already been sent that day.
#
# Dual tracking:
#   1. DynamoDB marker  (briefing-morning-YYYY-MM-DD) — authoritative, shared
#   2. Local marker file (~/.jarvis/briefing-markers/morning-YYYY-MM-DD.sent)
#      — fast offline check, avoids a DynamoDB call on every loop iteration
# ---------------------------------------------------------------------------


def _marker_path(briefing_type: str, date_str: str) -> Path:
    """Return the path for a local date-stamped marker file."""
    return BRIEFING_MARKER_DIR / f"{briefing_type}-{date_str}.sent"


def _briefing_sent_locally(briefing_type: str, date_str: str) -> bool:
    """Check the local marker file to see if a briefing was already sent today."""
    return _marker_path(briefing_type, date_str).exists()


def _mark_briefing_sent_locally(briefing_type: str, date_str: str) -> None:
    """Create a local marker file to record that a briefing was sent."""
    BRIEFING_MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker = _marker_path(briefing_type, date_str)
    marker.write_text(datetime.now(TIMEZONE).isoformat())
    logger.info(f"Local marker created: {marker}")


def _briefing_sent_in_dynamo(table, briefing_type: str, date_str: str) -> bool:
    """Check DynamoDB for the date-stamped briefing marker (authoritative)."""
    briefing_key = f"briefing-{briefing_type}-{date_str}"
    try:
        response = table.get_item(Key={"task_id": briefing_key})
        return bool(response.get("Item"))
    except Exception as e:
        logger.warning(f"DynamoDB check for {briefing_key} failed: {e}")
        return False


def _record_briefing_in_dynamo(table, briefing_type: str, date_str: str) -> None:
    """Write the date-stamped briefing marker to DynamoDB."""
    briefing_key = f"briefing-{briefing_type}-{date_str}"
    now = datetime.now(TIMEZONE).isoformat()
    try:
        table.put_item(Item={
            "task_id": briefing_key,
            "status": "sent",
            "created_at": now,
            "updated_at": now,
        })
        logger.info(f"DynamoDB marker written: {briefing_key}")
    except Exception as e:
        logger.error(f"Failed to write DynamoDB marker {briefing_key}: {e}")


def briefing_already_sent(table, briefing_type: str, date_str: str) -> bool:
    """
    Check whether a briefing has already been sent today.

    Uses a fast local marker file first, falling back to a DynamoDB query.
    If DynamoDB says it was sent but the local marker is missing (e.g.
    marker dir was cleared), re-create the local marker for consistency.
    """
    # Fast path — local file exists
    if _briefing_sent_locally(briefing_type, date_str):
        return True

    # Slow path — check DynamoDB (authoritative)
    if _briefing_sent_in_dynamo(table, briefing_type, date_str):
        # Re-sync local marker
        _mark_briefing_sent_locally(briefing_type, date_str)
        return True

    return False


def check_and_send_missed_briefings(table, send_briefing_fn) -> list:
    """
    Catch-up mechanism: detect and send any missed briefings for today.

    Call this when the worker starts up or resumes from sleep. It checks
    whether each scheduled briefing's window has already passed and, if
    the briefing wasn't sent, triggers it immediately.

    Args:
        table: boto3 DynamoDB Table resource for jarvis-tasks.
        send_briefing_fn: Callable(briefing_type: str) -> bool
            A function that generates and sends the briefing.
            Should return True on success, False on failure.
            Example: lambda btype: maybe_send_briefing(btype)

    Returns:
        List of briefing types that were caught up (e.g. ["morning"]).
    """
    now = datetime.now(TIMEZONE)
    today = now.strftime("%Y-%m-%d")
    current_hour = now.hour
    caught_up = []

    briefings = [
        ("morning", MORNING_BRIEFING_HOUR),
        ("evening", EVENING_SUMMARY_HOUR),
    ]

    for briefing_type, scheduled_hour in briefings:
        # Only catch up if we're past the scheduled hour
        if current_hour < scheduled_hour:
            continue

        # Skip if already sent today
        if briefing_already_sent(table, briefing_type, today):
            logger.debug(f"{briefing_type} briefing already sent for {today}")
            continue

        # Missed window — send now
        logger.info(
            f"Catch-up: {briefing_type} briefing was missed "
            f"(scheduled hour {scheduled_hour}, now {current_hour}). "
            f"Sending now."
        )

        try:
            success = send_briefing_fn(briefing_type)
        except Exception:
            logger.exception(f"Catch-up: failed to send {briefing_type} briefing")
            continue

        if success:
            _mark_briefing_sent_locally(briefing_type, today)
            _record_briefing_in_dynamo(table, briefing_type, today)
            caught_up.append(briefing_type)
            logger.info(f"Catch-up: {briefing_type} briefing sent successfully")
        else:
            logger.warning(f"Catch-up: {briefing_type} briefing send returned failure")

    return caught_up


def cleanup_old_markers(days_to_keep: int = 7) -> int:
    """
    Remove local marker files older than `days_to_keep` days.

    Call periodically (e.g. once per day) to prevent marker file buildup.
    Returns the number of files removed.
    """
    if not BRIEFING_MARKER_DIR.exists():
        return 0

    now = datetime.now(TIMEZONE)
    removed = 0

    for marker in BRIEFING_MARKER_DIR.glob("*.sent"):
        try:
            # Extract date from filename: e.g. "morning-2026-04-25.sent"
            parts = marker.stem.rsplit("-", 3)  # type, YYYY, MM, DD
            if len(parts) >= 4:
                date_str = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
                marker_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=TIMEZONE
                )
                age_days = (now - marker_date).days
                if age_days > days_to_keep:
                    marker.unlink()
                    removed += 1
                    logger.debug(f"Removed old marker: {marker.name}")
        except (ValueError, IndexError):
            logger.debug(f"Skipping unparseable marker: {marker.name}")

    if removed:
        logger.info(f"Cleaned up {removed} old briefing marker(s)")
    return removed


# ---------------------------------------------------------------------------
# Startup integration — call from the worker's main loop
# ---------------------------------------------------------------------------


def on_worker_startup(table, send_briefing_fn) -> None:
    """
    Run once when the jarvis-worker process starts or resumes from sleep.

    Performs catch-up for any missed briefings and cleans up stale markers.
    Wire this into the worker's initialization sequence:

        from task_runner import on_worker_startup
        on_worker_startup(table, send_briefing_fn=my_send_fn)

    Args:
        table: boto3 DynamoDB Table resource for jarvis-tasks.
        send_briefing_fn: Callable(briefing_type: str) -> bool
    """
    caught = check_and_send_missed_briefings(table, send_briefing_fn)
    if caught:
        logger.info(f"Worker startup catch-up sent: {', '.join(caught)}")
    cleanup_old_markers()


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

    # catchup command — check for missed briefings
    catchup_parser = subparsers.add_parser(
        "catchup", help="Check for and send any missed briefings"
    )
    catchup_parser.add_argument(
        "--dry-run", action="store_true",
        help="Only check — don't actually send briefings",
    )

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

    if args.command == "catchup":
        now = datetime.now(TIMEZONE)
        today = now.strftime("%Y-%m-%d")
        print(f"Checking for missed briefings ({now.strftime('%Y-%m-%d %H:%M %Z')})...")

        if args.dry_run:
            # Dry-run: report status without sending anything
            briefings = [
                ("morning", MORNING_BRIEFING_HOUR),
                ("evening", EVENING_SUMMARY_HOUR),
            ]
            any_missed = False
            for btype, hour in briefings:
                if now.hour < hour:
                    print(f"  {btype}: not yet due (scheduled at {hour}:00 ET)")
                elif briefing_already_sent(table, btype, today):
                    print(f"  {btype}: already sent for {today}")
                else:
                    any_missed = True
                    print(f"  {btype}: MISSED — would send now (dry-run)")
            if not any_missed:
                print("All briefings are up to date.")
        else:
            # Live mode: use the real catch-up mechanism.
            # Import the worker's send function if available; otherwise
            # abort rather than silently marking briefings as sent.
            try:
                from jarvis_worker import send_briefing as _real_send
            except ImportError:
                print(
                    "ERROR: Could not import send_briefing from jarvis_worker.\n"
                    "The catchup command must be run from an environment where "
                    "the jarvis-worker package is importable.\n"
                    "Use --dry-run to check status without sending."
                )
                raise SystemExit(1)

            caught = check_and_send_missed_briefings(table, _real_send)
            if caught:
                print(f"Caught up: {', '.join(caught)}")
            else:
                print("All briefings are up to date.")

    elif args.command == "close":
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
