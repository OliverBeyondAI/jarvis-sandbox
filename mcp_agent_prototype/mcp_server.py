#!/usr/bin/env python3
"""
MCP Server exposing mock tools (search_database, send_email) via the Model Context Protocol.

Run with:
    python mcp_server.py                  # stdio transport (for MCP clients)
    python mcp_server.py --transport sse  # SSE transport (HTTP, port 8000)
"""

import json
import random
import sys
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="mock-tools-server",
    instructions=(
        "This server provides two mock enterprise tools: search_database for "
        "querying a simulated customer/product database, and send_email for "
        "composing and dispatching emails through a simulated SMTP gateway."
    ),
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
_MOCK_CUSTOMERS = [
    {"id": "CUST-001", "name": "Acme Corp", "industry": "Manufacturing", "plan": "Enterprise", "mrr": 12500, "status": "active", "account_manager": "Sarah Chen"},
    {"id": "CUST-002", "name": "Globex Inc", "industry": "Technology", "plan": "Pro", "mrr": 4200, "status": "active", "account_manager": "James Rivera"},
    {"id": "CUST-003", "name": "Initech LLC", "industry": "Finance", "plan": "Enterprise", "mrr": 18000, "status": "active", "account_manager": "Sarah Chen"},
    {"id": "CUST-004", "name": "Umbrella Corp", "industry": "Healthcare", "plan": "Starter", "mrr": 990, "status": "churned", "account_manager": "James Rivera"},
    {"id": "CUST-005", "name": "Stark Industries", "industry": "Technology", "plan": "Enterprise", "mrr": 25000, "status": "active", "account_manager": "Maria Lopez"},
    {"id": "CUST-006", "name": "Wayne Enterprises", "industry": "Finance", "plan": "Pro", "mrr": 7800, "status": "active", "account_manager": "Maria Lopez"},
    {"id": "CUST-007", "name": "Cyberdyne Systems", "industry": "Technology", "plan": "Pro", "mrr": 5600, "status": "trial", "account_manager": "Sarah Chen"},
    {"id": "CUST-008", "name": "Soylent Corp", "industry": "Food & Beverage", "plan": "Starter", "mrr": 1200, "status": "active", "account_manager": "James Rivera"},
]

_MOCK_PRODUCTS = [
    {"sku": "PROD-100", "name": "Analytics Dashboard", "category": "Software", "price": 299.00, "stock": 999, "status": "available"},
    {"sku": "PROD-101", "name": "Data Pipeline Connector", "category": "Software", "price": 149.00, "stock": 999, "status": "available"},
    {"sku": "PROD-102", "name": "Enterprise SSO Module", "category": "Add-on", "price": 89.00, "stock": 999, "status": "available"},
    {"sku": "PROD-103", "name": "Custom Report Builder", "category": "Software", "price": 199.00, "stock": 999, "status": "beta"},
    {"sku": "PROD-104", "name": "API Rate Limit Upgrade", "category": "Add-on", "price": 49.00, "stock": 999, "status": "available"},
    {"sku": "PROD-105", "name": "Dedicated Support Package", "category": "Service", "price": 500.00, "stock": 50, "status": "available"},
]

_SENT_EMAILS: list[dict] = []

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_database(
    query: str,
    table: str = "customers",
    filters: str | None = None,
    limit: int = 10,
) -> str:
    """Search the company database for records matching a query.

    Searches across customer accounts, product catalog, or order history.
    Supports natural-language queries that are matched against record fields,
    plus optional structured filters.

    Args:
        query: Natural-language search query (e.g. "enterprise customers in tech",
               "products under $200", "churned accounts"). Matched against all
               text fields in the target table.
        table: Database table to search. One of:
               - "customers" — CRM accounts with plan, MRR, status, and manager info.
               - "products"  — Product catalog with SKU, pricing, and availability.
               Defaults to "customers".
        filters: Optional JSON-encoded filter object for structured queries.
                 Supported keys depend on the table:
                   customers: {"status": "active|churned|trial",
                               "plan": "Starter|Pro|Enterprise",
                               "industry": str}
                   products:  {"category": "Software|Add-on|Service",
                               "status": "available|beta|discontinued",
                               "max_price": float}
                 Example: '{"status": "active", "plan": "Enterprise"}'
        limit: Maximum number of results to return (1–100). Defaults to 10.

    Returns:
        JSON string with keys:
        - query: the original query
        - table: the table searched
        - total_results: number of matching records
        - results: list of matching record objects
        - searched_at: ISO-8601 timestamp of the search
    """
    limit = max(1, min(limit, 100))

    parsed_filters: dict = {}
    if filters:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in 'filters' parameter", "hint": "Provide a valid JSON string"})

    query_lower = query.lower()

    if table == "customers":
        results = _search_customers(query_lower, parsed_filters)
    elif table == "products":
        results = _search_products(query_lower, parsed_filters)
    else:
        return json.dumps({"error": f"Unknown table '{table}'", "valid_tables": ["customers", "products"]})

    results = results[:limit]

    return json.dumps({
        "query": query,
        "table": table,
        "total_results": len(results),
        "results": results,
        "searched_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2)


def _search_customers(query: str, filters: dict) -> list[dict]:
    results = []
    for c in _MOCK_CUSTOMERS:
        # Filter checks
        if "status" in filters and c["status"] != filters["status"]:
            continue
        if "plan" in filters and c["plan"] != filters["plan"]:
            continue
        if "industry" in filters and c["industry"].lower() != filters["industry"].lower():
            continue

        # Text matching across all string fields
        searchable = " ".join(str(v) for v in c.values()).lower()
        if query and not any(term in searchable for term in query.split()):
            continue

        results.append(c)
    return results


def _search_products(query: str, filters: dict) -> list[dict]:
    results = []
    for p in _MOCK_PRODUCTS:
        if "category" in filters and p["category"] != filters["category"]:
            continue
        if "status" in filters and p["status"] != filters["status"]:
            continue
        if "max_price" in filters and p["price"] > float(filters["max_price"]):
            continue

        searchable = " ".join(str(v) for v in p.values()).lower()
        if query and not any(term in searchable for term in query.split()):
            continue

        results.append(p)
    return results


@mcp.tool()
def send_email(
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
    bcc: str | None = None,
    priority: str = "normal",
    reply_to: str | None = None,
) -> str:
    """Send an email through the company's simulated SMTP gateway.

    Composes and dispatches an email message. In this mock environment, emails
    are logged internally rather than actually transmitted. Returns a delivery
    confirmation with a tracking message ID.

    Args:
        to: Recipient email address(es). For multiple recipients, separate with
            commas (e.g. "alice@example.com, bob@example.com").
        subject: Email subject line. Should be concise and descriptive.
        body: Email body content. Supports plain text. For longer messages,
              use newlines for paragraph separation.
        cc: Optional CC recipient(s), comma-separated.
        bcc: Optional BCC recipient(s), comma-separated.
        priority: Email priority level. One of "low", "normal", or "high".
                  High-priority emails are flagged in the recipient's inbox.
                  Defaults to "normal".
        reply_to: Optional Reply-To address if different from the sender.

    Returns:
        JSON string with keys:
        - status: "sent" on success
        - message_id: unique tracking ID for the email
        - to / cc / bcc: resolved recipients
        - subject: the subject line
        - priority: the priority level
        - sent_at: ISO-8601 timestamp
        - note: reminder that this is a simulated environment
    """
    if priority not in ("low", "normal", "high"):
        return json.dumps({"error": f"Invalid priority '{priority}'", "valid_values": ["low", "normal", "high"]})

    recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
    if not recipients:
        return json.dumps({"error": "No valid recipient addresses provided"})

    for addr in recipients:
        if "@" not in addr:
            return json.dumps({"error": f"Invalid email address: '{addr}'", "hint": "Provide a valid email with @ symbol"})

    message_id = f"MSG-{random.randint(100000, 999999)}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    email_record = {
        "message_id": message_id,
        "to": recipients,
        "cc": [a.strip() for a in cc.split(",") if a.strip()] if cc else [],
        "bcc": [a.strip() for a in bcc.split(",") if a.strip()] if bcc else [],
        "subject": subject,
        "body": body,
        "priority": priority,
        "reply_to": reply_to,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    _SENT_EMAILS.append(email_record)

    return json.dumps({
        "status": "sent",
        "message_id": message_id,
        "to": recipients,
        "cc": email_record["cc"],
        "bcc": email_record["bcc"],
        "subject": subject,
        "priority": priority,
        "sent_at": email_record["sent_at"],
        "note": "This is a simulated email — no actual message was transmitted.",
    }, indent=2)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    mcp.run(transport=transport)
