# Mock Weather MCP Server

A mock MCP (Model Context Protocol) server exposing weather tools that return predefined data. Designed to be connected to a Claude Agent SDK agent.

## Tools

| Tool | Description |
|------|-------------|
| `get_current_weather` | Current conditions (temp, humidity, wind, UV, etc.) for a city |
| `get_forecast` | Multi-day forecast (1–7 days) with highs, lows, and precipitation |
| `get_weather_alerts` | Active severe weather advisories, watches, and warnings |

**Supported cities:** New York, San Francisco, Miami, Chicago, Denver (plus aliases like NYC, SF, etc.)

## Quick Start

```bash
# Install dependencies
pip install mcp anthropic

# Tool discovery + direct MCP calls (no API key needed)
python run_demo.py

# Discovery only
python run_demo.py --discover

# Full agent demo (requires ANTHROPIC_API_KEY)
python run_demo.py --agent
```

## Connecting to a Claude Agent

The server runs on stdio transport by default, making it compatible with any MCP client:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mock_weather_mcp/mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("get_current_weather", {"city": "NYC"})
```

## SSE Transport

```bash
python mcp_server.py --transport sse  # HTTP server on port 8000
```
