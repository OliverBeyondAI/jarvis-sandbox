#!/usr/bin/env python3
"""
Mock Weather MCP Server — exposes weather tools via the Model Context Protocol.

Provides three tools returning predefined weather data for demonstration:
  - get_current_weather: current conditions for a city
  - get_forecast: multi-day forecast
  - get_weather_alerts: active severe weather alerts

Run with:
    python mcp_server.py                  # stdio transport (for MCP clients)
    python mcp_server.py --transport sse  # SSE transport (HTTP, port 8000)
"""

import json
import sys
from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="mock-weather-server",
    instructions=(
        "This server provides mock weather data tools: get_current_weather "
        "for real-time conditions, get_forecast for multi-day predictions, "
        "and get_weather_alerts for active severe weather advisories. "
        "All data is simulated for demonstration purposes."
    ),
)

# ---------------------------------------------------------------------------
# Mock weather data
# ---------------------------------------------------------------------------
_CITIES = {
    "new york": {
        "city": "New York",
        "state": "NY",
        "country": "US",
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York",
        "current": {
            "temp_f": 72,
            "temp_c": 22,
            "feels_like_f": 74,
            "feels_like_c": 23,
            "humidity": 58,
            "wind_mph": 12,
            "wind_direction": "SW",
            "condition": "Partly Cloudy",
            "uv_index": 6,
            "visibility_miles": 10,
            "pressure_mb": 1015,
        },
        "forecast": [
            {"day": 0, "high_f": 75, "low_f": 62, "condition": "Partly Cloudy", "precip_chance": 10, "humidity": 55},
            {"day": 1, "high_f": 78, "low_f": 64, "condition": "Sunny", "precip_chance": 5, "humidity": 50},
            {"day": 2, "high_f": 80, "low_f": 66, "condition": "Sunny", "precip_chance": 0, "humidity": 45},
            {"day": 3, "high_f": 76, "low_f": 63, "condition": "Thunderstorms", "precip_chance": 70, "humidity": 72},
            {"day": 4, "high_f": 71, "low_f": 58, "condition": "Overcast", "precip_chance": 30, "humidity": 65},
            {"day": 5, "high_f": 73, "low_f": 60, "condition": "Partly Cloudy", "precip_chance": 15, "humidity": 58},
            {"day": 6, "high_f": 77, "low_f": 63, "condition": "Sunny", "precip_chance": 5, "humidity": 48},
        ],
        "alerts": [
            {
                "type": "Heat Advisory",
                "severity": "moderate",
                "headline": "Heat Advisory in effect from Wednesday afternoon through Thursday evening",
                "description": "Temperatures expected to reach the upper 90s with heat index values up to 105°F.",
                "expires_in_hours": 48,
            }
        ],
    },
    "san francisco": {
        "city": "San Francisco",
        "state": "CA",
        "country": "US",
        "lat": 37.7749,
        "lon": -122.4194,
        "timezone": "America/Los_Angeles",
        "current": {
            "temp_f": 61,
            "temp_c": 16,
            "feels_like_f": 59,
            "feels_like_c": 15,
            "humidity": 78,
            "wind_mph": 18,
            "wind_direction": "W",
            "condition": "Foggy",
            "uv_index": 3,
            "visibility_miles": 4,
            "pressure_mb": 1018,
        },
        "forecast": [
            {"day": 0, "high_f": 63, "low_f": 54, "condition": "Foggy", "precip_chance": 5, "humidity": 80},
            {"day": 1, "high_f": 65, "low_f": 55, "condition": "Partly Cloudy", "precip_chance": 10, "humidity": 72},
            {"day": 2, "high_f": 68, "low_f": 56, "condition": "Sunny", "precip_chance": 0, "humidity": 60},
            {"day": 3, "high_f": 66, "low_f": 55, "condition": "Foggy", "precip_chance": 5, "humidity": 82},
            {"day": 4, "high_f": 64, "low_f": 53, "condition": "Overcast", "precip_chance": 20, "humidity": 75},
            {"day": 5, "high_f": 62, "low_f": 52, "condition": "Light Rain", "precip_chance": 60, "humidity": 85},
            {"day": 6, "high_f": 65, "low_f": 54, "condition": "Partly Cloudy", "precip_chance": 15, "humidity": 70},
        ],
        "alerts": [],
    },
    "miami": {
        "city": "Miami",
        "state": "FL",
        "country": "US",
        "lat": 25.7617,
        "lon": -80.1918,
        "timezone": "America/New_York",
        "current": {
            "temp_f": 88,
            "temp_c": 31,
            "feels_like_f": 95,
            "feels_like_c": 35,
            "humidity": 75,
            "wind_mph": 8,
            "wind_direction": "SE",
            "condition": "Scattered Thunderstorms",
            "uv_index": 9,
            "visibility_miles": 8,
            "pressure_mb": 1012,
        },
        "forecast": [
            {"day": 0, "high_f": 90, "low_f": 78, "condition": "Scattered Thunderstorms", "precip_chance": 60, "humidity": 78},
            {"day": 1, "high_f": 91, "low_f": 79, "condition": "Partly Cloudy", "precip_chance": 40, "humidity": 72},
            {"day": 2, "high_f": 92, "low_f": 80, "condition": "Thunderstorms", "precip_chance": 80, "humidity": 82},
            {"day": 3, "high_f": 89, "low_f": 77, "condition": "Tropical Storm Warning", "precip_chance": 90, "humidity": 88},
            {"day": 4, "high_f": 85, "low_f": 76, "condition": "Heavy Rain", "precip_chance": 95, "humidity": 92},
            {"day": 5, "high_f": 87, "low_f": 77, "condition": "Scattered Showers", "precip_chance": 50, "humidity": 80},
            {"day": 6, "high_f": 90, "low_f": 79, "condition": "Partly Cloudy", "precip_chance": 30, "humidity": 70},
        ],
        "alerts": [
            {
                "type": "Tropical Storm Warning",
                "severity": "high",
                "headline": "Tropical Storm Warning in effect through Friday",
                "description": "Tropical Storm Andrea approaching from the southeast. Expect sustained winds of 50-65 mph with gusts to 80 mph. Storm surge of 2-4 feet possible along the coast.",
                "expires_in_hours": 72,
            },
            {
                "type": "Flood Watch",
                "severity": "moderate",
                "headline": "Flood Watch in effect through Saturday morning",
                "description": "Heavy rainfall of 4-8 inches expected. Flash flooding possible in low-lying areas.",
                "expires_in_hours": 96,
            },
        ],
    },
    "chicago": {
        "city": "Chicago",
        "state": "IL",
        "country": "US",
        "lat": 41.8781,
        "lon": -87.6298,
        "timezone": "America/Chicago",
        "current": {
            "temp_f": 65,
            "temp_c": 18,
            "feels_like_f": 62,
            "feels_like_c": 17,
            "humidity": 52,
            "wind_mph": 22,
            "wind_direction": "NW",
            "condition": "Windy & Clear",
            "uv_index": 5,
            "visibility_miles": 10,
            "pressure_mb": 1020,
        },
        "forecast": [
            {"day": 0, "high_f": 67, "low_f": 52, "condition": "Windy & Clear", "precip_chance": 0, "humidity": 50},
            {"day": 1, "high_f": 70, "low_f": 55, "condition": "Sunny", "precip_chance": 5, "humidity": 45},
            {"day": 2, "high_f": 72, "low_f": 58, "condition": "Partly Cloudy", "precip_chance": 15, "humidity": 55},
            {"day": 3, "high_f": 68, "low_f": 54, "condition": "Rain", "precip_chance": 65, "humidity": 70},
            {"day": 4, "high_f": 64, "low_f": 50, "condition": "Overcast", "precip_chance": 25, "humidity": 60},
            {"day": 5, "high_f": 66, "low_f": 51, "condition": "Partly Cloudy", "precip_chance": 10, "humidity": 52},
            {"day": 6, "high_f": 71, "low_f": 56, "condition": "Sunny", "precip_chance": 5, "humidity": 45},
        ],
        "alerts": [],
    },
    "denver": {
        "city": "Denver",
        "state": "CO",
        "country": "US",
        "lat": 39.7392,
        "lon": -104.9903,
        "timezone": "America/Denver",
        "current": {
            "temp_f": 58,
            "temp_c": 14,
            "feels_like_f": 55,
            "feels_like_c": 13,
            "humidity": 35,
            "wind_mph": 15,
            "wind_direction": "N",
            "condition": "Sunny",
            "uv_index": 8,
            "visibility_miles": 15,
            "pressure_mb": 1025,
        },
        "forecast": [
            {"day": 0, "high_f": 62, "low_f": 40, "condition": "Sunny", "precip_chance": 0, "humidity": 30},
            {"day": 1, "high_f": 65, "low_f": 42, "condition": "Sunny", "precip_chance": 5, "humidity": 28},
            {"day": 2, "high_f": 55, "low_f": 35, "condition": "Snow Showers", "precip_chance": 70, "humidity": 60},
            {"day": 3, "high_f": 48, "low_f": 30, "condition": "Snow", "precip_chance": 85, "humidity": 72},
            {"day": 4, "high_f": 52, "low_f": 33, "condition": "Partly Cloudy", "precip_chance": 20, "humidity": 50},
            {"day": 5, "high_f": 60, "low_f": 38, "condition": "Sunny", "precip_chance": 5, "humidity": 32},
            {"day": 6, "high_f": 64, "low_f": 41, "condition": "Sunny", "precip_chance": 0, "humidity": 28},
        ],
        "alerts": [
            {
                "type": "Winter Storm Watch",
                "severity": "high",
                "headline": "Winter Storm Watch in effect from Tuesday evening through Thursday morning",
                "description": "Heavy snow expected with accumulations of 8-14 inches above 7,000 feet. Winds gusting to 45 mph may cause blowing and drifting snow.",
                "expires_in_hours": 60,
            }
        ],
    },
}

# City name aliases for flexible lookups
_ALIASES: dict[str, str] = {
    "nyc": "new york",
    "ny": "new york",
    "sf": "san francisco",
    "chi": "chicago",
    "den": "denver",
}


def _resolve_city(city: str) -> dict | None:
    """Look up city data by name, supporting common aliases."""
    key = city.strip().lower()
    key = _ALIASES.get(key, key)
    return _CITIES.get(key)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_current_weather(city: str, units: str = "fahrenheit") -> str:
    """Get current weather conditions for a city.

    Returns real-time weather data including temperature, humidity, wind,
    and sky conditions. Data is simulated for demonstration.

    Args:
        city: City name to look up (e.g. "New York", "San Francisco", "Miami",
              "Chicago", "Denver"). Common abbreviations accepted (NYC, SF, etc.).
        units: Temperature units — "fahrenheit" or "celsius". Defaults to "fahrenheit".

    Returns:
        JSON string with current weather conditions including temperature,
        feels-like, humidity, wind speed/direction, condition, UV index,
        visibility, and barometric pressure.
    """
    data = _resolve_city(city)
    if data is None:
        available = [c["city"] for c in _CITIES.values()]
        return json.dumps({
            "error": f"City '{city}' not found",
            "available_cities": available,
            "hint": "Try one of the available cities listed above.",
        }, indent=2)

    if units not in ("fahrenheit", "celsius"):
        return json.dumps({"error": f"Invalid units '{units}'", "valid_values": ["fahrenheit", "celsius"]})

    current = data["current"]
    use_f = units == "fahrenheit"

    return json.dumps({
        "city": data["city"],
        "state": data["state"],
        "country": data["country"],
        "coordinates": {"lat": data["lat"], "lon": data["lon"]},
        "timezone": data["timezone"],
        "current": {
            "temperature": current["temp_f"] if use_f else current["temp_c"],
            "feels_like": current["feels_like_f"] if use_f else current["feels_like_c"],
            "units": "°F" if use_f else "°C",
            "humidity_pct": current["humidity"],
            "wind_speed_mph": current["wind_mph"],
            "wind_direction": current["wind_direction"],
            "condition": current["condition"],
            "uv_index": current["uv_index"],
            "visibility_miles": current["visibility_miles"],
            "pressure_mb": current["pressure_mb"],
        },
        "observed_at": _now_iso(),
        "note": "Simulated data for demonstration purposes.",
    }, indent=2)


@mcp.tool()
def get_forecast(city: str, days: int = 5, units: str = "fahrenheit") -> str:
    """Get a multi-day weather forecast for a city.

    Returns daily high/low temperatures, conditions, precipitation chance,
    and humidity for the requested number of days ahead.

    Args:
        city: City name to look up (e.g. "New York", "San Francisco", "Miami",
              "Chicago", "Denver"). Common abbreviations accepted (NYC, SF, etc.).
        days: Number of forecast days (1–7). Defaults to 5.
        units: Temperature units — "fahrenheit" or "celsius". Defaults to "fahrenheit".

    Returns:
        JSON string with daily forecast entries including date, high/low
        temperatures, condition, precipitation chance, and humidity.
    """
    data = _resolve_city(city)
    if data is None:
        available = [c["city"] for c in _CITIES.values()]
        return json.dumps({
            "error": f"City '{city}' not found",
            "available_cities": available,
        }, indent=2)

    if units not in ("fahrenheit", "celsius"):
        return json.dumps({"error": f"Invalid units '{units}'", "valid_values": ["fahrenheit", "celsius"]})

    days = max(1, min(days, 7))
    use_f = units == "fahrenheit"
    today = datetime.now(timezone.utc).date()

    forecast_entries = []
    for entry in data["forecast"][:days]:
        forecast_date = today + timedelta(days=entry["day"])

        if use_f:
            high, low = entry["high_f"], entry["low_f"]
        else:
            high = round((entry["high_f"] - 32) * 5 / 9)
            low = round((entry["low_f"] - 32) * 5 / 9)

        forecast_entries.append({
            "date": forecast_date.isoformat(),
            "day_of_week": forecast_date.strftime("%A"),
            "high": high,
            "low": low,
            "units": "°F" if use_f else "°C",
            "condition": entry["condition"],
            "precipitation_chance_pct": entry["precip_chance"],
            "humidity_pct": entry["humidity"],
        })

    return json.dumps({
        "city": data["city"],
        "state": data["state"],
        "country": data["country"],
        "forecast_days": days,
        "forecast": forecast_entries,
        "generated_at": _now_iso(),
        "note": "Simulated forecast data for demonstration purposes.",
    }, indent=2)


@mcp.tool()
def get_weather_alerts(city: str) -> str:
    """Get active severe weather alerts for a city.

    Returns any active weather advisories, watches, or warnings including
    severity level, description, and expiration time.

    Args:
        city: City name to look up (e.g. "New York", "San Francisco", "Miami",
              "Chicago", "Denver"). Common abbreviations accepted (NYC, SF, etc.).

    Returns:
        JSON string with a list of active alerts. Each alert includes type,
        severity (low/moderate/high/extreme), headline, description, and
        expiration timestamp. Returns an empty list if no alerts are active.
    """
    data = _resolve_city(city)
    if data is None:
        available = [c["city"] for c in _CITIES.values()]
        return json.dumps({
            "error": f"City '{city}' not found",
            "available_cities": available,
        }, indent=2)

    now = datetime.now(timezone.utc)
    alerts = []
    for alert in data["alerts"]:
        expires = now + timedelta(hours=alert["expires_in_hours"])
        alerts.append({
            "type": alert["type"],
            "severity": alert["severity"],
            "headline": alert["headline"],
            "description": alert["description"],
            "issued_at": (now - timedelta(hours=6)).isoformat(),
            "expires_at": expires.isoformat(),
        })

    return json.dumps({
        "city": data["city"],
        "state": data["state"],
        "country": data["country"],
        "active_alerts": len(alerts),
        "alerts": alerts,
        "checked_at": _now_iso(),
        "note": "Simulated alert data for demonstration purposes.",
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
