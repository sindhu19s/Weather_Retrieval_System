"""
weather_client.py

Client for the National Weather Service (NWS) API (https://api.weather.gov).
Mirrors the structure of massive_client.py: a small client class that
handles requests, resolves locations, and normalizes raw NWS responses
into flat "document" dicts ready to upsert into weather_documents.

No API key required. NWS asks that requests include a descriptive
User-Agent with contact info (not an auth requirement, just good citizenship
and how they throttle abusive clients).
"""
import hashlib
import time
from datetime import datetime, timezone

import requests

NWS_BASE_URL = "https://api.weather.gov"

# NWS asks for a User-Agent identifying the app + contact info.
# Replace the email with a real one before deploying.
USER_AGENT = "(weather-homework-app, sindhu.example@example.com)"

# Hardcoded lat/lon for the five tracked cities. NWS only covers the US,
# so all locations here must resolve to a US grid point.
KNOWN_LOCATIONS = {
    "San Francisco, California": (37.7749, -122.4194),
    "New York, New York": (40.7128, -74.0060),
    "Salt Lake City, Utah": (40.7608, -111.8910),
    "Chicago, Illinois": (41.8781, -87.6298),
    "Miami, Florida": (25.7617, -80.1918),
}

# Map full state names to 2-letter codes for the /alerts/active?area= param.
STATE_ABBREVIATIONS = {
    "California": "CA",
    "New York": "NY",
    "Utah": "UT",
    "Illinois": "IL",
    "Florida": "FL",
}


class WeatherClientError(Exception):
    """Raised when the NWS API returns an unexpected response."""


class WeatherClient:
    def __init__(self, base_url: str = NWS_BASE_URL, user_agent: str = USER_AGENT,
                 request_delay_seconds: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/geo+json",
        })
        # NWS doesn't publish a hard rate limit like Massive's free tier,
        # but a small delay between calls is polite and avoids 429s when
        # syncing several locations back to back.
        self.request_delay_seconds = request_delay_seconds

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=15)
        time.sleep(self.request_delay_seconds)
        if resp.status_code != 200:
            raise WeatherClientError(
                f"NWS API error {resp.status_code} for {url}: {resp.text[:300]}"
            )
        return resp.json()

    # ---- Step 1: resolve lat/lon -> grid point -----------------------------

    def get_grid_point(self, lat: float, lon: float) -> dict:
        """
        Resolves a lat/lon to NWS grid metadata (office id + x,y + forecast
        URLs). This is always the first call for any location, per the NWS
        API design (forecasts are issued per 2.5km grid square, not per
        arbitrary coordinate).
        """
        data = self._get(f"/points/{lat},{lon}")
        props = data.get("properties", {})
        return {
            "grid_id": props.get("gridId"),
            "grid_x": props.get("gridX"),
            "grid_y": props.get("gridY"),
            "forecast_url": props.get("forecast"),
            "forecast_hourly_url": props.get("forecastHourly"),
            "state": props.get("relativeLocation", {})
                          .get("properties", {})
                          .get("state"),
        }

    # ---- Step 2a: active alerts ---------------------------------------------

    def get_active_alerts(self, state_code: str) -> list[dict]:
        """
        Fetches active alerts for a US state (2-letter code, e.g. 'CA').
        Returns the raw list of alert features.
        """
        data = self._get("/alerts/active", params={"area": state_code})
        return data.get("features", [])

    # ---- Step 2b: forecast narrative -----------------------------------------

    def get_forecast(self, forecast_url: str) -> list[dict]:
        """
        Fetches the multi-day narrative forecast for a grid point.
        Returns the list of forecast periods, each with a detailedForecast
        free-text string.
        """
        data = self._get(forecast_url)
        return data.get("properties", {}).get("periods", [])

    # ---- Normalization --------------------------------------------------------

    @staticmethod
    def _stable_id(*parts: str) -> str:
        """Builds a stable dedup key from arbitrary string parts."""
        joined = "|".join(str(p) for p in parts if p is not None)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]

    def normalize_alert(self, location_name: str, alert: dict) -> dict:
        props = alert.get("properties", {})
        description = (props.get("description") or "").strip()
        instruction = (props.get("instruction") or "").strip()
        narrative_text = "\n\n".join(t for t in (description, instruction) if t)

        alert_id = props.get("id") or self._stable_id(
            location_name, props.get("event"), props.get("onset")
        )

        return {
            "id": f"alert:{alert_id}",
            "location": location_name,
            "source_type": "alert",
            "headline": props.get("event"),
            "narrative_text": narrative_text,
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective") or props.get("onset"),
            "payload": alert,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    def normalize_forecast_period(self, location_name: str, period: dict) -> dict:
        narrative_text = (period.get("detailedForecast") or "").strip()
        issued_at = period.get("startTime")

        doc_id = self._stable_id(
            location_name, "forecast", period.get("number"), issued_at
        )

        return {
            "id": f"forecast:{doc_id}",
            "location": location_name,
            "source_type": "forecast",
            "headline": period.get("name"),  # e.g. "Tonight", "Monday"
            "narrative_text": narrative_text,
            "issued_at": issued_at,
            "effective_at": period.get("startTime"),
            "payload": period,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---- High-level entry point ------------------------------------------------

    def fetch_documents_for_location(self, location_name: str, limit: int = 50) -> list[dict]:
        """
        Given a known location name, fetches active alerts + forecast
        periods and returns a flat list of normalized document dicts,
        ready to upsert into weather_documents. `limit` caps the total
        number of documents returned per location (alerts first, then
        forecast periods, to prioritize the more time-sensitive content).
        """
        if location_name not in KNOWN_LOCATIONS:
            raise WeatherClientError(
                f"Unknown location '{location_name}'. "
                f"Add it to KNOWN_LOCATIONS and STATE_ABBREVIATIONS first."
            )

        lat, lon = KNOWN_LOCATIONS[location_name]
        grid = self.get_grid_point(lat, lon)

        state_full = location_name.split(",")[-1].strip()
        state_code = STATE_ABBREVIATIONS.get(state_full) or grid.get("state")
        if not state_code:
            raise WeatherClientError(
                f"Could not resolve state code for '{location_name}'."
            )

        documents: list[dict] = []

        # Alerts
        for alert in self.get_active_alerts(state_code):
            documents.append(self.normalize_alert(location_name, alert))
            if len(documents) >= limit:
                return documents

        # Forecast periods
        if grid.get("forecast_url"):
            for period in self.get_forecast(grid["forecast_url"]):
                documents.append(self.normalize_forecast_period(location_name, period))
                if len(documents) >= limit:
                    return documents

        return documents

    def fetch_documents(self, locations: list[str], limit: int = 50) -> list[dict]:
        """
        Fetches documents for multiple locations. `limit` is applied
        per-location (matching the /weather/sync request contract).
        """
        all_documents: list[dict] = []
        for location_name in locations:
            all_documents.extend(
                self.fetch_documents_for_location(location_name, limit=limit)
            )
        return all_documents
