"""
app.py

Weather Intelligence Flask app.

Endpoints (Part 1 of the assignment -- more added in later parts):
  GET  /healthz         - health check
  POST /weather/sync    - fetch alerts + forecasts from NWS, upsert into Lakebase

Run locally with:
    python app.py
"""
import os

from flask import Flask, jsonify, request, render_template

from weather_client import WeatherClient, WeatherClientError, KNOWN_LOCATIONS
from lakebase_weather import ensure_weather_schema, upsert_weather_documents
from weather_embeddings import run_embedding_ingestion, search_weather_embeddings

app = Flask(__name__)

# Single shared client instance -- reused across requests rather than
# reconnecting/reinstantiating per call.
weather_client = WeatherClient()

# Create the weather schema + tables on startup if they don't already
# exist. Safe to call every time the app boots (idempotent DDL).
ensure_weather_schema()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """
    POST /weather/sync
    Body: {"locations": ["Chicago, Illinois", "Miami, Florida"], "limit": 50}

    Fetches active alerts + forecast narrative for each requested location
    from the National Weather Service, normalizes into document records,
    and upserts into weather.weather_documents. Returns a count synced.
    """
    body = request.get_json(silent=True) or {}
    locations = body.get("locations")
    limit = body.get("limit", 50)

    if not locations or not isinstance(locations, list):
        return jsonify({
            "error": "Request body must include a non-empty 'locations' list.",
            "known_locations": list(KNOWN_LOCATIONS.keys()),
        }), 400

    unknown = [loc for loc in locations if loc not in KNOWN_LOCATIONS]
    if unknown:
        return jsonify({
            "error": f"Unknown location(s): {unknown}",
            "known_locations": list(KNOWN_LOCATIONS.keys()),
        }), 400

    try:
        limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        limit = 50

    try:
        documents = weather_client.fetch_documents(locations, limit=limit)
    except WeatherClientError as e:
        return jsonify({"error": str(e)}), 502

    synced_count = upsert_weather_documents(documents)

    return jsonify({
        "synced": synced_count,
        "locations": locations,
    }), 200


@app.route("/weather/embed", methods=["POST"])
def weather_embed():
    """
    POST /weather/embed
    Body: {"limit": 500}   (optional, defaults to 500)

    Finds weather_documents rows that don't have embeddings yet, chunks
    their narrative_text, embeds each chunk with sentence-transformers,
    and writes vectors into weather.weather_embeddings. Safe to call
    repeatedly -- only processes documents that aren't already embedded.
    """
    body = request.get_json(silent=True) or {}
    limit = body.get("limit", 500)

    try:
        limit = max(1, min(int(limit), 2000))
    except (TypeError, ValueError):
        limit = 500

    result = run_embedding_ingestion(batch_limit=limit)
    return jsonify(result), 200


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """
    POST /weather/search
    Body: {"query": "risk of flooding near rivers", "top_k": 5}

    Embeds the query, runs a cosine-similarity search against
    weather_embeddings (via pgvector's <=> operator), and returns the
    top matches. Handles: empty embeddings table, missing/malformed
    query, and top_k bounds (clamped to 1-20).
    """
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    top_k = body.get("top_k", 5)
    source_type = body.get("source_type")  # optional: "alert" or "forecast"

    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Request body must include a non-empty 'query' string."}), 400

    if source_type is not None and source_type not in ("alert", "forecast"):
        return jsonify({"error": "source_type must be 'alert', 'forecast', or omitted."}), 400

    try:
        top_k = max(1, min(int(top_k), 20))
    except (TypeError, ValueError):
        top_k = 5

    try:
        results = search_weather_embeddings(query.strip(), top_k=top_k, source_type=source_type)
    except Exception as e:
        # Covers e.g. an empty weather_embeddings table producing no rows
        # (not an error -- just return empty) vs. a real DB/model failure.
        return jsonify({"error": f"Search failed: {e}"}), 502

    return jsonify({
        "query": query,
        "results": results,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)