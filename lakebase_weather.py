"""
lakebase_weather.py

Lakebase (Postgres) helpers for the weather feature. Mirrors lakebase.py's
connection pattern (psycopg2 + RealDictCursor via a get_connection()
context manager) but keeps everything weather-specific isolated in its own
schema so it never touches the existing massive/ticker tables.

Import get_connection from the existing lakebase.py rather than duplicating
connection logic -- this module only owns the weather-specific DDL and
upsert queries.
"""
import json

from psycopg2.extras import RealDictCursor, execute_values

# Reuse the existing connection helper -- same LAKEBASE_URL, same pooling.
# lakebase.py's get_connection() is expected to be a @contextmanager that
# yields a psycopg2 connection (see the reference app's lakebase.py).
from lakebase import get_connection

WEATHER_SCHEMA = "weather"
WEATHER_DOCUMENTS_TABLE = f"{WEATHER_SCHEMA}.weather_documents"
WEATHER_EMBEDDINGS_TABLE = f"{WEATHER_SCHEMA}.weather_embeddings"


def ensure_weather_schema():
    """
    Idempotently creates the weather schema + tables + index. Safe to call
    on every app startup, same as ensure_table()-style helpers elsewhere
    in the app.
    """
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {WEATHER_SCHEMA};

    CREATE TABLE IF NOT EXISTS {WEATHER_DOCUMENTS_TABLE} (
        id              TEXT PRIMARY KEY,
        location        TEXT NOT NULL,
        source_type     TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
        headline        TEXT,
        narrative_text  TEXT NOT NULL,
        issued_at       TIMESTAMPTZ,
        effective_at    TIMESTAMPTZ,
        payload         JSONB,
        synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS {WEATHER_EMBEDDINGS_TABLE} (
        id              BIGSERIAL PRIMARY KEY,
        document_id     TEXT NOT NULL REFERENCES {WEATHER_DOCUMENTS_TABLE}(id) ON DELETE CASCADE,
        chunk_index     INT NOT NULL,
        chunk_text      TEXT NOT NULL,
        embedding       vector(384) NOT NULL,
        model_name      TEXT NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS weather_embeddings_hnsw_idx
        ON {WEATHER_EMBEDDINGS_TABLE}
        USING hnsw (embedding vector_cosine_ops);
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def upsert_weather_documents(documents: list[dict]) -> int:
    """
    Upserts a list of normalized document dicts (as produced by
    weather_client.py) into weather_documents. Re-syncing the same
    alert/forecast period updates it in place rather than duplicating,
    keyed on `id`.

    Returns the number of rows upserted.
    """
    if not documents:
        return 0

    rows = [
        (
            doc["id"],
            doc["location"],
            doc["source_type"],
            doc.get("headline"),
            doc["narrative_text"],
            doc.get("issued_at"),
            doc.get("effective_at"),
            json.dumps(doc.get("payload")) if doc.get("payload") is not None else None,
            doc.get("synced_at"),
        )
        for doc in documents
    ]

    query = f"""
        INSERT INTO {WEATHER_DOCUMENTS_TABLE}
            (id, location, source_type, headline, narrative_text,
             issued_at, effective_at, payload, synced_at)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            location       = EXCLUDED.location,
            source_type    = EXCLUDED.source_type,
            headline       = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at      = EXCLUDED.issued_at,
            effective_at   = EXCLUDED.effective_at,
            payload        = EXCLUDED.payload,
            synced_at      = EXCLUDED.synced_at;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows, template="(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)")
        conn.commit()

    return len(rows)


def fetch_unembedded_documents(limit: int = 500) -> list[dict]:
    """
    Returns documents from weather_documents that don't yet have any rows
    in weather_embeddings. Used by the Part 2 embedding ingestion script.
    """
    query = f"""
        SELECT d.*
        FROM {WEATHER_DOCUMENTS_TABLE} d
        LEFT JOIN {WEATHER_EMBEDDINGS_TABLE} e ON e.document_id = d.id
        WHERE e.id IS NULL
        ORDER BY d.synced_at ASC
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (limit,))
            return cur.fetchall()
