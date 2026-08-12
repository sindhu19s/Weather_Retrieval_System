"""
weather_embeddings.py

Embedding pipeline for weather documents: chunks long narrative_text,
embeds each chunk with sentence-transformers, and writes vectors into
weather.weather_embeddings via psycopg2 (NOT spark.write.jdbc -- per the
assignment, Spark JDBC writes are not supported against this Lakebase
instance).

Model is loaded once at module level (not per-call) since loading it is
slow (~seconds) and it's safe to reuse across calls.
"""
from datetime import datetime, timezone

from psycopg2.extras import execute_values

from lakebase import get_connection
from lakebase_weather import WEATHER_EMBEDDINGS_TABLE, WEATHER_DOCUMENTS_TABLE, fetch_unembedded_documents

# Matches the reference app's existing news embedding pipeline so both
# stay compatible/queryable with the same distance operator conventions.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Lazily loaded on first use -- importing sentence-transformers and
# downloading the model is slow, so we don't want that to happen at
# import time (e.g. when app.py boots and imports this module for the
# /weather/embed route registration).
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks using a simple sliding window over
    characters. Most NWS narrative_text is well under chunk_size, so this
    usually returns a single chunk -- it mainly matters for combined
    alert description+instruction text that runs long.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of strings in one batch call (more efficient than
    embedding one at a time). Returns a list of 384-dim float vectors.
    """
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def _write_embeddings(rows: list[tuple]) -> int:
    """
    rows: list of (document_id, chunk_index, chunk_text, embedding, model_name, created_at)
    Writes via execute_values with the embedding cast to ::vector, per the
    assignment's guidance -- psycopg2 + pgvector's adapter handles the cast
    from a Python list.
    """
    if not rows:
        return 0

    query = f"""
        INSERT INTO {WEATHER_EMBEDDINGS_TABLE}
            (document_id, chunk_index, chunk_text, embedding, model_name, created_at)
        VALUES %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur, query, rows,
                template="(%s, %s, %s, %s::vector, %s, %s)"
            )
        conn.commit()
    return len(rows)


def embed_query(query: str) -> list[float]:
    """
    Embeds a single query string using the same model/loading pattern as
    embed_texts(), for use by the /weather/search endpoint.
    """
    vectors = embed_texts([query])
    return vectors[0] if vectors else []


def search_weather_embeddings(query: str, top_k: int = 5, source_type: str | None = None) -> list[dict]:
    """
    Embeds the query and runs a cosine-similarity search against
    weather_embeddings, joined back to weather_documents for display
    fields. Returns the top_k matches as a list of dicts.

    source_type: optional filter, either "alert" or "forecast" -- when
    set, only documents of that type are searched. None/omitted searches
    across both sources combined (the stretch-goal behavior).
    """
    query_vector = embed_query(query)

    where_clause = ""
    params = [query_vector]
    if source_type in ("alert", "forecast"):
        where_clause = "WHERE d.source_type = %s"
        params.append(source_type)
    params.append(query_vector)
    params.append(top_k)

    sql = f"""
        SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text,
               e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {WEATHER_EMBEDDINGS_TABLE} e
        JOIN {WEATHER_DOCUMENTS_TABLE} d ON d.id = e.document_id
        {where_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def run_embedding_ingestion(batch_limit: int = 500) -> dict:
    """
    Main entry point: fetches unembedded documents from weather_documents,
    chunks + embeds their narrative_text, and writes the resulting vectors
    into weather_embeddings.

    Returns a summary dict: {"documents_processed": N, "chunks_embedded": M}
    """
    documents = fetch_unembedded_documents(limit=batch_limit)
    if not documents:
        return {"documents_processed": 0, "chunks_embedded": 0}

    rows = []
    for doc in documents:
        chunks = chunk_text(doc["narrative_text"])
        if not chunks:
            continue
        vectors = embed_texts(chunks)
        now = datetime.now(timezone.utc)
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            rows.append((
                doc["id"],
                idx,
                chunk,
                vector,
                EMBEDDING_MODEL_NAME,
                now,
            ))

    written = _write_embeddings(rows)

    return {
        "documents_processed": len(documents),
        "chunks_embedded": written,
    }