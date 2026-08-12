# Weather Intelligence — README

Unstructured weather data → Lakebase vector search → REST API, built for
DataExpert.io assignment 4938, following the pattern established in
`databricks-lakebase-app-day-2`.

## Data source

**National Weather Service API (api.weather.gov)** was used, for the reasons the
assignment recommended: it's free, requires no API key, has generous rate
limits, and returns rich unstructured narrative text (`description` +
`instruction` on alerts, `detailedForecast` on forecast periods) that's
well suited to embedding. No other source was mixed in.

Coverage is limited to the continental US, Alaska, Hawaii, and US
territories. Five representative cities were tracked: San Francisco CA,
New York NY, Salt Lake City UT, Chicago IL, and Miami FL — chosen for
varied climates/alert profiles (coastal, continental, high-desert,
hurricane-prone) to give the embeddings something to meaningfully
differentiate.

## Schema decisions

Everything lives in its own Postgres schema, `weather`, kept fully
separate from the reference app's `ticker_news_*` / `massive_*` tables.

**`weather.weather_documents`**
| column | type | notes |
|---|---|---|
| id | TEXT PK | stable dedup key — NWS's own alert id, or a hash of location+period+startTime for forecasts |
| location | TEXT | e.g. "Chicago, Illinois" |
| source_type | TEXT | `alert` or `forecast` — lets Part 3's retrieval filter by source |
| headline | TEXT | e.g. "Flash Flood Warning" or "Tonight" |
| narrative_text | TEXT | the actual text that gets embedded |
| issued_at / effective_at | TIMESTAMPTZ | |
| payload | JSONB | raw NWS response, kept for provenance |
| synced_at | TIMESTAMPTZ | |

**`weather.weather_embeddings`**
| column | type | notes |
|---|---|---|
| id | BIGSERIAL PK | |
| document_id | TEXT FK → weather_documents.id | ON DELETE CASCADE |
| chunk_index | INT | 0-based; almost always 0 since NWS text is short |
| chunk_text | TEXT | |
| embedding | vector(384) | pgvector column |
| model_name | TEXT | |
| created_at | TIMESTAMPTZ | |

An `hnsw (embedding vector_cosine_ops)` index is created on
`weather_embeddings.embedding` for retrieval performance.

**Chunking**: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` characters, sliding
window — matches the assignment's suggested defaults. In practice, NWS
narrative text is short enough that nearly every document produces a
single chunk (`chunk_index=0`); chunking only kicks in for the rare
long combined alert description+instruction.

**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`, 384
dimensions — same model as the reference app's news pipeline, so both
stay compatible with the same `<=>` cosine-distance convention.

## Pipeline: sync → embed → search

All three steps are exposed as endpoints on the same Flask app (rather
than a separate scheduled notebook, though `notebooks/ingest_weather_embeddings.py`
is also included as a thin wrapper around the same embedding logic for
the literal deliverable requirement / optional scheduling later).

**1. Sync** — harvest alerts + forecasts from NWS into `weather_documents`:
```
POST /weather/sync
{"locations": ["Chicago, Illinois", "Miami, Florida"], "limit": 50}
→ {"synced": 15, "locations": [...]}
```
Upserts on `id`, so re-running doesn't create duplicates.

**2. Embed** — chunk + vectorize any documents without embeddings yet:
```
POST /weather/embed
{"limit": 500}
→ {"documents_processed": 15, "chunks_embedded": 15}
```
Safe to call repeatedly — only processes documents that aren't already
embedded, via a `LEFT JOIN ... WHERE e.id IS NULL` query.

**3. Search** — semantic search over ingested documents:
```
POST /weather/search
{"query": "risk of flooding near rivers", "top_k": 5, "source_type": null}
→ {"query": "...", "results": [{"location": ..., "headline": ..., "chunk_text": ..., "similarity": 0.71}, ...]}
```
`source_type` is optional — `"alert"` or `"forecast"` restricts retrieval
to just that source (stretch goal: combining two sources with a filter);
omitted searches both together. `top_k` is clamped to 1–20.

A browser UI is also served at `GET /` (Flask + `templates/index.html`,
plain fetch() calls to the same three endpoints above) for interactively
triggering sync/embed and running searches without curl/Postman.

**Auth note**: this app runs on Databricks Apps with a Lakebase
"Database" resource attached, which authenticates via a service-principal
OAuth token (`w.postgres.generate_database_credential()`) rather than a
static connection string — `lakebase.py` generates a fresh token on every
connection since tokens expire hourly.

## Known limitations / what I'd improve with more time

- Only 5 hardcoded US locations are supported today; a geocoding step
  (e.g. Nominatim) was drafted to support arbitrary US place names but
  not yet fully wired into the sync endpoint.
- No scheduled re-sync — alerts/forecasts only update when `/weather/sync`
  is called manually or from the UI, not on a recurring job.
- The stretch-goal HNSW-vs-no-index latency benchmark wasn't run; the
  index is created but its performance impact wasn't formally measured
  given the small (tens of rows) dataset size in testing.
- No RAG summary variant (`GET /weather/search?query=...` with an
  LLM-generated summary) was built.
