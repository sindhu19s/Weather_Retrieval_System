"""
lakebase.py

Connection helper for Lakebase (Databricks-managed Postgres), using the
"Lakebase database" Databricks Apps resource. This resource type does NOT
give you a single connection-string secret. Instead, when you attach it to
the app, Databricks injects these environment variables automatically:

    PGHOST      - Postgres server hostname
    PGPORT      - Postgres server port
    PGDATABASE  - database name
    PGUSER      - service principal's client ID (acts as the Postgres role)
    PGSSLMODE   - ssl mode (should be "require")
    PGAPPNAME   - app name

There is no PGPASSWORD -- auth is via a short-lived OAuth token that the
app must fetch itself using the Databricks SDK, and that token expires
after ~60 minutes. So we generate a fresh token on every new connection
rather than caching one.

Everything else in the app (lakebase_weather.py, app.py) imports
get_connection() from here rather than opening its own connections.
"""
import os
from contextlib import contextmanager

import psycopg2
from databricks.sdk import WorkspaceClient

PGHOST = os.environ.get("PGHOST")
PGPORT = os.environ.get("PGPORT", "5432")
PGDATABASE = os.environ.get("PGDATABASE", "databricks_postgres")
PGUSER = os.environ.get("PGUSER")
PGSSLMODE = os.environ.get("PGSSLMODE", "require")

# The endpoint identifier the SDK needs to mint a database credential.
# For Lakebase Autoscaling this looks like:
#   projects/<project>/branches/<branch>/endpoints/<endpoint>
# Set this once you know your project/branch/endpoint names (visible in
# the Lakebase UI's "Connect" dialog, or in the app's Resources tab).
LAKEBASE_ENDPOINT = os.environ.get("LAKEBASE_ENDPOINT")

_workspace_client = WorkspaceClient()


def _get_password() -> str:
    """
    Returns a fresh OAuth token to use as the Postgres password.

    Locally (developer running `databricks auth login`), PGUSER should be
    your own email and the SDK will authenticate as you.
    When deployed as a Databricks App, PGUSER is the service principal's
    client ID and the SDK authenticates as that service principal
    automatically -- no extra config needed there.
    """
    if not LAKEBASE_ENDPOINT:
        raise RuntimeError(
            "LAKEBASE_ENDPOINT is not set. Set it to "
            "'projects/<project>/branches/<branch>/endpoints/<endpoint>' "
            "(find these names in the Lakebase 'Connect' dialog in the "
            "Databricks UI)."
        )
    credential = _workspace_client.postgres.generate_database_credential(
        endpoint=LAKEBASE_ENDPOINT
    )
    return credential.token


@contextmanager
def get_connection():
    """
    Yields a live psycopg2 connection to Lakebase, closing it automatically
    when the `with` block exits (success or error). Generates a fresh
    OAuth token on every call, since tokens expire after ~60 minutes and
    connections are short-lived per request in this app.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            conn.commit()
    """
    if not PGHOST or not PGUSER:
        raise RuntimeError(
            "PGHOST/PGUSER are not set. Confirm the app has a Lakebase "
            "'Database' resource attached (App resources > + Add resource "
            "> Database) -- Databricks injects PGHOST/PGPORT/PGDATABASE/"
            "PGUSER/PGSSLMODE automatically once that's attached."
        )

    password = _get_password()

    conn = psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=password,
        sslmode=PGSSLMODE,
    )
    try:
        yield conn
    finally:
        conn.close()