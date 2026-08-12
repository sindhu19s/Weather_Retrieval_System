#--COMMAND---
%pip install --upgrade databricks-sdk

#--COMMAND---
dbutils.library.restartPython()

#--COMMAND---
import importlib.metadata as md
from databricks.sdk import WorkspaceClient
print(f"databricks-sdk version: {md.version('databricks-sdk')}")
w = WorkspaceClient()
print(f"Has 'postgres' attribute: {hasattr(w, 'postgres')}")

#--COMMAND---
import sys
import os

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
	
os.environ.setdefault("PGHOST", "ep-rapid-surf-d8vdudu8.database.us-east-2.cloud.databricks.com")
os.environ.setdefault("PGPORT", "5432")
os.environ.setdefault("PGDATABASE", "databricks_postgres")
os.environ.setdefault("PGSSLMODE", "require")
os.environ.setdefault("LAKEBASE_ENDPOINT", "projects/project2-weather/branches/production/endpoints/primary")

if "PGUSER" not in os.environ:
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    os.environ["PGUSER"] = current_user
#--COMMAND---	
dbutils.widgets.text("batch_limit", "500", "Max documents to embed per run")

#--COMMAND---
import sys 
sys.path.insert(0, "/Workspace/Users/sindhunsuresh19@gmail.com/Weather_Retrieval_Service")

#--COMMAND---
import os 
print(os.listdir("/Workspace/Users/sindhunsuresh19@gmail.com/Weather_Retrieval_Service"))

#--COMMAND---
import importlib.metadata as md
import sys
import os

# Check if we have the upgraded SDK (>= 0.118.0)
current_version = md.version('databricks-sdk')
if tuple(map(int, current_version.split('.')[:2])) < (0, 118):
    # Need to upgrade and restart
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "databricks-sdk>=0.118.0"])
    print(f"Upgraded databricks-sdk from {current_version}. Restarting Python...")
    dbutils.library.restartPython()
else:
    # SDK is current, clear any cached modules first
    for mod in ['lakebase', 'lakebase_weather', 'weather_embeddings']:
        if mod in sys.modules:
            del sys.modules[mod]
    
    # Set up environment BEFORE imports
    os.environ.setdefault("PGHOST", "ep-rapid-surf-d8vdudu8.database.us-east-2.cloud.databricks.com")
    os.environ.setdefault("PGPORT", "5432")
    os.environ.setdefault("PGDATABASE", "databricks_postgres")
    os.environ.setdefault("PGSSLMODE", "require")
    os.environ.setdefault("LAKEBASE_ENDPOINT", "projects/project2-weather/branches/production/endpoints/primary")
    
    if "PGUSER" not in os.environ:
        current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
        os.environ["PGUSER"] = current_user
    
    # Now set up path and import
    PROJECT_ROOT = "/Workspace/Users/sindhunsuresh19@gmail.com/Weather_Retrieval_Service"
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    from weather_embeddings import run_embedding_ingestion
    result = run_embedding_ingestion(batch_limit=500)
    print(result)