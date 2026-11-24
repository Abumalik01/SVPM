# Deploying SVPM on Railway

SVPM can run as one Railway web service. FastAPI serves the browser interface at `/` and the JSON API under `/predict` and `/admin`. The trained model bundles in `models/` are included in the repository and loaded by the service.

## Set up the project

1. Push this repository to a GitHub repository connected to Railway.
2. In Railway, create a project from that repository. Railway detects `railway.json` for the build, start command, and `/health` health check.
3. Add a Railway Volume to the web service mounted at `/data`.
4. Set these service variables:

   - `SVPM_LOG_DB=/data/predictions.db` — keeps the SQLite prediction log on the volume.
   - `SVPM_ADMIN_TOKEN` — a long, randomly generated secret for the admin endpoints.

5. Generate a public domain for the web service. Open it and confirm that the active model appears in the page header.

The volume is needed because a container's normal filesystem is not a durable location for the prediction log. Railway mounts the volume at runtime; the app creates the SQLite database on first use. The model files stay in the deployed repository, so model updates are made by committing the reviewed model bundle and redeploying.

## Local run

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Streamlit remains available for local use with `python -m streamlit run ui/app.py`.

## Admin and data handling

The Administration page uses `SVPM_ADMIN_TOKEN` for model-version and recent-log routes. Uploaded source files are parsed in memory; only extracted metrics and prediction results are stored. Retraining and model promotion remain command-line tasks and are not exposed to the public web interface.

For a multi-instance deployment, move prediction logging from SQLite to a managed PostgreSQL service. SQLite on an attached volume is intended for a single web-service instance.
