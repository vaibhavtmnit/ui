# main.py
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.staticfiles import StaticFiles
import os
import dashboard_ui

app = FastAPI(title="Dashboard (File-based, Multi-user)")

DATA_ROOT = os.path.abspath("./data")
os.makedirs(DATA_ROOT, exist_ok=True)
app.mount("/data", StaticFiles(directory=DATA_ROOT), name="data")

app.mount("/dash", WSGIMiddleware(dashboard_ui.server))

@app.get("/health")
def health():
    return {"status": "ok"}
