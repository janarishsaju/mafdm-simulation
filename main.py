"""
main.py
=======
Entry point for the MA-FDE-LLM Simulation Platform backend.

Usage:
    python main.py

Environment variables required:
    OPENAI_API_KEY   — OpenAI API key for LLM calls

Optional:
    PORT             — port to listen on (default: 8000)
    HOST             — host to bind to  (default: 0.0.0.0)
"""

import os
import uvicorn

from backend.api import app  # noqa: F401 — imported for uvicorn
from fastapi.staticfiles import StaticFiles

# Serve the built React frontend if the dist/ folder exists.
# In production (Render) the dist/ is committed to git alongside the backend.
# In local dev the Vite dev server runs separately on port 5173.
_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    print("=" * 60)
    print("MA-FDE-LLM Simulation Platform")
    print(f"Backend  : http://{host}:{port}")
    print(f"API docs : http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run(
        "backend.api:app",
        host    = host,
        port    = port,
        reload  = False,
        workers = 1,
    )
