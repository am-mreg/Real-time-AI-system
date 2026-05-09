"""
src/main.py

Entrypoint for running the FastAPI app with uvicorn.

This file assumes app.py (with create_app) lives in the same package (src).
When using `uvicorn` with reload during development, the recommended
app string is "src.main:app" when launching from the repository root.
If you run this script directly from the src folder (python main.py),
the fallback "main:app" is attempted.
"""
import os
import uvicorn

# Import create_app from the local package module (app.py is next to this file)
from app import create_app

app = create_app()


def _run_uvicorn():
    """
    Start uvicorn with a reload-friendly app string.
    Try the most common module paths so reload works both when running:
      - from repo root:  uvicorn src.main:app --reload
      - from src folder: uvicorn main:app --reload
    The function attempts the best candidate and falls back gracefully.
    """
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    # Prefer explicit module path for reload to work when running from repo root
    candidates = [
        os.environ.get("UVICORN_APP", "src.main:app"),
        "main:app",
        "src.main:app",
    ]

    for candidate in candidates:
        try:
            uvicorn.run(candidate, host=host, port=port, reload=True)
            return
        except Exception:
            # Try next candidate
            continue

    # As a last resort, run using the app instance (reload won't be available)
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    _run_uvicorn()
