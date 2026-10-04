"""Entry point for Google Cloud Functions (gen2) deployment.

Cloud Functions gen2 expects a module-level ASGI/WSGI app or a function.
With functions-framework, pointing --entry-point to the FastAPI app object
lets it serve the app directly — no Docker needed.
"""

from src.api.app import app  # noqa: F401
