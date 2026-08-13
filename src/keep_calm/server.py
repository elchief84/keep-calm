"""Keep Calm REST API server.

Privacy-first by design: binds to 127.0.0.1 by default, logs no message
content, and keeps the model loaded in memory for low-latency analysis.

Usage:
    keep-calm-serve                 # -> http://127.0.0.1:8000
    keep-calm-serve --port 8080
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field

from keep_calm.schemas import AnalysisResult

_analyzer: Any = None


def get_analyzer() -> Any:
    """Load the analyzer once (lazy singleton)."""
    global _analyzer
    if _analyzer is None:
        from keep_calm.analyzer import KeepCalmAnalyzer

        _analyzer = KeepCalmAnalyzer()
    return _analyzer


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@asynccontextmanager
async def lifespan(app):  # noqa: ARG001
    # Preload the model at startup so /health is truthful and the first
    # request is not penalized by model loading.
    get_analyzer()
    yield


def create_app():
    """Build the FastAPI application (lazy imports keep the CLI dependency-free)."""
    from fastapi import FastAPI

    app = FastAPI(
        title="Keep Calm",
        description="Privacy-first, pre-send communication analysis. Runs entirely locally.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze", response_model=AnalysisResult)
    def analyze(request: AnalyzeRequest) -> AnalysisResult:
        return get_analyzer().analyze(request.text)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep Calm REST API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
