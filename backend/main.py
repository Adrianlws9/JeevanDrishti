"""
JEEVAN-DRISHTI FastAPI Application Entrypoint
============================================
Phase 17: FastAPI Backend & Analytical Engine Integration

Modular public-health backend serving as the communication layer between
surveillance data, predictive analytical engines, SQLite database, and frontend.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.api.v1 import api_v1_router
from backend.config import settings
from backend.database.connection import init_db, seed_db
from backend.engines.engine_registry import get_engine_registry

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("jeevan_backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: initialize database and load engines at startup."""
    logger.info("Starting up JEEVAN-DRISHTI FastAPI Application...")
    # 1. Initialize & seed SQLite database
    init_db()
    seed_db()
    # 2. Warm up analytical engine caches
    registry = get_engine_registry()
    registry.load_all_engines()
    logger.info("Startup complete. All predictive engines and database tables ready.")
    yield
    logger.info("Shutting down JEEVAN-DRISHTI backend.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Execution Time & Logging Middleware
@app.middleware("http")
async def add_process_time_and_log(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    logger.info(
        "%s %s - Status: %d - Latency: %.4fs",
        request.method,
        request.url.path,
        response.status_code,
        process_time,
    )
    return response


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception on %s %s: %s", request.method, request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred processing the surveillance intelligence request.",
            "detail": str(exc) if settings.DEBUG else None,
        },
    )


# Frontend (Phase 19): static single-page app served at /app
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


@app.get("/ui", include_in_schema=False)
def ui_redirect():
    return RedirectResponse(url="/app/")


# Root Endpoint
@app.get("/", tags=["Health"])
def root():
    """Health check and API catalog discovery endpoint."""
    return {
        "platform": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "status": "HEALTHY",
        "territory": "Maharashtra State (34 active districts, 2 unsupported documented)",
        "api_v1_docs": "/docs",
        "openapi_spec": "/api/v1/openapi.json",
        "governance": (
            "Epidemiological decision-support platform. Human verification mandatory for all alerts."
        ),
    }


# Mount API v1 router
app.include_router(api_v1_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
