"""
Felix REST API - Main FastAPI Application

This is the main entry point for the Felix REST API server.
Provides programmatic access to Felix's multi-agent workflow system.

Usage:
    uvicorn src.api.main:app --reload --port 8000

Environment Variables:
    FELIX_API_KEY: API key for authentication (optional, skips auth if not set)
    FELIX_LM_HOST: LM Studio host (default: 127.0.0.1)
    FELIX_LM_PORT: LM Studio port (default: 1234)
    FELIX_MAX_AGENTS: Maximum concurrent agents (default: 10)
    FELIX_ENABLE_KNOWLEDGE_BRAIN: Enable knowledge brain (default: false)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.api import __version__ as api_version
from src.api.dependencies import (
    initialize_felix,
    shutdown_felix,
    is_felix_running,
    get_felix,
    verify_api_key
)
from src.api.models import SystemStatus, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application state
app_start_time = None


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown of Felix system.
    """
    global app_start_time

    # Startup
    logger.info("=" * 60)
    logger.info("Felix REST API Server Starting")
    logger.info("=" * 60)

    app_start_time = time.time()

    # Felix is initialized on-demand via POST /api/v1/system/start
    # This allows API to start even if LM Studio isn't ready
    logger.info("Felix REST API ready (Felix system will initialize on first /system/start request)")

    yield

    # Shutdown
    logger.info("=" * 60)
    logger.info("Felix REST API Server Shutting Down")
    logger.info("=" * 60)

    shutdown_felix()

    logger.info("Shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Felix REST API",
    description=(
        "REST API for the Felix multi-agent AI framework. "
        "Provides programmatic access to workflows, agents, knowledge brain, and system management."
    ),
    version=api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================================================
# Middleware
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred. Check server logs for details.",
            "details": str(exc) if logger.level == logging.DEBUG else None
        }
    )


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """
    API root endpoint.

    Returns basic information about the API and available endpoints.
    """
    return {
        "name": "Felix REST API",
        "version": api_version,
        "status": "online",
        "felix_running": is_felix_running(),
        "documentation": "/docs",
        "openapi_schema": "/openapi.json"
    }


@app.get("/health", tags=["Root"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns API health status. Felix system status is checked separately.
    """
    return {
        "status": "healthy",
        "api_version": api_version,
        "timestamp": time.time(),
        "felix_initialized": is_felix_running()
    }


# ============================================================================
# System Management Endpoints
# ============================================================================

@app.post("/api/v1/system/start", tags=["System"], response_model=SystemStatus)
async def start_system(api_key: str = Depends(verify_api_key)) -> SystemStatus:
    """
    Initialize and start the Felix system.

    Requires authentication. Initializes all Felix components including:
    - Helix geometry
    - LLM client
    - Agent factory
    - CentralPost communication hub
    - Knowledge brain (if enabled)

    Returns:
        SystemStatus: Current system status

    Raises:
        HTTPException: If Felix is already running or fails to initialize
    """
    if is_felix_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Felix system is already running"
        )

    try:
        felix = initialize_felix()

        # Get system status with defensive defaults
        felix_status = felix.get_system_status()

        try:
            return SystemStatus(
                status="running",
                felix_version=felix_status.get("felix_version", "unknown"),
                api_version=api_version,
                uptime_seconds=0.0,
                active_workflows=felix_status.get("active_workflows", 0),
                active_agents=felix_status.get("active_agents", 0),
                llm_provider=felix_status.get("llm_provider", "unknown"),
                knowledge_brain_enabled=felix_status.get("knowledge_brain_enabled", False)
            )
        except ValidationError as e:
            logger.error(f"System status validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"System status validation failed. Missing required fields: {str(e)}"
            )

    except RuntimeError as e:
        # This typically happens when LM Studio connection fails
        error_msg = str(e)
        logger.error(f"Failed to start Felix system: {error_msg}")
        if "Failed to connect to LLM" in error_msg or "LM Studio" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to start Felix system: Cannot connect to LLM provider. "
                       "Please ensure LM Studio is running with a model loaded on the configured port, "
                       "or configure an alternative LLM provider in config/llm.yaml"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start Felix system: {error_msg}"
            )

    except Exception as e:
        logger.exception("Failed to start Felix system")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start Felix system: {str(e)}"
        )


@app.post("/api/v1/system/stop", tags=["System"])
async def stop_system(api_key: str = Depends(verify_api_key)) -> Dict[str, str]:
    """
    Stop the Felix system and cleanup resources.

    Requires authentication. Shuts down:
    - All active agents
    - Knowledge brain daemon (if running)
    - LLM connections
    - Database connections

    Returns:
        Success message

    Raises:
        HTTPException: If Felix is not running
    """
    if not is_felix_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Felix system is not running"
        )

    try:
        shutdown_felix()
        return {"status": "stopped", "message": "Felix system stopped successfully"}

    except Exception as e:
        logger.exception("Error stopping Felix system")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping Felix system: {str(e)}"
        )


@app.get("/api/v1/system/status", tags=["System"], response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """
    Get current system status.

    Returns detailed information about the Felix system including:
    - Running status
    - Active workflows and agents
    - LLM provider information
    - System uptime

    Does not require authentication for monitoring purposes.

    Returns:
        SystemStatus: Current system status

    Raises:
        HTTPException: If Felix system is not initialized
    """
    if not is_felix_running():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Felix system is not running. Call POST /api/v1/system/start first."
        )

    try:
        felix = get_felix()
        felix_status = felix.get_system_status()

        # Calculate uptime
        uptime = time.time() - app_start_time if app_start_time else 0

        try:
            return SystemStatus(
                status="running",
                felix_version=felix_status.get("felix_version", "unknown"),
                api_version=api_version,
                uptime_seconds=uptime,
                active_workflows=felix_status.get("active_workflows", 0),
                active_agents=felix_status.get("active_agents", 0),
                llm_provider=felix_status.get("llm_provider", "unknown"),
                knowledge_brain_enabled=felix_status.get("knowledge_brain_enabled", False)
            )
        except ValidationError as e:
            logger.error(f"System status validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"System status validation failed. Missing required fields: {str(e)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.exception("Error getting system status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system status: {str(e)}"
        )


# ============================================================================
# Router Registration
# ============================================================================

# Import routers
from src.api.routers import workflows, agents

# Register routers
app.include_router(workflows.router)
app.include_router(agents.router)

# Knowledge Brain router (optional - only if enabled)
try:
    from src.api.routers import knowledge
    app.include_router(knowledge.router)
    logger.info("Knowledge Brain API enabled")
except ImportError as e:
    logger.warning(f"Knowledge Brain API not available: {e}")

# WebSocket support (optional - only if dependencies available)
try:
    from src.api.websockets import workflow_stream
    app.include_router(workflow_stream.router)
    logger.info("WebSocket streaming enabled")
except ImportError as e:
    logger.warning(f"WebSocket support not available: {e}")

# Memory & History API routers
try:
    from src.api.routers import task_memory, workflow_history, knowledge_memory, compression
    app.include_router(task_memory.router)
    app.include_router(workflow_history.router)
    app.include_router(knowledge_memory.router)
    app.include_router(compression.router)
    logger.info("Memory & History API enabled (22 endpoints)")
except ImportError as e:
    logger.warning(f"Memory & History API not available: {e}")


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("Felix REST API Server")
    print("=" * 60)
    print(f"Version: {api_version}")
    print(f"Documentation: http://localhost:8000/docs")
    print(f"OpenAPI Schema: http://localhost:8000/openapi.json")
    print("=" * 60 + "\n")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
