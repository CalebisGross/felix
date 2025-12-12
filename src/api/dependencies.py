"""
FastAPI dependencies for authentication, Felix instance management, and utilities.
"""

import os
import logging
from typing import Optional
from fastapi import Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

from src.gui.felix_system import FelixSystem, FelixConfig

logger = logging.getLogger(__name__)

# Global Felix instance (singleton pattern for API)
_felix_instance: Optional[FelixSystem] = None
_felix_config: Optional[FelixConfig] = None

# Security
security = HTTPBearer(auto_error=False)


# ============================================================================
# Authentication
# ============================================================================

def get_api_key_from_env() -> str:
    """Get API key from environment variable."""
    api_key = os.getenv("FELIX_API_KEY")
    if not api_key:
        raise ValueError(
            "FELIX_API_KEY environment variable not set. "
            "Please set it to enable API authentication."
        )
    return api_key


async def verify_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Verify API key from Authorization header.

    Expected format: Authorization: Bearer <api_key>

    Raises:
        HTTPException: If API key is invalid or missing

    Returns:
        API key if valid
    """
    # Check if API key is configured
    try:
        expected_key = get_api_key_from_env()
    except ValueError:
        # No API key configured - skip auth (development mode)
        logger.warning("No API key configured. Running in development mode without authentication.")
        return "development"

    # Check authorization header
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify API key
    if authorization.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return authorization.credentials


async def optional_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Optional API key verification.
    Returns None if no key is provided, raises exception if key is invalid.
    """
    if not authorization:
        return None

    try:
        return await verify_api_key(authorization)
    except HTTPException:
        raise


# ============================================================================
# Felix Instance Management
# ============================================================================

def get_felix_config() -> FelixConfig:
    """
    Get Felix configuration from environment variables or defaults.

    Environment variables:
        FELIX_LM_HOST: LM Studio host (default: 127.0.0.1)
        FELIX_LM_PORT: LM Studio port (default: 1234)
        FELIX_MAX_AGENTS: Maximum agents (default: 10)
        FELIX_BASE_TOKEN_BUDGET: Base token budget (default: 2500)
        FELIX_ENABLE_KNOWLEDGE_BRAIN: Enable knowledge brain (default: false)
        FELIX_VERBOSE_LOGGING: Enable verbose logging (default: false)
    """
    global _felix_config

    if _felix_config is None:
        _felix_config = FelixConfig(
            lm_host=os.getenv("FELIX_LM_HOST", "127.0.0.1"),
            lm_port=int(os.getenv("FELIX_LM_PORT", "1234")),
            helix_top_radius=float(os.getenv("FELIX_HELIX_TOP_RADIUS", "3.0")),
            helix_bottom_radius=float(os.getenv("FELIX_HELIX_BOTTOM_RADIUS", "0.5")),
            helix_height=float(os.getenv("FELIX_HELIX_HEIGHT", "8.0")),
            helix_turns=int(os.getenv("FELIX_HELIX_TURNS", "2")),
            max_agents=int(os.getenv("FELIX_MAX_AGENTS", "10")),
            base_token_budget=int(os.getenv("FELIX_BASE_TOKEN_BUDGET", "20000")),  # 50K context window
            enable_knowledge_brain=os.getenv("FELIX_ENABLE_KNOWLEDGE_BRAIN", "false").lower() == "true",
            verbose_llm_logging=os.getenv("FELIX_VERBOSE_LOGGING", "false").lower() == "true",

            # Additional config with sensible defaults
            workflow_max_steps_simple=int(os.getenv("FELIX_MAX_STEPS_SIMPLE", "5")),
            workflow_max_steps_medium=int(os.getenv("FELIX_MAX_STEPS_MEDIUM", "10")),
            workflow_max_steps_complex=int(os.getenv("FELIX_MAX_STEPS_COMPLEX", "20")),
            enable_dynamic_spawning=os.getenv("FELIX_ENABLE_DYNAMIC_SPAWNING", "true").lower() == "true",
        )

        logger.info(f"Felix configuration loaded: max_agents={_felix_config.max_agents}, "
                   f"knowledge_brain={_felix_config.enable_knowledge_brain}")

    return _felix_config


def initialize_felix() -> FelixSystem:
    """
    Initialize Felix system with configuration.

    Returns:
        Initialized FelixSystem instance

    Raises:
        RuntimeError: If Felix fails to initialize
    """
    global _felix_instance

    if _felix_instance is not None:
        logger.warning("Felix instance already initialized")
        return _felix_instance

    logger.info("Initializing Felix system...")

    config = get_felix_config()
    _felix_instance = FelixSystem(config)

    success = _felix_instance.start()

    if not success:
        _felix_instance = None
        raise RuntimeError("Failed to start Felix system. Check logs for details.")

    logger.info("Felix system initialized successfully")
    return _felix_instance


def get_felix() -> FelixSystem:
    """
    Get the global Felix instance.

    Returns:
        FelixSystem instance

    Raises:
        HTTPException: If Felix is not initialized
    """
    global _felix_instance

    if _felix_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Felix system is not initialized. Call POST /api/v1/system/start first."
        )

    return _felix_instance


def shutdown_felix() -> None:
    """Shutdown Felix system and cleanup resources."""
    global _felix_instance

    # Cleanup knowledge brain first
    cleanup_knowledge_brain()

    # Cleanup memory systems
    cleanup_memory_systems()

    if _felix_instance is not None:
        logger.info("Shutting down Felix system...")
        _felix_instance.stop()
        _felix_instance = None
        logger.info("Felix system shutdown complete")


def is_felix_running() -> bool:
    """Check if Felix system is running."""
    return _felix_instance is not None


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_authenticated_felix(
    api_key: str = Depends(verify_api_key),
    felix: FelixSystem = Depends(get_felix)
) -> FelixSystem:
    """
    Dependency that requires both authentication and Felix instance.

    Use this for endpoints that need authentication and Felix access.
    """
    return felix


async def get_optional_auth_felix(
    api_key: Optional[str] = Depends(optional_api_key),
    felix: FelixSystem = Depends(get_felix)
) -> FelixSystem:
    """
    Dependency with optional authentication.

    Use this for endpoints that allow anonymous access.
    """
    return felix


# ============================================================================
# Knowledge Brain Dependencies
# ============================================================================

# Global singletons for knowledge brain components
_knowledge_daemon: Optional[any] = None
_document_reader: Optional[any] = None
_knowledge_retriever: Optional[any] = None
_graph_builder: Optional[any] = None


def verify_knowledge_brain_enabled() -> None:
    """
    Verify that knowledge brain is enabled in configuration.

    Raises:
        HTTPException: If knowledge brain is disabled
    """
    config = get_felix_config()
    if not config.enable_knowledge_brain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge Brain is not enabled. Set FELIX_ENABLE_KNOWLEDGE_BRAIN=true to enable."
        )


def get_knowledge_store():
    """
    Get KnowledgeStore instance.

    Returns:
        KnowledgeStore instance

    Raises:
        HTTPException: If knowledge brain is disabled or store unavailable
    """
    verify_knowledge_brain_enabled()

    try:
        from src.memory.knowledge_store import KnowledgeStore
        return KnowledgeStore()
    except Exception as e:
        logger.exception("Failed to initialize KnowledgeStore")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Knowledge Store: {str(e)}"
        )


def get_document_reader():
    """
    Get DocumentReader instance (singleton).

    Returns:
        DocumentReader instance

    Raises:
        HTTPException: If knowledge brain is disabled
    """
    global _document_reader

    verify_knowledge_brain_enabled()

    if _document_reader is None:
        try:
            from src.knowledge.document_ingest import DocumentReader
            _document_reader = DocumentReader()
            logger.info("DocumentReader initialized")
        except Exception as e:
            logger.exception("Failed to initialize DocumentReader")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize Document Reader: {str(e)}"
            )

    return _document_reader


def get_knowledge_retriever():
    """
    Get KnowledgeRetriever instance (singleton).

    Returns:
        KnowledgeRetriever instance

    Raises:
        HTTPException: If knowledge brain is disabled
    """
    global _knowledge_retriever

    verify_knowledge_brain_enabled()

    if _knowledge_retriever is None:
        try:
            from src.knowledge.retrieval import KnowledgeRetriever
            knowledge_store = get_knowledge_store()
            _knowledge_retriever = KnowledgeRetriever(knowledge_store)
            logger.info("KnowledgeRetriever initialized")
        except Exception as e:
            logger.exception("Failed to initialize KnowledgeRetriever")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize Knowledge Retriever: {str(e)}"
            )

    return _knowledge_retriever


def get_graph_builder():
    """
    Get KnowledgeGraphBuilder instance (singleton).

    Returns:
        KnowledgeGraphBuilder instance

    Raises:
        HTTPException: If knowledge brain is disabled
    """
    global _graph_builder

    verify_knowledge_brain_enabled()

    if _graph_builder is None:
        try:
            from src.knowledge.graph_builder import KnowledgeGraphBuilder
            knowledge_store = get_knowledge_store()
            _graph_builder = KnowledgeGraphBuilder(knowledge_store)
            logger.info("KnowledgeGraphBuilder initialized")
        except Exception as e:
            logger.exception("Failed to initialize KnowledgeGraphBuilder")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize Knowledge Graph Builder: {str(e)}"
            )

    return _graph_builder


def get_knowledge_daemon():
    """
    Get KnowledgeDaemon instance (singleton).

    Note: The daemon is not started automatically. Call daemon.start() explicitly.

    Returns:
        KnowledgeDaemon instance

    Raises:
        HTTPException: If knowledge brain is disabled
    """
    global _knowledge_daemon

    verify_knowledge_brain_enabled()

    if _knowledge_daemon is None:
        try:
            from src.knowledge.knowledge_daemon import KnowledgeDaemon, DaemonConfig
            from src.knowledge.document_ingest import DocumentReader
            from src.knowledge.comprehension import KnowledgeComprehensionEngine
            from src.knowledge.embeddings import EmbeddingProvider
            from src.knowledge.graph_builder import KnowledgeGraphBuilder

            knowledge_store = get_knowledge_store()
            document_reader = get_document_reader()

            # Get LLM provider from Felix if available
            llm_provider = None
            if is_felix_running():
                try:
                    felix = get_felix()
                    llm_provider = felix.llm_provider
                except:
                    logger.warning("Could not get LLM provider from Felix for daemon")

            # Create comprehension engine
            comprehension_engine = KnowledgeComprehensionEngine(
                llm_provider=llm_provider,
                knowledge_store=knowledge_store
            )

            # Create embedding provider
            embedding_provider = EmbeddingProvider(
                lm_studio_client=llm_provider if llm_provider else None
            )

            # Create graph builder
            graph_builder = get_graph_builder()

            # Get watch directories from config
            config = get_felix_config()
            watch_dirs = os.getenv("FELIX_KNOWLEDGE_WATCH_DIRS", "./knowledge_sources")
            watch_dirs = [d.strip() for d in watch_dirs.split(",")]

            # Create daemon config
            daemon_config = DaemonConfig(
                watch_directories=watch_dirs,
                refinement_interval=int(os.getenv("FELIX_KNOWLEDGE_REFINEMENT_INTERVAL", "3600")),
                processing_threads=int(os.getenv("FELIX_KNOWLEDGE_PROCESSING_THREADS", "2")),
                max_memory_mb=int(os.getenv("FELIX_KNOWLEDGE_MAX_MEMORY_MB", "512")),
                enable_file_watcher=os.getenv("FELIX_KNOWLEDGE_FILE_WATCHER", "true").lower() == "true"
            )

            # Create daemon
            _knowledge_daemon = KnowledgeDaemon(
                document_reader=document_reader,
                comprehension_engine=comprehension_engine,
                embedding_provider=embedding_provider,
                graph_builder=graph_builder,
                knowledge_store=knowledge_store,
                config=daemon_config
            )

            logger.info("KnowledgeDaemon initialized (not started)")
        except Exception as e:
            logger.exception("Failed to initialize KnowledgeDaemon")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize Knowledge Daemon: {str(e)}"
            )

    return _knowledge_daemon


def cleanup_knowledge_brain() -> None:
    """Clean up knowledge brain singletons (called on shutdown)."""
    global _knowledge_daemon, _document_reader, _knowledge_retriever, _graph_builder

    if _knowledge_daemon is not None:
        try:
            logger.info("Stopping knowledge daemon...")
            _knowledge_daemon.stop()
        except Exception as e:
            logger.error(f"Error stopping knowledge daemon: {e}")
        _knowledge_daemon = None

    _document_reader = None
    _knowledge_retriever = None
    _graph_builder = None

    logger.info("Knowledge Brain cleanup complete")


# ============================================================================
# Memory System Dependencies
# ============================================================================

# Global singletons for memory components
_task_memory: Optional[any] = None
_workflow_history: Optional[any] = None
_context_compressor: Optional[any] = None


def get_task_memory():
    """
    Get TaskMemory instance (singleton).

    Returns:
        TaskMemory instance

    Raises:
        HTTPException: If TaskMemory unavailable
    """
    global _task_memory

    if _task_memory is None:
        try:
            from src.memory.task_memory import TaskMemory
            _task_memory = TaskMemory()
            logger.info("TaskMemory initialized")
        except Exception as e:
            logger.exception("Failed to initialize TaskMemory")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize Task Memory: {str(e)}"
            )

    return _task_memory


def get_workflow_history():
    """
    Get WorkflowHistory instance (singleton).

    Returns:
        WorkflowHistory instance

    Raises:
        HTTPException: If WorkflowHistory unavailable
    """
    global _workflow_history

    if _workflow_history is None:
        try:
            from src.memory.workflow_history import WorkflowHistory
            _workflow_history = WorkflowHistory()
            logger.info("WorkflowHistory initialized")
        except Exception as e:
            logger.exception("Failed to initialize WorkflowHistory")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize Workflow History: {str(e)}"
            )

    return _workflow_history


def get_context_compressor():
    """
    Get ContextCompressor instance (singleton).

    Returns:
        ContextCompressor instance

    Raises:
        HTTPException: If ContextCompressor unavailable
    """
    global _context_compressor

    if _context_compressor is None:
        try:
            from src.memory.context_compression import ContextCompressor, CompressionConfig
            # Use default configuration
            config = CompressionConfig()
            _context_compressor = ContextCompressor(config)
            logger.info("ContextCompressor initialized")
        except Exception as e:
            logger.exception("Failed to initialize ContextCompressor")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize Context Compressor: {str(e)}"
            )

    return _context_compressor


def get_knowledge_store_memory():
    """
    Get KnowledgeStore instance for memory API (without knowledge brain check).

    This is separate from get_knowledge_store() which requires knowledge brain enabled.
    Memory API can access knowledge store even if knowledge brain is disabled.

    Returns:
        KnowledgeStore instance

    Raises:
        HTTPException: If KnowledgeStore unavailable
    """
    try:
        from src.memory.knowledge_store import KnowledgeStore
        return KnowledgeStore()
    except Exception as e:
        logger.exception("Failed to initialize KnowledgeStore for memory API")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Knowledge Store: {str(e)}"
        )


def cleanup_memory_systems() -> None:
    """Clean up memory system singletons."""
    global _task_memory, _workflow_history, _context_compressor

    _task_memory = None
    _workflow_history = None
    _context_compressor = None

    logger.info("Memory systems cleanup complete")
