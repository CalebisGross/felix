"""
Knowledge Daemon for Felix Knowledge Brain

Autonomous background processing system with six concurrent modes:
- Mode A: Initial Batch Processing - Process existing documents in directories
- Mode B: Continuous Refinement - Periodically re-analyze knowledge for new connections
- Mode C: File System Watching - Monitor directories for new documents
- Mode D: Scheduled Backups - Automatic database and JSON backups
- Mode E: Gap-Directed Learning - Proactive knowledge acquisition to fill gaps (OPT-IN)
- Mode F: Knowledge Pruning - Archive low-value knowledge to close feedback loops (Issue #22)

Runs indefinitely in background threads with graceful shutdown support.
"""

import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from queue import Queue, Empty
from datetime import datetime

# File system monitoring (optional)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None
    FileCreatedEvent = None
    FileModifiedEvent = None

from .document_ingest import DocumentReader, BatchDocumentProcessor
from .comprehension import KnowledgeComprehensionEngine
from .graph_builder import KnowledgeGraphBuilder
from .embeddings import EmbeddingProvider
from src.memory.knowledge_store import KnowledgeStore
from src.core.felixignore import should_ignore, load_felixignore

# Strategic comprehension (efficiency improvements)
try:
    from .strategic_comprehension import StrategicComprehensionEngine
    STRATEGIC_COMPREHENSION_AVAILABLE = True
except ImportError:
    STRATEGIC_COMPREHENSION_AVAILABLE = False
    StrategicComprehensionEngine = None

# Optional backup manager (Phase 5 feature)
try:
    from .backup_manager_extended import KnowledgeBackupManager
    BACKUP_MANAGER_AVAILABLE = True
except ImportError:
    BACKUP_MANAGER_AVAILABLE = False
    KnowledgeBackupManager = None

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for knowledge daemon."""
    watch_directories: List[str]
    enable_batch_processing: bool = True
    enable_refinement: bool = True
    enable_file_watching: bool = True
    enable_scheduled_backup: bool = False  # Enable automatic backups
    enable_gap_learning: bool = False  # Mode E: Gap-directed learning (OPT-IN)
    enable_pruning: bool = True  # Mode F: Knowledge pruning (Issue #22)
    enable_task_memory_cleanup: bool = True  # Issue #24: Auto-cleanup TaskMemory patterns
    use_strategic_comprehension: bool = True  # Use optimized comprehension (50-70% fewer LLM calls)
    refinement_interval: int = 3600  # seconds (1 hour)
    backup_interval: int = 86400  # seconds (24 hours)
    gap_learning_interval: int = 1800  # seconds (30 minutes) for gap acquisition
    pruning_interval: int = 604800  # seconds (7 days) - weekly pruning cycle
    pruning_usefulness_threshold: float = 0.3  # Archive if avg usefulness below this
    pruning_min_usage_count: int = 10  # Minimum usage samples before archiving
    pruning_stale_days: int = 90  # Archive if not accessed in this many days
    task_memory_max_age_days: int = 60  # Issue #24: Max age for task patterns
    task_memory_min_usage: int = 2  # Issue #24: Min usage count to keep patterns
    gap_min_severity: float = 0.6  # Minimum gap severity for acquisition
    gap_min_occurrences: int = 3  # Minimum gap occurrences for acquisition
    backup_compress: bool = True  # Use compression for JSON backups
    backup_keep_days: int = 30  # Keep backups for N days
    processing_threads: int = 2
    max_memory_mb: int = 512
    chunk_size: int = 1000
    chunk_overlap: int = 200
    exclusion_patterns: List[str] = None  # Path patterns to exclude from scanning

    def __post_init__(self):
        """Initialize exclusion patterns with defaults if not provided."""
        if self.exclusion_patterns is None:
            self.exclusion_patterns = [
                "*/.venv/*",
                "*/.venv/**",
                "*/venv/*",
                "*/venv/**",
                "*/node_modules/*",
                "*/node_modules/**",
                "*/.git/*",
                "*/.git/**",
                "*/__pycache__/*",
                "*/__pycache__/**",
                "*/dist/*",
                "*/dist/**",
                "*/build/*",
                "*/build/**",
                "*/.pytest_cache/*",
                "*/.mypy_cache/*",
                "*/site-packages/*",
                "*/site-packages/**",
                "*/.tox/*",
                "*/.nox/*",
                "*/htmlcov/*",
                "*/.coverage",
                "*/.env",
                "*/.vscode/*",
                "*/.idea/*",
            ]


@dataclass
class DaemonStatus:
    """Current status of daemon."""
    running: bool
    batch_processor_active: bool
    refiner_active: bool
    file_watcher_active: bool
    backup_active: bool  # Phase 5
    pruner_active: bool  # Issue #22
    documents_pending: int
    documents_processed: int
    documents_failed: int
    last_refinement: Optional[float]
    last_backup: Optional[float]  # Phase 5
    last_pruning: Optional[float]  # Issue #22
    last_activity: Optional[float]
    uptime_seconds: float
    last_task_memory_cleanup: Optional[float] = None  # Issue #24
    entries_archived: int = 0  # Issue #22 - total entries archived by pruner
    task_patterns_cleaned: int = 0  # Issue #24 - total task patterns cleaned


class DocumentQueue:
    """Thread-safe queue for document processing."""

    def __init__(self):
        self.queue = Queue()
        self.processing = set()
        self.completed = set()
        self.failed = set()
        self.lock = threading.Lock()

    def add(self, file_path: str):
        """Add document to queue."""
        with self.lock:
            if file_path not in self.processing and file_path not in self.completed:
                self.queue.put(file_path)

    def get(self, timeout: float = 1.0) -> Optional[str]:
        """Get next document from queue."""
        try:
            file_path = self.queue.get(timeout=timeout)
            with self.lock:
                self.processing.add(file_path)
            return file_path
        except Empty:
            return None

    def mark_completed(self, file_path: str):
        """Mark document as completed."""
        with self.lock:
            self.processing.discard(file_path)
            self.completed.add(file_path)

    def mark_failed(self, file_path: str):
        """Mark document as failed."""
        with self.lock:
            self.processing.discard(file_path)
            self.failed.add(file_path)

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self.lock:
            return {
                'pending': self.queue.qsize(),
                'processing': len(self.processing),
                'completed': len(self.completed),
                'failed': len(self.failed)
            }


if WATCHDOG_AVAILABLE:
    class DocumentFileHandler(FileSystemEventHandler):
        """Handles file system events for document monitoring."""

        def __init__(self, document_queue: DocumentQueue, file_patterns: List[str],
                     exclusion_patterns: List[str] = None):
            super().__init__()
            self.document_queue = document_queue
            self.file_patterns = file_patterns
            self.exclusion_patterns = exclusion_patterns or []

        def _should_exclude(self, file_path: str) -> bool:
            """Check if file path matches any exclusion patterns."""
            import fnmatch

            for pattern in self.exclusion_patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    return True
            return False

        def _should_process(self, file_path: str) -> bool:
            """Check if file should be processed."""
            # First check exclusions
            if self._should_exclude(file_path):
                return False

            # Then check if matches file patterns
            path = Path(file_path)
            return any(path.match(pattern) for pattern in self.file_patterns)

        def on_created(self, event):
            """Handle file creation events."""
            if not event.is_directory and self._should_process(event.src_path):
                logger.info(f"New document detected: {event.src_path}")
                self.document_queue.add(event.src_path)

        def on_modified(self, event):
            """Handle file modification events."""
            if not event.is_directory and self._should_process(event.src_path):
                # Check if file hash changed (significant modification)
                # For now, just re-queue
                logger.info(f"Document modified: {event.src_path}")
                self.document_queue.add(event.src_path)
else:
    # Dummy class when watchdog not available
    class DocumentFileHandler:
        def __init__(self, *args, **kwargs):
            pass


class KnowledgeDaemon:
    """
    Autonomous knowledge processing daemon.

    Runs three concurrent background threads for document ingestion and learning.
    """

    def __init__(self,
                 config: DaemonConfig,
                 knowledge_store: KnowledgeStore,
                 llm_client,
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """
        Initialize knowledge daemon.

        Args:
            config: Daemon configuration
            knowledge_store: KnowledgeStore instance
            llm_client: LLM client for comprehension agents
            embedding_provider: Optional embedding provider
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.knowledge_store = knowledge_store

        # Initialize centralized .felixignore patterns
        root_path = config.watch_directories[0] if config.watch_directories else '.'
        load_felixignore(root_path)
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.progress_callback = progress_callback

        # Processing components
        self.document_reader = DocumentReader(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        # Initialize comprehension engine (strategic or original)
        self.use_strategic = config.use_strategic_comprehension and STRATEGIC_COMPREHENSION_AVAILABLE
        if self.use_strategic:
            self.strategic_engine = StrategicComprehensionEngine(
                knowledge_store=knowledge_store,
                llm_client=llm_client,
                embedding_provider=embedding_provider
            )
            self.comprehension_engine = None  # Not used in strategic mode
            logger.info("✓ Strategic comprehension enabled (50-70% fewer LLM calls)")
        else:
            self.strategic_engine = None
            self.comprehension_engine = KnowledgeComprehensionEngine(
                knowledge_store=knowledge_store,
                llm_client=llm_client,
                embedding_provider=embedding_provider  # Now generates embeddings for each concept
            )
            if config.use_strategic_comprehension and not STRATEGIC_COMPREHENSION_AVAILABLE:
                logger.warning("Strategic comprehension requested but not available, using original")

        self.graph_builder = KnowledgeGraphBuilder(
            knowledge_store=knowledge_store,
            embedding_provider=embedding_provider
        )

        # Document queue
        self.document_queue = DocumentQueue()

        # Thread management
        self.running = False
        self.threads = []
        self.start_time = None
        self.last_refinement = None
        self.last_backup = None
        self.last_activity = None
        self._stop_event = threading.Event()  # For interruptible shutdown

        # File watching
        self.observer = None

        # Backup manager (Phase 5)
        self.backup_manager = None
        if BACKUP_MANAGER_AVAILABLE and config.enable_scheduled_backup:
            try:
                self.backup_manager = KnowledgeBackupManager(knowledge_store)
                logger.info("✓ Backup manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize backup manager: {e}")

    def _should_exclude_path(self, file_path: Path) -> bool:
        """
        Check if a file path should be excluded using centralized .felixignore.

        Args:
            file_path: Path to check

        Returns:
            True if path should be excluded, False otherwise
        """
        return should_ignore(file_path)

    def start(self):
        """Start all daemon modes."""
        if self.running:
            logger.warning("Daemon already running")
            return

        logger.info("Starting Knowledge Daemon...")
        self.running = True
        self._stop_event.clear()  # Reset stop event for fresh start
        self.start_time = time.time()

        # Mode A: Batch Processing
        if self.config.enable_batch_processing:
            batch_thread = threading.Thread(
                target=self._batch_processing_loop,
                name="KnowledgeDaemon-Batch",
                daemon=True
            )
            batch_thread.start()
            self.threads.append(batch_thread)
            logger.info("✓ Mode A: Batch processing started")

        # Mode B: Continuous Refinement
        if self.config.enable_refinement:
            refine_thread = threading.Thread(
                target=self._refinement_loop,
                name="KnowledgeDaemon-Refine",
                daemon=True
            )
            refine_thread.start()
            self.threads.append(refine_thread)
            logger.info("✓ Mode B: Continuous refinement started")

        # Mode C: File Watching
        if self.config.enable_file_watching:
            self._start_file_watching()
            logger.info("✓ Mode C: File watching started")

        # Mode D: Scheduled Backups (Phase 5)
        if self.config.enable_scheduled_backup and self.backup_manager:
            backup_thread = threading.Thread(
                target=self._backup_loop,
                name="KnowledgeDaemon-Backup",
                daemon=True
            )
            backup_thread.start()
            self.threads.append(backup_thread)
            logger.info("✓ Mode D: Scheduled backups started")

        # Mode E: Gap-Directed Learning (Phase 7 - OPT-IN)
        if self.config.enable_gap_learning:
            gap_thread = threading.Thread(
                target=self._gap_directed_learning_loop,
                name="KnowledgeDaemon-GapLearning",
                daemon=True
            )
            gap_thread.start()
            self.threads.append(gap_thread)
            logger.info("✓ Mode E: Gap-directed learning started (opt-in)")

        # Mode F: Knowledge Pruning (Issue #22)
        if self.config.enable_pruning:
            prune_thread = threading.Thread(
                target=self._pruning_loop,
                name="KnowledgeDaemon-Pruning",
                daemon=True
            )
            prune_thread.start()
            self.threads.append(prune_thread)
            logger.info("✓ Mode F: Knowledge pruning started (weekly cycle)")

        # Load any pending documents from database into queue
        # This ensures documents that were pending before a restart get processed
        self._load_pending_from_database()

        logger.info("Knowledge Daemon fully operational")

    def stop(self):
        """Stop all daemon modes gracefully."""
        if not self.running:
            return

        logger.info("Stopping Knowledge Daemon...")
        self.running = False
        self._stop_event.set()  # Signal all threads to stop immediately

        # Stop file watching
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping file observer: {e}")
            finally:
                self.observer = None  # Reset observer for clean restart

        # Wait for threads to finish (with increased timeout for graceful shutdown)
        for thread in self.threads:
            thread.join(timeout=10.0)  # Increased from 2.0 to 10.0 seconds
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} still alive after 10 second timeout")

        # Clear thread list for clean restart
        self.threads.clear()

        logger.info("Knowledge Daemon stopped")

    def force_stop(self):
        """
        Forcefully stop daemon without waiting for threads.

        Use this as an emergency option if normal stop() hangs.
        Threads may remain active after this call.
        """
        logger.warning("Force-stopping Knowledge Daemon (threads may remain active)")
        self.running = False

        # Stop file watching immediately
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=1)  # Very short timeout
            except Exception as e:
                logger.warning(f"Error force-stopping file observer: {e}")
            finally:
                self.observer = None

        # Don't wait for threads - just clear the list
        self.threads.clear()

        logger.warning("Knowledge Daemon force-stopped (check for orphaned threads)")

    def get_status(self) -> DaemonStatus:
        """Get current daemon status."""
        queue_stats = self.document_queue.get_stats()

        uptime = time.time() - self.start_time if self.start_time else 0

        # Get both queue count and database count
        # Queue count: documents added but not yet pulled for processing
        # Database count: documents with pending/processing status (persistent)
        db_pending = self._get_database_pending_count()
        queue_pending = queue_stats['pending']

        # Use max to capture both newly queued and persistent state
        # This handles the case where docs are in queue but DB not yet updated
        total_pending = max(queue_pending, db_pending)

        return DaemonStatus(
            running=self.running,
            batch_processor_active=self.config.enable_batch_processing,
            refiner_active=self.config.enable_refinement,
            file_watcher_active=self.config.enable_file_watching and self.observer is not None,
            backup_active=self.config.enable_scheduled_backup and self.backup_manager is not None,
            pruner_active=self.config.enable_pruning,
            documents_pending=total_pending,  # Max of queue and database
            documents_processed=queue_stats['completed'],
            documents_failed=queue_stats['failed'],
            last_refinement=self.last_refinement,
            last_backup=self.last_backup,
            last_pruning=getattr(self, '_last_pruning_time', None),
            last_task_memory_cleanup=getattr(self, '_last_pruning_time', None),  # Issue #24: Same as pruning
            last_activity=self.last_activity,
            uptime_seconds=uptime,
            entries_archived=getattr(self, '_total_entries_archived', 0),
            task_patterns_cleaned=getattr(self, '_total_task_patterns_cleaned', 0)  # Issue #24
        )

    def _get_completed_documents(self) -> set:
        """
        Get set of file paths that have been successfully processed.

        Returns:
            Set of absolute file paths for documents with status='completed'
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0)
            cursor = conn.execute("""
                SELECT file_path FROM document_sources
                WHERE ingestion_status = 'completed'
            """)
            completed = {row[0] for row in cursor.fetchall()}
            conn.close()
            logger.info(f"Found {len(completed)} already-processed documents in database")
            return completed
        except Exception as e:
            logger.error(f"Failed to load completed documents: {e}")
            return set()

    def _get_database_pending_count(self) -> int:
        """
        Get count of documents that need processing from database.

        Returns:
            Number of documents with ingestion_status='pending' or 'processing'
            (processing means stuck from a previous crashed/restarted session)
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM document_sources WHERE ingestion_status IN ('pending', 'processing')"
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def _load_pending_from_database(self):
        """
        Load pending/stuck documents from database into processing queue.

        Called on daemon start to restore queue state from persistent storage.
        This ensures documents that were pending or stuck in 'processing' status
        from a previous session get (re)processed.
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0)
            # Load both 'pending' and 'processing' - the latter are stuck from previous runs
            cursor = conn.execute(
                "SELECT file_path FROM document_sources WHERE ingestion_status IN ('pending', 'processing')"
            )
            loaded = 0
            for row in cursor:
                file_path = row[0]
                if Path(file_path).exists():
                    self.document_queue.add(file_path)
                    loaded += 1
                else:
                    logger.warning(f"Pending document no longer exists: {file_path}")
            conn.close()
            if loaded > 0:
                logger.info(f"Loaded {loaded} pending/stuck documents from database into queue")
        except Exception as e:
            logger.error(f"Failed to load pending documents from database: {e}")

    def _create_pending_entries_batch(self, file_paths: list):
        """
        Create or update database entries for multiple pending documents in a single transaction.

        Called when documents are queued via process_directory_now() to ensure
        the database reflects the queue state immediately.

        Uses UPSERT logic: inserts new files, updates existing files to 'pending' status.
        Batches all operations in a single transaction for performance.
        """
        import sqlite3
        import hashlib

        if not file_paths:
            return

        # File type mapping (extension -> database value)
        file_type_map = {
            '.pdf': 'pdf', '.txt': 'text', '.md': 'markdown',
            '.py': 'python', '.js': 'javascript', '.java': 'java',
            '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.hpp': 'cpp',
        }

        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0)

            # Prepare batch data with file_type
            batch_data = []
            for file_path in file_paths:
                path = Path(file_path)
                doc_id = hashlib.md5(file_path.encode()).hexdigest()[:16]
                file_type = file_type_map.get(path.suffix.lower(), 'text')
                batch_data.append((doc_id, file_path, path.name, file_type))

            # Execute all inserts in a single transaction
            conn.executemany("""
                INSERT INTO document_sources (doc_id, file_path, file_name, file_type, ingestion_status)
                VALUES (?, ?, ?, ?, 'pending')
                ON CONFLICT(file_path) DO UPDATE SET ingestion_status = 'pending'
            """, batch_data)

            conn.commit()
            conn.close()
            logger.info(f"Created/updated {len(file_paths)} pending entries in database")
        except Exception as e:
            logger.warning(f"Failed to create pending entries: {e}")

    def _batch_processing_loop(self):
        """
        Mode A: Batch process existing documents.

        Scans watch directories and queues all documents for processing.
        Only queues documents that haven't been successfully processed yet.
        """
        logger.info("Batch processing: Scanning directories...")

        # Get list of already-completed documents from database
        already_processed = self._get_completed_documents()

        # Scan all watch directories
        for watch_dir in self.config.watch_directories:
            path = Path(watch_dir)
            if not path.exists():
                logger.warning(f"Watch directory does not exist: {watch_dir}")
                continue

            # Find all documents
            patterns = ['*.pdf', '*.txt', '*.md', '*.py', '*.js', '*.java']
            for pattern in patterns:
                for file_path in path.rglob(pattern):
                    # Skip excluded paths
                    if self._should_exclude_path(file_path):
                        continue

                    file_path_str = str(file_path.absolute())
                    # Skip if already processed
                    if file_path_str not in already_processed:
                        self.document_queue.add(file_path_str)

        logger.info(f"Batch processing: {self.document_queue.get_stats()['pending']} new documents queued "
                   f"({len(already_processed)} already processed)")

        # Process queue
        while self.running and not self._stop_event.is_set():
            file_path = self.document_queue.get(timeout=0.5)  # Shorter timeout for responsive shutdown

            if file_path is None:
                # Queue empty - use interruptible wait instead of sleep
                if self._stop_event.wait(timeout=0.5):  # Returns True if stop signaled
                    break
                continue

            # Check stop flag before processing
            if not self.running or self._stop_event.is_set():
                break

            try:
                self._process_document(file_path)
                self.document_queue.mark_completed(file_path)
                self.last_activity = time.time()

                # Report progress
                if self.progress_callback:
                    stats = self.document_queue.get_stats()
                    self.progress_callback('batch_progress', stats)

                # Brief yield between documents - interruptible
                if self._stop_event.wait(timeout=0.2):
                    break

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.document_queue.mark_failed(file_path)

        logger.info("Batch processing loop exited")

    def _refinement_loop(self):
        """
        Mode B: Continuous refinement of knowledge.

        Periodically re-analyzes existing knowledge to discover new connections.
        """
        logger.info("Refinement: Starting continuous refinement loop")

        while self.running and not self._stop_event.is_set():
            # Wait for refinement interval - interruptible via _stop_event
            if self._stop_event.wait(timeout=self.config.refinement_interval):
                break  # Stop was signaled

            if not self.running:
                break

            try:
                logger.info("Refinement: Starting refinement cycle...")
                start_time = time.time()

                # Build/update knowledge graph
                graph_stats = self.graph_builder.build_global_graph()

                # Log statistics
                duration = time.time() - start_time
                logger.info(f"Refinement cycle complete: {graph_stats.get('total_relationships', 0)} relationships, "
                           f"{duration:.1f}s")

                self.last_refinement = time.time()

                # Report progress
                if self.progress_callback:
                    self.progress_callback('refinement_complete', {
                        'duration': duration,
                        **graph_stats
                    })

            except Exception as e:
                logger.error(f"Refinement cycle failed: {e}")

        logger.info("Refinement loop exited")

    def _backup_loop(self):
        """
        Mode D: Scheduled backups of knowledge base.

        Periodically creates database and JSON backups with automatic cleanup.
        Phase 5 feature - requires KnowledgeBackupManager.
        """
        if not self.backup_manager:
            logger.warning("Backup loop started but backup manager not available")
            return

        logger.info("Backup: Starting scheduled backup loop")

        # Perform initial backup after startup
        try:
            logger.info("Backup: Creating initial backup...")
            backups = self.backup_manager.create_scheduled_backup(
                include_database=True,
                include_json=True,
                compress=self.config.backup_compress
            )
            self.last_backup = time.time()
            logger.info(f"Initial backup complete: {list(backups.keys())}")

            if self.progress_callback:
                self.progress_callback('backup_complete', {
                    'backups': backups,
                    'initial': True
                })
        except Exception as e:
            logger.error(f"Initial backup failed: {e}")

        while self.running and not self._stop_event.is_set():
            # Wait for backup interval - interruptible via _stop_event
            if self._stop_event.wait(timeout=self.config.backup_interval):
                break  # Stop was signaled

            if not self.running:
                break

            try:
                logger.info("Backup: Starting backup cycle...")
                start_time = time.time()

                # Create backups
                backups = self.backup_manager.create_scheduled_backup(
                    include_database=True,
                    include_json=True,
                    compress=self.config.backup_compress
                )

                # Cleanup old backups
                deleted = self.backup_manager.cleanup_old_backups(
                    max_age_days=self.config.backup_keep_days,
                    keep_minimum=5
                )

                duration = time.time() - start_time
                self.last_backup = time.time()

                logger.info(f"Backup cycle complete: {list(backups.keys())}, "
                          f"deleted {deleted['database'] + deleted['json']} old backups, "
                          f"{duration:.1f}s")

                # Report progress
                if self.progress_callback:
                    self.progress_callback('backup_complete', {
                        'duration': duration,
                        'backups': backups,
                        'deleted': deleted
                    })

            except Exception as e:
                logger.error(f"Backup cycle failed: {e}")

        logger.info("Backup loop exited")

    def _gap_directed_learning_loop(self):
        """
        Mode E: Gap-Directed Learning (OPT-IN).

        Periodically queries for high-priority knowledge gaps and attempts
        to fill them through web search. This is an opt-in feature that
        remains disabled by default.

        Phase 7 - Knowledge Gap Cartography feature.
        """
        logger.info("Gap Learning: Starting gap-directed learning loop (opt-in)")

        # Import gap-directed learning components
        try:
            from .gap_tracker import GapTracker
            from .gap_directed_learning import GapDirectedLearner
        except ImportError as e:
            logger.error(f"Gap learning components not available: {e}")
            return

        # Initialize gap tracker and learner
        gap_tracker = GapTracker(self.knowledge_store.storage_path)

        # Try to get web search client from knowledge store or create one
        web_search_client = None
        try:
            from src.llm.web_search import WebSearchClient
            web_search_client = WebSearchClient()
            logger.info("Gap Learning: Web search client available")
        except Exception as e:
            logger.warning(f"Gap Learning: Web search not available ({e}), "
                          "gap filling will be limited to manual resolution")

        gap_learner = GapDirectedLearner(
            gap_tracker=gap_tracker,
            knowledge_store=self.knowledge_store,
            web_search_client=web_search_client
        )

        # Track last run time
        last_gap_check = 0

        while self.running and not self._stop_event.is_set():
            # Wait for gap learning interval - interruptible via _stop_event
            if self._stop_event.wait(timeout=self.config.gap_learning_interval):
                break  # Stop was signaled

            if not self.running:
                break

            try:
                logger.info("Gap Learning: Starting acquisition cycle...")
                start_time = time.time()

                # Run acquisition cycle with configured thresholds
                result = gap_learner.run_acquisition_cycle(
                    max_targets=3  # Process up to 3 gaps per cycle
                )

                duration = time.time() - start_time
                last_gap_check = time.time()

                # Get acquisition stats
                stats = gap_learner.get_acquisition_stats()

                logger.info(f"Gap Learning cycle complete: "
                           f"{result['targets_resolved']}/{result['targets_found']} gaps resolved, "
                           f"{result['entries_created']} entries created, "
                           f"{duration:.1f}s")

                # Report progress
                if self.progress_callback:
                    self.progress_callback('gap_learning_complete', {
                        'duration': duration,
                        'targets_found': result['targets_found'],
                        'targets_resolved': result['targets_resolved'],
                        'entries_created': result['entries_created'],
                        'stats': stats
                    })

            except Exception as e:
                logger.error(f"Gap learning cycle failed: {e}")

        logger.info("Gap learning loop exited")

    def _pruning_loop(self):
        """
        Mode F: Knowledge Pruning (Issue #22).

        Periodically prunes low-value knowledge entries to prevent the
        knowledge base from becoming cluttered with noise. This closes
        the learning feedback loop by actually removing knowledge that
        has proven unhelpful.

        Runs weekly by default (configurable via pruning_interval).
        """
        logger.info("Pruning: Starting knowledge pruning loop")

        # Track total archived across all cycles
        self._total_entries_archived = 0
        self._last_pruning_time = None

        while self.running and not self._stop_event.is_set():
            # Wait for pruning interval - interruptible via _stop_event
            if self._stop_event.wait(timeout=self.config.pruning_interval):
                break  # Stop was signaled

            if not self.running:
                break

            try:
                logger.info("Pruning: Starting pruning cycle...")
                start_time = time.time()

                # Step 1: Prune low-value knowledge
                prune_result = self.knowledge_store.prune_low_value_knowledge(
                    min_usefulness_threshold=self.config.pruning_usefulness_threshold,
                    min_usage_count=self.config.pruning_min_usage_count,
                    stale_days=self.config.pruning_stale_days,
                    dry_run=False
                )

                # Step 2: Apply confidence decay to remaining entries
                decay_result = self.knowledge_store.apply_confidence_decay(
                    stale_days=self.config.pruning_stale_days
                )

                # Step 3 (Issue #24): Clean up old TaskMemory patterns
                task_patterns_cleaned = 0
                if self.config.enable_task_memory_cleanup:
                    task_patterns_cleaned = self._cleanup_task_memory_patterns()

                duration = time.time() - start_time
                self._last_pruning_time = time.time()
                self._total_entries_archived += prune_result.get('total_archived', 0)
                self._total_task_patterns_cleaned = getattr(self, '_total_task_patterns_cleaned', 0) + task_patterns_cleaned

                logger.info(f"Pruning cycle complete: "
                           f"{prune_result.get('total_archived', 0)} entries archived "
                           f"({prune_result.get('archived_low_usefulness', 0)} low-usefulness, "
                           f"{prune_result.get('archived_stale', 0)} stale), "
                           f"{decay_result.get('entries_decayed', 0)} entries decayed, "
                           f"{task_patterns_cleaned} task patterns cleaned, "
                           f"{duration:.1f}s")

                # Report progress
                if self.progress_callback:
                    self.progress_callback('pruning_complete', {
                        'duration': duration,
                        'archived_low_usefulness': prune_result.get('archived_low_usefulness', 0),
                        'archived_stale': prune_result.get('archived_stale', 0),
                        'total_archived': prune_result.get('total_archived', 0),
                        'entries_decayed': decay_result.get('entries_decayed', 0),
                        'task_patterns_cleaned': task_patterns_cleaned,
                        'total_archived_all_time': self._total_entries_archived
                    })

            except Exception as e:
                logger.error(f"Pruning cycle failed: {e}")

        logger.info("Pruning loop exited")

    def _cleanup_task_memory_patterns(self) -> int:
        """
        Clean up old/unused TaskMemory patterns (Issue #24).

        Calls TaskMemory.cleanup_old_patterns() to remove patterns that are:
        - Older than config.task_memory_max_age_days
        - Used fewer than config.task_memory_min_usage times

        Returns:
            Number of patterns deleted
        """
        try:
            from src.memory.task_memory import TaskMemory

            # Use default path if not specified
            task_memory = TaskMemory()

            deleted = task_memory.cleanup_old_patterns(
                max_age_days=self.config.task_memory_max_age_days,
                min_usage_count=self.config.task_memory_min_usage
            )

            if deleted > 0:
                logger.info(f"TaskMemory cleanup: removed {deleted} old/unused patterns")

            return deleted

        except ImportError:
            logger.debug("TaskMemory not available for cleanup")
            return 0
        except Exception as e:
            logger.warning(f"TaskMemory cleanup failed: {e}")
            return 0

    def _start_file_watching(self):
        """
        Mode C: Watch file system for new/modified documents.
        """
        try:
            self.observer = Observer()

            # Set up handlers for each watch directory
            file_patterns = ['*.pdf', '*.txt', '*.md', '*.py', '*.js', '*.java']
            handler = DocumentFileHandler(
                self.document_queue,
                file_patterns,
                self.config.exclusion_patterns
            )

            for watch_dir in self.config.watch_directories:
                path = Path(watch_dir)
                if path.exists():
                    self.observer.schedule(handler, str(path), recursive=True)
                    logger.info(f"Watching directory: {watch_dir}")
                else:
                    logger.warning(f"Cannot watch non-existent directory: {watch_dir}")

            self.observer.start()

        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            self.observer = None

    def _process_document(self, file_path: str):
        """
        Process a single document: ingest, comprehend, store.

        Args:
            file_path: Path to document file
        """
        logger.info(f"Processing document: {file_path}")
        start_time = time.time()

        try:
            # Step 1: Ingest document
            ingestion_result = self.document_reader.ingest_document(file_path)

            if not ingestion_result.success:
                raise Exception(f"Ingestion failed: {ingestion_result.error_message}")

            # Store document metadata
            self._store_document_metadata(ingestion_result)

            # Step 2: Generate embeddings for chunks (if provider available)
            if self.embedding_provider:
                chunk_texts = [chunk.content for chunk in ingestion_result.chunks]
                self.embedding_provider.fit_tfidf(chunk_texts)  # Update TF-IDF model

            # Step 3: Comprehend chunks (strategic or original approach)
            if self.use_strategic and self.strategic_engine:
                # STRATEGIC: Process entire document at once with intelligent routing
                stats = self.strategic_engine.process_and_store(
                    chunks=ingestion_result.chunks,
                    document_id=ingestion_result.document_id,
                    document_metadata=ingestion_result.metadata.to_dict()
                )
                total_concepts = stats.total_concepts
                logger.info(f"Strategic processing: {stats.llm_processed} LLM calls, "
                           f"{stats.code_extracted} code-extracted, {stats.skipped} skipped "
                           f"(saved {stats.llm_calls_saved} LLM calls)")
            else:
                # ORIGINAL: Process each chunk individually with 2 LLM calls per chunk
                total_concepts = 0
                for chunk in ingestion_result.chunks:
                    try:
                        # Comprehend chunk
                        comprehension_result = self.comprehension_engine.comprehend_chunk(
                            chunk=chunk,
                            document_metadata=ingestion_result.metadata.to_dict()
                        )

                        # Generate embedding for chunk if available
                        embedding = None
                        if self.embedding_provider:
                            embed_result = self.embedding_provider.embed(chunk.content)
                            embedding = embed_result.embedding

                        # Store in knowledge base
                        concepts_stored = self.comprehension_engine.store_comprehension_result(
                            result=comprehension_result,
                            document_metadata=ingestion_result.metadata.to_dict(),
                            embedding=embedding
                        )

                        total_concepts += concepts_stored

                    except Exception as e:
                        logger.error(f"Failed to comprehend chunk {chunk.chunk_index}: {e}")
                        continue

            # Step 4: Build knowledge graph for this document
            graph_stats = self.graph_builder.build_graph_for_document(ingestion_result.document_id)

            # Update document status
            self._update_document_status(
                doc_id=ingestion_result.document_id,
                status='completed',
                chunk_count=len(ingestion_result.chunks),
                concept_count=total_concepts
            )

            # Update directory index
            self._update_directory_index(
                file_path=file_path,
                doc_id=ingestion_result.document_id,
                file_hash=ingestion_result.metadata.file_hash,
                status='completed',
                entry_count=total_concepts
            )

            duration = time.time() - start_time
            logger.info(f"Document processed: {ingestion_result.metadata.file_name} - "
                       f"{total_concepts} concepts, {graph_stats.get('relationships_created', 0)} relationships, "
                       f"{duration:.1f}s")

        except Exception as e:
            logger.error(f"Document processing failed: {file_path} - {e}")
            raise

    def _store_document_metadata(self, ingestion_result):
        """Store document metadata in document_sources table."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0)

            metadata = ingestion_result.metadata

            conn.execute("""
                INSERT OR REPLACE INTO document_sources
                (doc_id, file_path, file_name, file_type, file_size, file_hash,
                 page_count, title, author, created_date, modified_date, encoding,
                 ingestion_status, ingestion_started, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ingestion_result.document_id,
                metadata.file_path,
                metadata.file_name,
                metadata.file_type.value,
                metadata.file_size,
                metadata.file_hash,
                metadata.page_count,
                metadata.title,
                metadata.author,
                metadata.created_date,
                metadata.modified_date,
                metadata.encoding,
                'processing',
                time.time(),
                len(ingestion_result.chunks)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")

    def _update_document_status(self, doc_id: str, status: str, chunk_count: int, concept_count: int):
        """Update document processing status."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0)

            conn.execute("""
                UPDATE document_sources
                SET ingestion_status = ?,
                    ingestion_completed = ?,
                    chunk_count = ?,
                    concept_count = ?
                WHERE doc_id = ?
            """, (status, time.time(), chunk_count, concept_count, doc_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update document status: {e}")

    def _update_directory_index(self, file_path: str, doc_id: str, file_hash: str,
                                 status: str, entry_count: int):
        """
        Update directory index file with document processing info.

        Args:
            file_path: Absolute path to the document
            doc_id: Document ID
            file_hash: File hash
            status: Processing status
            entry_count: Number of entries created
        """
        try:
            from src.knowledge.directory_index import DirectoryIndex
            import sqlite3

            # Get directory path
            file_path_obj = Path(file_path)
            directory_path = file_path_obj.parent
            file_name = file_path_obj.name

            # Get index
            index = DirectoryIndex(str(directory_path))

            # Get entry IDs from database
            entry_ids = []
            try:
                conn = sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0)
                cursor = conn.execute("""
                    SELECT knowledge_id FROM knowledge_entries
                    WHERE source_doc_id = ?
                """, (doc_id,))
                entry_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
            except Exception as e:
                logger.warning(f"Could not fetch entry IDs: {e}")

            # Update index
            index.add_document(
                file_name=file_name,
                doc_id=doc_id,
                file_hash=file_hash,
                status=status,
                entry_count=entry_count,
                entry_ids=entry_ids
            )

            logger.debug(f"Updated directory index for: {file_name}")

        except Exception as e:
            logger.warning(f"Failed to update directory index: {e}")
            # Non-fatal - don't let this stop document processing

    def trigger_refinement(self):
        """Manually trigger a refinement cycle."""
        logger.info("Manual refinement trigger")
        try:
            graph_stats = self.graph_builder.build_global_graph()
            self.last_refinement = time.time()
            return graph_stats
        except Exception as e:
            logger.error(f"Manual refinement failed: {e}")
            return {'error': str(e)}

    def process_directory_now(self, directory_path: str) -> Dict[str, Any]:
        """
        Manually trigger processing of a directory.

        Args:
            directory_path: Path to directory to process

        Returns:
            Dict with statistics
        """
        logger.info(f"Manual directory processing: {directory_path}")

        path = Path(directory_path)
        if not path.exists():
            return {'error': f'Directory does not exist: {directory_path}'}

        # Queue all documents
        patterns = ['*.pdf', '*.txt', '*.md', '*.py', '*.js', '*.java']
        queued_count = 0
        excluded_count = 0

        # Collect all files first
        files_to_queue = []
        for pattern in patterns:
            for file_path in path.rglob(pattern):
                # Skip excluded paths
                if self._should_exclude_path(file_path):
                    excluded_count += 1
                    continue

                files_to_queue.append(str(file_path))

        # Batch create database entries (single transaction)
        if files_to_queue:
            self._create_pending_entries_batch(files_to_queue)

        # Add to in-memory queue
        for file_path_str in files_to_queue:
            self.document_queue.add(file_path_str)
            queued_count += 1

        return {
            'queued': queued_count,
            'excluded': excluded_count,
            'directory': directory_path
        }

    def add_watch_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Add a directory to the persistent watch list.

        Args:
            directory_path: Path to directory to watch

        Returns:
            Dict with success status
        """
        logger.info(f"Adding watch directory: {directory_path}")

        # Validate directory exists
        if not Path(directory_path).is_dir():
            return {
                'success': False,
                'error': 'Directory does not exist'
            }

        # Check if already in watch list
        if directory_path in self.config.watch_directories:
            return {
                'success': False,
                'error': 'Directory already in watch list'
            }

        # Add to config
        self.config.watch_directories.append(directory_path)

        # If file watching is active, restart it to update observer
        if self.config.enable_file_watching and self.observer:
            self._restart_file_watching()

        logger.info(f"Successfully added directory: {directory_path}")
        return {
            'success': True,
            'added': directory_path
        }

    def remove_watch_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Remove a directory from the watch list.

        Args:
            directory_path: Path to directory to remove

        Returns:
            Dict with success status
        """
        logger.info(f"Removing watch directory: {directory_path}")

        if directory_path not in self.config.watch_directories:
            return {
                'success': False,
                'error': 'Directory not in watch list'
            }

        # Remove from config
        self.config.watch_directories.remove(directory_path)

        # If file watching is active, restart it to update observer
        if self.config.enable_file_watching and self.observer:
            self._restart_file_watching()

        logger.info(f"Successfully removed directory: {directory_path}")
        return {
            'success': True,
            'removed': directory_path
        }

    def reprocess_document(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Re-process an existing document (detects changes by hash).

        Args:
            file_path: Path to document
            force: If True, re-process even if hash unchanged

        Returns:
            Dict with processing results
        """
        try:
            import hashlib

            # Normalize path
            file_path = str(Path(file_path).resolve())

            # Check if file exists
            if not Path(file_path).exists():
                return {
                    'status': 'error',
                    'error': 'File does not exist'
                }

            # Calculate current file hash
            with open(file_path, 'rb') as f:
                current_hash = hashlib.md5(f.read()).hexdigest()

            # Check if document exists in database
            with sqlite3.connect(self.knowledge_store.storage_path, timeout=30.0) as conn:
                cursor = conn.execute("""
                    SELECT doc_id, file_hash, ingestion_status
                    FROM document_sources
                    WHERE file_path = ?
                """, (file_path,))
                row = cursor.fetchone()

                if not row:
                    # New document - process normally
                    logger.info(f"Document not in database, adding to queue: {file_path}")
                    self.document_queue.add(file_path)
                    return {
                        'status': 'queued',
                        'reason': 'new_document'
                    }

                doc_id, old_hash, status = row

                # Check if file changed
                if not force and old_hash == current_hash:
                    logger.info(f"Document unchanged (hash match), skipping: {file_path}")
                    return {
                        'status': 'skipped',
                        'reason': 'no_changes',
                        'doc_id': doc_id
                    }

                # Delete old entries
                logger.info(f"Re-processing document (hash changed or forced): {file_path}")
                cursor.execute("""
                    DELETE FROM knowledge_entries
                    WHERE source_doc_id = ?
                """, (doc_id,))

                # Delete from tags table
                cursor.execute("""
                    DELETE FROM knowledge_tags
                    WHERE knowledge_id IN (
                        SELECT knowledge_id FROM knowledge_entries
                        WHERE source_doc_id = ?
                    )
                """, (doc_id,))

                # Reset document status
                cursor.execute("""
                    UPDATE document_sources
                    SET ingestion_status = 'pending',
                        file_hash = ?,
                        chunk_count = 0,
                        concept_count = 0
                    WHERE doc_id = ?
                """, (current_hash, doc_id))

                conn.commit()

            # Queue for re-processing
            self.document_queue.add(file_path)

            return {
                'status': 'queued',
                'reason': 'file_changed' if old_hash != current_hash else 'forced',
                'doc_id': doc_id,
                'old_hash': old_hash,
                'new_hash': current_hash
            }

        except Exception as e:
            logger.error(f"Error re-processing document: {e}")
            return {'status': 'error', 'error': str(e)}

    def _restart_file_watching(self):
        """Restart file watching with updated directory list."""
        logger.info("Restarting file watching...")

        try:
            # Stop current observer
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
                self.observer = None

            # Start new observer with updated directories
            self._start_file_watching()

            logger.info("File watching restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart file watching: {e}")
