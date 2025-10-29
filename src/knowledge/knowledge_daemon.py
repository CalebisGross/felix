"""
Knowledge Daemon for Felix Knowledge Brain

Autonomous background processing system with three concurrent modes:
- Mode A: Initial Batch Processing - Process existing documents in directories
- Mode B: Continuous Refinement - Periodically re-analyze knowledge for new connections
- Mode C: File System Watching - Monitor directories for new documents

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

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for knowledge daemon."""
    watch_directories: List[str]
    enable_batch_processing: bool = True
    enable_refinement: bool = True
    enable_file_watching: bool = True
    refinement_interval: int = 3600  # seconds (1 hour)
    processing_threads: int = 2
    max_memory_mb: int = 512
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class DaemonStatus:
    """Current status of daemon."""
    running: bool
    batch_processor_active: bool
    refiner_active: bool
    file_watcher_active: bool
    documents_pending: int
    documents_processed: int
    documents_failed: int
    last_refinement: Optional[float]
    last_activity: Optional[float]
    uptime_seconds: float


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

        def __init__(self, document_queue: DocumentQueue, file_patterns: List[str]):
            super().__init__()
            self.document_queue = document_queue
            self.file_patterns = file_patterns

        def _should_process(self, file_path: str) -> bool:
            """Check if file should be processed."""
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
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.progress_callback = progress_callback

        # Processing components
        self.document_reader = DocumentReader(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.comprehension_engine = KnowledgeComprehensionEngine(
            knowledge_store=knowledge_store,
            llm_client=llm_client
        )
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
        self.last_activity = None

        # File watching
        self.observer = None

    def start(self):
        """Start all daemon modes."""
        if self.running:
            logger.warning("Daemon already running")
            return

        logger.info("Starting Knowledge Daemon...")
        self.running = True
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

        logger.info("Knowledge Daemon fully operational")

    def stop(self):
        """Stop all daemon modes gracefully."""
        if not self.running:
            return

        logger.info("Stopping Knowledge Daemon...")
        self.running = False

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

        return DaemonStatus(
            running=self.running,
            batch_processor_active=self.config.enable_batch_processing,
            refiner_active=self.config.enable_refinement,
            file_watcher_active=self.config.enable_file_watching and self.observer is not None,
            documents_pending=queue_stats['pending'],
            documents_processed=queue_stats['completed'],
            documents_failed=queue_stats['failed'],
            last_refinement=self.last_refinement,
            last_activity=self.last_activity,
            uptime_seconds=uptime
        )

    def _get_completed_documents(self) -> set:
        """
        Get set of file paths that have been successfully processed.

        Returns:
            Set of absolute file paths for documents with status='completed'
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path)
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
                    file_path_str = str(file_path.absolute())
                    # Skip if already processed
                    if file_path_str not in already_processed:
                        self.document_queue.add(file_path_str)

        logger.info(f"Batch processing: {self.document_queue.get_stats()['pending']} new documents queued "
                   f"({len(already_processed)} already processed)")

        # Process queue
        while self.running:
            file_path = self.document_queue.get(timeout=0.5)  # Shorter timeout for responsive shutdown

            if file_path is None:
                # Queue empty - check if we should stop before sleeping
                if not self.running:
                    break
                time.sleep(0.5)  # Shorter sleep for responsive shutdown
                continue

            # Check stop flag before processing
            if not self.running:
                break

            try:
                self._process_document(file_path)
                self.document_queue.mark_completed(file_path)
                self.last_activity = time.time()

                # Report progress
                if self.progress_callback:
                    stats = self.document_queue.get_stats()
                    self.progress_callback('batch_progress', stats)

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

        while self.running:
            # Wait for refinement interval with interruptible sleep
            # Sleep in 1-second increments to check self.running frequently
            sleep_remaining = self.config.refinement_interval
            while sleep_remaining > 0 and self.running:
                sleep_time = min(1.0, sleep_remaining)
                time.sleep(sleep_time)
                sleep_remaining -= sleep_time

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

    def _start_file_watching(self):
        """
        Mode C: Watch file system for new/modified documents.
        """
        try:
            self.observer = Observer()

            # Set up handlers for each watch directory
            file_patterns = ['*.pdf', '*.txt', '*.md', '*.py', '*.js', '*.java']
            handler = DocumentFileHandler(self.document_queue, file_patterns)

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

            # Step 3: Comprehend each chunk
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
            conn = sqlite3.connect(self.knowledge_store.storage_path)

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
            conn = sqlite3.connect(self.knowledge_store.storage_path)

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

        for pattern in patterns:
            for file_path in path.rglob(pattern):
                self.document_queue.add(str(file_path))
                queued_count += 1

        return {
            'queued': queued_count,
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
