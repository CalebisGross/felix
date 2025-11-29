"""
Directory Index Management

Manages .felix_index.json files in watched directories to track processed documents.
Enables directory-level operations like re-processing and bulk deletion.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DirectoryIndex:
    """
    Manages index files for watched directories.

    Index file structure:
    {
        "directory": "/absolute/path/to/directory",
        "created_at": "2024-01-01T00:00:00",
        "last_updated": "2024-01-01T12:00:00",
        "documents": {
            "file.txt": {
                "doc_id": "abc123...",
                "file_hash": "def456...",
                "processed_at": "2024-01-01T10:00:00",
                "status": "completed",
                "entry_count": 15,
                "entry_ids": ["entry1", "entry2", ...]
            }
        },
        "statistics": {
            "total_documents": 10,
            "completed": 8,
            "failed": 1,
            "pending": 1,
            "total_entries": 150
        }
    }
    """

    INDEX_FILENAME = ".felix_index.json"

    def __init__(self, directory_path: str):
        """
        Initialize directory index manager.

        Args:
            directory_path: Absolute path to watched directory
        """
        self.directory_path = Path(directory_path)
        self.index_path = self.directory_path / self.INDEX_FILENAME

    def exists(self) -> bool:
        """Check if index file exists."""
        return self.index_path.exists()

    def create(self) -> bool:
        """
        Create a new index file for this directory.

        Returns:
            True if created successfully, False otherwise
        """
        try:
            index_data = {
                "directory": str(self.directory_path.absolute()),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "documents": {},
                "statistics": {
                    "total_documents": 0,
                    "completed": 0,
                    "failed": 0,
                    "pending": 0,
                    "total_entries": 0
                }
            }

            with open(self.index_path, 'w') as f:
                json.dump(index_data, f, indent=2)

            logger.info(f"Created directory index: {self.index_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create directory index: {e}")
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load index file.

        Returns:
            Index data dict or None if error
        """
        try:
            if not self.exists():
                return None

            with open(self.index_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Failed to load directory index: {e}")
            return None

    def save(self, index_data: Dict[str, Any]) -> bool:
        """
        Save index data to file.

        Args:
            index_data: Index data to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            index_data["last_updated"] = datetime.now().isoformat()

            with open(self.index_path, 'w') as f:
                json.dump(index_data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save directory index: {e}")
            return False

    def add_document(self, file_name: str, doc_id: str, file_hash: str,
                     status: str = "pending", entry_count: int = 0,
                     entry_ids: List[str] = None) -> bool:
        """
        Add or update a document in the index.

        Args:
            file_name: Name of the file (relative to directory)
            doc_id: Document ID from database
            file_hash: MD5 hash of file
            status: Processing status (pending/processing/completed/failed)
            entry_count: Number of knowledge entries created
            entry_ids: List of knowledge entry IDs

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            index_data = self.load()
            if index_data is None:
                # Create new index if doesn't exist
                self.create()
                index_data = self.load()

            if index_data is None:
                return False

            # Add/update document
            index_data["documents"][file_name] = {
                "doc_id": doc_id,
                "file_hash": file_hash,
                "processed_at": datetime.now().isoformat(),
                "status": status,
                "entry_count": entry_count,
                "entry_ids": entry_ids or []
            }

            # Update statistics
            self._update_statistics(index_data)

            return self.save(index_data)

        except Exception as e:
            logger.error(f"Failed to add document to index: {e}")
            return False

    def remove_document(self, file_name: str) -> bool:
        """
        Remove a document from the index.

        Args:
            file_name: Name of the file to remove

        Returns:
            True if removed successfully, False otherwise
        """
        try:
            index_data = self.load()
            if index_data is None:
                return False

            if file_name in index_data["documents"]:
                del index_data["documents"][file_name]

                # Update statistics
                self._update_statistics(index_data)

                return self.save(index_data)

            return False

        except Exception as e:
            logger.error(f"Failed to remove document from index: {e}")
            return False

    def get_document(self, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Get document info from index.

        Args:
            file_name: Name of the file

        Returns:
            Document info dict or None if not found
        """
        try:
            index_data = self.load()
            if index_data is None:
                return None

            return index_data["documents"].get(file_name)

        except Exception as e:
            logger.error(f"Failed to get document from index: {e}")
            return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from index.

        Returns:
            List of document info dicts with file names
        """
        try:
            index_data = self.load()
            if index_data is None:
                return []

            documents = []
            for file_name, doc_info in index_data["documents"].items():
                doc_info_copy = doc_info.copy()
                doc_info_copy["file_name"] = file_name
                documents.append(doc_info_copy)

            return documents

        except Exception as e:
            logger.error(f"Failed to get all documents from index: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get directory statistics from index.

        Returns:
            Statistics dict
        """
        try:
            index_data = self.load()
            if index_data is None:
                return {
                    "total_documents": 0,
                    "completed": 0,
                    "failed": 0,
                    "pending": 0,
                    "total_entries": 0
                }

            return index_data.get("statistics", {})

        except Exception as e:
            logger.error(f"Failed to get statistics from index: {e}")
            return {}

    def _update_statistics(self, index_data: Dict[str, Any]):
        """
        Update statistics based on current documents.

        Args:
            index_data: Index data to update
        """
        documents = index_data.get("documents", {})

        stats = {
            "total_documents": len(documents),
            "completed": 0,
            "failed": 0,
            "pending": 0,
            "processing": 0,
            "total_entries": 0
        }

        for doc_info in documents.values():
            status = doc_info.get("status", "pending")
            stats[status] = stats.get(status, 0) + 1
            stats["total_entries"] += doc_info.get("entry_count", 0)

        index_data["statistics"] = stats

    def rebuild_from_database(self, knowledge_store) -> bool:
        """
        Rebuild index from database information.

        Useful for recovering from corrupted or missing index files.

        Args:
            knowledge_store: KnowledgeStore instance

        Returns:
            True if rebuilt successfully, False otherwise
        """
        try:
            import sqlite3

            # Create new index
            if not self.create():
                return False

            index_data = self.load()
            if index_data is None:
                return False

            # Query all documents from this directory
            with sqlite3.connect(knowledge_store.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT doc_id, file_path, file_name, file_hash,
                           ingestion_status, concept_count
                    FROM document_sources
                    WHERE file_path LIKE ?
                """, (f"{self.directory_path}%",))

                for row in cursor.fetchall():
                    doc_id, file_path, file_name, file_hash, status, concept_count = row

                    # Get entry IDs for this document
                    entry_cursor = conn.execute("""
                        SELECT knowledge_id FROM knowledge_entries
                        WHERE source_doc_id = ?
                    """, (doc_id,))
                    entry_ids = [r[0] for r in entry_cursor.fetchall()]

                    # Add to index
                    index_data["documents"][file_name] = {
                        "doc_id": doc_id,
                        "file_hash": file_hash,
                        "processed_at": datetime.now().isoformat(),
                        "status": status,
                        "entry_count": len(entry_ids),
                        "entry_ids": entry_ids
                    }

            # Update statistics
            self._update_statistics(index_data)

            return self.save(index_data)

        except Exception as e:
            logger.error(f"Failed to rebuild index from database: {e}")
            return False

    def cleanup_missing_files(self) -> int:
        """
        Remove entries for files that no longer exist.

        Returns:
            Number of entries removed
        """
        try:
            index_data = self.load()
            if index_data is None:
                return 0

            removed = 0
            documents_to_remove = []

            for file_name in index_data["documents"].keys():
                file_path = self.directory_path / file_name
                if not file_path.exists():
                    documents_to_remove.append(file_name)

            for file_name in documents_to_remove:
                del index_data["documents"][file_name]
                removed += 1

            if removed > 0:
                self._update_statistics(index_data)
                self.save(index_data)
                logger.info(f"Removed {removed} missing files from index")

            return removed

        except Exception as e:
            logger.error(f"Failed to cleanup missing files: {e}")
            return 0


class DirectoryIndexManager:
    """
    Manages multiple directory indexes.
    """

    @staticmethod
    def get_index(directory_path: str) -> DirectoryIndex:
        """
        Get index for a directory.

        Args:
            directory_path: Absolute path to directory

        Returns:
            DirectoryIndex instance
        """
        return DirectoryIndex(directory_path)

    @staticmethod
    def create_all_indexes(watch_directories: List[str]) -> Dict[str, bool]:
        """
        Create index files for all watched directories.

        Args:
            watch_directories: List of directory paths

        Returns:
            Dict mapping directory paths to success status
        """
        results = {}

        for directory_path in watch_directories:
            index = DirectoryIndex(directory_path)
            if not index.exists():
                results[directory_path] = index.create()
            else:
                results[directory_path] = True  # Already exists

        return results

    @staticmethod
    def rebuild_all_indexes(watch_directories: List[str],
                           knowledge_store) -> Dict[str, bool]:
        """
        Rebuild all directory indexes from database.

        Args:
            watch_directories: List of directory paths
            knowledge_store: KnowledgeStore instance

        Returns:
            Dict mapping directory paths to success status
        """
        results = {}

        for directory_path in watch_directories:
            index = DirectoryIndex(directory_path)
            results[directory_path] = index.rebuild_from_database(knowledge_store)

        return results

    @staticmethod
    def cleanup_all_indexes(watch_directories: List[str]) -> Dict[str, int]:
        """
        Clean up all directory indexes (remove missing files).

        Args:
            watch_directories: List of directory paths

        Returns:
            Dict mapping directory paths to number of entries removed
        """
        results = {}

        for directory_path in watch_directories:
            index = DirectoryIndex(directory_path)
            results[directory_path] = index.cleanup_missing_files()

        return results
