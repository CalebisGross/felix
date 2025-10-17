"""
Memory and Context Persistence system for the Felix Framework.

This module provides persistent memory capabilities including:
- Shared knowledge base for cross-run learning
- Task memory for pattern recognition
- Context compression for efficient information management

Integration with Central Post enables learning and memory across
multiple helix processing runs.
"""

from .knowledge_store import KnowledgeStore, KnowledgeEntry, KnowledgeQuery
from .task_memory import TaskMemory, TaskPattern, TaskOutcome
from .context_compression import ContextCompressor, CompressionStrategy, CompressedContext

__all__ = [
    "KnowledgeStore", "KnowledgeEntry", "KnowledgeQuery",
    "TaskMemory", "TaskPattern", "TaskOutcome", 
    "ContextCompressor", "CompressionStrategy", "CompressedContext"
]
