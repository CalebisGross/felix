"""
Document Tool for managing knowledge base documents in conversational CLI.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_tool import BaseTool, ToolResult


class DocumentTool(BaseTool):
    """Tool for document ingestion and management."""

    @property
    def name(self) -> str:
        return "document"

    @property
    def description(self) -> str:
        return "Manage documents in the knowledge base"

    @property
    def usage(self) -> str:
        return "/document [ingest|list|show|delete] <path/id>"

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Execute document command.

        Commands:
            /document ingest <file_path>     - Ingest a document
            /document list                   - List all documents
            /document show <doc_id>          - Show document details
            /document delete <doc_id>        - Delete a document
        """
        if not args:
            return self.format_error("Usage: /document ingest <file_path>")

        command = args[0]

        if command == "ingest":
            return self._ingest_document(args[1:], kwargs)
        elif command == "list":
            return self._list_documents(kwargs)
        elif command == "show":
            return self._show_document(args[1:], kwargs)
        elif command == "delete":
            return self._delete_document(args[1:], kwargs)
        else:
            # If not a command, assume it's a file path for ingestion
            return self._ingest_document(args, kwargs)

    def _ingest_document(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Ingest a document into the knowledge base."""
        if not args:
            return self.format_error("Usage: /document ingest <file_path>")

        file_path = " ".join(args)  # Handle paths with spaces

        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            return self.format_error(f"File not found: {file_path}")

        if not path.is_file():
            return self.format_error(f"Not a file: {file_path}")

        try:
            # Check if knowledge brain is enabled
            felix_system = self.felix_context.get('felix_system')
            if not felix_system:
                return self.format_error("Felix system not initialized")

            if not felix_system.config.enable_knowledge_brain:
                return self.format_error(
                    "Knowledge brain is not enabled. "
                    "Enable it in configuration to use document ingestion."
                )

            # Import knowledge brain components
            from src.knowledge.document_reader import DocumentReader
            from src.knowledge.comprehension import KnowledgeComprehensionEngine
            from src.memory.knowledge_store import KnowledgeStore

            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                knowledge_store = KnowledgeStore()

            # Read document
            reader = DocumentReader()
            doc_chunks = reader.read_document(str(path.absolute()))

            if not doc_chunks:
                return self.format_error(f"Failed to read document or document is empty")

            # Register document in knowledge store
            doc_id = knowledge_store.add_document_source(
                file_path=str(path.absolute()),
                doc_type=path.suffix.lstrip('.'),
                processing_status='processing'
            )

            # Process with comprehension engine
            comprehension_engine = KnowledgeComprehensionEngine(
                knowledge_store=knowledge_store,
                llm_client=felix_system.llm_client if hasattr(felix_system, 'llm_client') else None
            )

            # Process chunks
            knowledge_entries = comprehension_engine.process_document(
                document_id=doc_id,
                chunks=doc_chunks,
                metadata={'source': str(path.absolute())}
            )

            # Update document status
            knowledge_store.update_document_status(doc_id, 'completed')

            # Format output
            output_lines = [
                f"Document ingested successfully",
                "",
                f"File: {path.name}",
                f"Document ID: {doc_id}",
                f"Chunks processed: {len(doc_chunks)}",
                f"Knowledge entries created: {len(knowledge_entries)}",
                "",
                "Use '/knowledge search <query>' to query the new knowledge"
            ]

            return self.format_success("\n".join(output_lines))

        except ImportError as e:
            return self.format_error(
                f"Knowledge brain components not available. "
                f"Make sure all dependencies are installed: {e}"
            )
        except Exception as e:
            return self.format_error(f"Failed to ingest document: {str(e)}")

    def _list_documents(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List all documents in the knowledge base."""
        try:
            from src.memory.knowledge_store import KnowledgeStore

            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                knowledge_store = KnowledgeStore()

            # Get all documents
            documents = knowledge_store.get_all_document_sources()

            if not documents:
                return self.format_success("No documents in knowledge base")

            # Format output
            output_lines = ["Documents in Knowledge Base:", ""]

            for doc in documents:
                doc_id = doc.get('doc_id', 'unknown')
                file_path = doc.get('file_path', 'unknown')
                doc_type = doc.get('doc_type', 'unknown')
                status = doc.get('processing_status', 'unknown')
                ingested_at = doc.get('ingested_at', 'unknown')

                # Get filename
                filename = Path(file_path).name if file_path != 'unknown' else 'unknown'

                output_lines.append(f"[{doc_id}] {filename}")
                output_lines.append(f"  Type: {doc_type} | Status: {status}")
                output_lines.append(f"  Path: {file_path}")
                output_lines.append(f"  Ingested: {ingested_at}")
                output_lines.append("")

            output_lines.append(f"Total: {len(documents)} documents")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to list documents: {str(e)}")

    def _show_document(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Show details of a specific document."""
        if not args:
            return self.format_error("Usage: /document show <doc_id>")

        doc_id = args[0]

        try:
            from src.memory.knowledge_store import KnowledgeStore

            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                knowledge_store = KnowledgeStore()

            # Get document
            document = knowledge_store.get_document_source(doc_id)

            if not document:
                return self.format_error(f"Document not found: {doc_id}")

            # Get knowledge entries from this document
            entries = knowledge_store.get_knowledge_entries_by_document(doc_id)

            # Format output
            output_lines = [
                f"Document {doc_id}",
                "=" * 60,
                "",
                f"File: {document.get('file_path', 'unknown')}",
                f"Type: {document.get('doc_type', 'unknown')}",
                f"Status: {document.get('processing_status', 'unknown')}",
                f"Ingested: {document.get('ingested_at', 'unknown')}",
                f"Processed: {document.get('processed_at', 'unknown')}",
                ""
            ]

            if entries:
                output_lines.append(f"Knowledge Entries: {len(entries)}")
                output_lines.append("")
                output_lines.append("Sample entries:")
                for entry in entries[:5]:  # Show first 5
                    content = entry.get('content', {})
                    concept = content.get('concept', 'Unknown')
                    domain = entry.get('domain', 'general')
                    output_lines.append(f"  - [{domain}] {concept}")

                if len(entries) > 5:
                    output_lines.append(f"  ... and {len(entries) - 5} more")
            else:
                output_lines.append("No knowledge entries extracted")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to show document: {str(e)}")

    def _delete_document(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Delete a document from the knowledge base."""
        if not args:
            return self.format_error("Usage: /document delete <doc_id>")

        doc_id = args[0]

        try:
            from src.memory.knowledge_store import KnowledgeStore

            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                knowledge_store = KnowledgeStore()

            # Check if document exists
            document = knowledge_store.get_document_source(doc_id)
            if not document:
                return self.format_error(f"Document not found: {doc_id}")

            # Get count of knowledge entries before deletion
            entries = knowledge_store.get_knowledge_entries_by_document(doc_id)
            entry_count = len(entries)

            # Delete document (should cascade to knowledge entries)
            knowledge_store.delete_document_source(doc_id)

            output_lines = [
                f"Document {doc_id} deleted successfully",
                "",
                f"File: {document.get('file_path', 'unknown')}",
                f"Knowledge entries removed: {entry_count}"
            ]

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to delete document: {str(e)}")
