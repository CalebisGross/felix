"""
Document Ingestion Module for Felix Knowledge Brain

Handles reading, chunking, and metadata extraction from multiple document formats:
- PDF files (using PyPDF2)
- Text files (.txt)
- Markdown files (.md)
- Python/code files (.py, .js, .java, etc.)

Provides intelligent chunking with semantic boundaries and overlap handling.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    TEXT = "txt"
    MARKDOWN = "md"
    PYTHON = "py"
    JAVASCRIPT = "js"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    UNKNOWN = "unknown"


class ChunkingStrategy(Enum):
    """Strategies for chunking documents."""
    FIXED_SIZE = "fixed_size"  # Fixed token count
    PARAGRAPH = "paragraph"  # Split on paragraph boundaries
    SECTION = "section"  # Split on section headers
    SEMANTIC = "semantic"  # Smart splitting (paragraphs + headers)


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    file_path: str
    file_name: str
    file_type: DocumentType
    file_size: int
    file_hash: str
    page_count: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[float] = None
    modified_date: Optional[float] = None
    encoding: Optional[str] = "utf-8"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_type': self.file_type.value,
            'file_size': self.file_size,
            'file_hash': self.file_hash,
            'page_count': self.page_count,
            'title': self.title,
            'author': self.author,
            'created_date': self.created_date,
            'modified_date': self.modified_date,
            'encoding': self.encoding
        }


@dataclass
class DocumentChunk:
    """A single chunk of a document with metadata."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()

    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.document_id}_chunk_{self.chunk_index}_{content_hash}"


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'document_id': self.document_id,
            'metadata': self.metadata.to_dict(),
            'chunk_count': len(self.chunks),
            'success': self.success,
            'error_message': self.error_message,
            'processing_time': self.processing_time
        }


class DocumentReader:
    """
    Multi-format document reader with intelligent chunking.

    Supports PDF, TXT, MD, and code files with configurable chunking strategies.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC):
        """
        Initialize document reader.

        Args:
            chunk_size: Target characters per chunk
            chunk_overlap: Characters to overlap between chunks
            chunking_strategy: Strategy for chunking documents
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

        # Lazy import PyPDF2 to avoid dependency if not processing PDFs
        self.pypdf2 = None

    def ingest_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a document and return chunked results.

        Args:
            file_path: Path to document file

        Returns:
            IngestionResult with metadata and chunks
        """
        start_time = time.time()
        path = Path(file_path)

        try:
            # Validate file exists
            if not path.exists():
                return IngestionResult(
                    document_id="",
                    metadata=None,
                    chunks=[],
                    success=False,
                    error_message=f"File not found: {file_path}"
                )

            # Extract metadata
            metadata = self._extract_metadata(path)
            document_id = metadata.file_hash

            logger.info(f"Ingesting document: {path.name} ({metadata.file_type.value})")

            # Read document based on type
            if metadata.file_type == DocumentType.PDF:
                content, page_info = self._read_pdf(path)
                metadata.page_count = len(page_info)
            elif metadata.file_type in [DocumentType.TEXT, DocumentType.MARKDOWN]:
                content = self._read_text(path)
                page_info = None
            elif metadata.file_type in [DocumentType.PYTHON, DocumentType.JAVASCRIPT,
                                       DocumentType.JAVA, DocumentType.CPP, DocumentType.C]:
                content = self._read_code(path)
                page_info = None
            else:
                return IngestionResult(
                    document_id=document_id,
                    metadata=metadata,
                    chunks=[],
                    success=False,
                    error_message=f"Unsupported file type: {metadata.file_type.value}"
                )

            # Chunk the content
            chunks = self._chunk_content(
                content=content,
                document_id=document_id,
                page_info=page_info
            )

            processing_time = time.time() - start_time

            logger.info(f"Successfully ingested {path.name}: {len(chunks)} chunks, "
                       f"{processing_time:.2f}s")

            return IngestionResult(
                document_id=document_id,
                metadata=metadata,
                chunks=chunks,
                success=True,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to ingest {file_path}: {e}")

            return IngestionResult(
                document_id=metadata.file_hash if 'metadata' in locals() else "",
                metadata=metadata if 'metadata' in locals() else None,
                chunks=[],
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    def _extract_metadata(self, path: Path) -> DocumentMetadata:
        """Extract metadata from file."""
        stat = path.stat()

        # Determine file type
        ext = path.suffix.lower().lstrip('.')
        try:
            file_type = DocumentType(ext)
        except ValueError:
            file_type = DocumentType.UNKNOWN

        # Compute file hash
        with open(path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # Extract title from filename
        title = path.stem.replace('_', ' ').replace('-', ' ').title()

        return DocumentMetadata(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_type=file_type,
            file_size=stat.st_size,
            file_hash=file_hash,
            title=title,
            created_date=stat.st_ctime,
            modified_date=stat.st_mtime
        )

    def _read_pdf(self, path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Read PDF file and extract text with page information.

        Returns:
            Tuple of (full_text, page_info_list)
        """
        # Lazy import PyPDF2
        if self.pypdf2 is None:
            try:
                import PyPDF2
                self.pypdf2 = PyPDF2
            except ImportError:
                raise ImportError(
                    "PyPDF2 is required for PDF reading. Install with: pip install PyPDF2"
                )

        full_text = []
        page_info = []

        with open(path, 'rb') as f:
            pdf_reader = self.pypdf2.PdfReader(f)

            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    text = page.extract_text()
                    start_pos = len(''.join(full_text))
                    full_text.append(text)
                    end_pos = len(''.join(full_text))

                    page_info.append({
                        'page_number': page_num,
                        'start_position': start_pos,
                        'end_position': end_pos,
                        'text_length': len(text)
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")

        return '\n\n'.join(full_text), page_info

    def _read_text(self, path: Path) -> str:
        """Read text or markdown file."""
        encodings = ['utf-8', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Last resort: read as binary and decode with errors='replace'
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')

    def _read_code(self, path: Path) -> str:
        """Read code file (Python, JavaScript, etc.)."""
        return self._read_text(path)

    def _chunk_content(self,
                       content: str,
                       document_id: str,
                       page_info: Optional[List[Dict[str, Any]]] = None) -> List[DocumentChunk]:
        """
        Chunk content based on configured strategy.

        Args:
            content: Full document content
            document_id: Document identifier
            page_info: Optional page information for PDFs

        Returns:
            List of DocumentChunk objects
        """
        if self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(content, document_id, page_info)
        elif self.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraph(content, document_id, page_info)
        elif self.chunking_strategy == ChunkingStrategy.SECTION:
            return self._chunk_section(content, document_id, page_info)
        else:  # FIXED_SIZE
            return self._chunk_fixed_size(content, document_id, page_info)

    def _chunk_semantic(self,
                        content: str,
                        document_id: str,
                        page_info: Optional[List[Dict[str, Any]]]) -> List[DocumentChunk]:
        """
        Smart semantic chunking combining paragraphs and sections.

        Prioritizes natural boundaries (paragraphs, sections) while respecting chunk_size.
        """
        chunks = []

        # Split on section headers (markdown-style or numbered)
        section_pattern = r'\n(#{1,6}\s+.+|\d+\.\s+[A-Z].+)\n'
        sections = re.split(section_pattern, content)

        current_chunk = ""
        current_start = 0
        chunk_index = 0
        section_title = None

        for i, section in enumerate(sections):
            # Check if this is a section header
            is_header = i % 2 == 1 and i > 0

            if is_header:
                section_title = section.strip()
                continue

            # Split section into paragraphs
            paragraphs = section.split('\n\n')

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # Check if adding this paragraph would exceed chunk_size
                if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk = self._create_chunk(
                        content=current_chunk,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        start_pos=current_start,
                        section_title=section_title,
                        page_info=page_info
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk = overlap_text + paragraph
                    current_start += len(current_chunk) - len(overlap_text)
                    chunk_index += 1
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                        current_start = content.find(paragraph)

        # Save final chunk
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                document_id=document_id,
                chunk_index=chunk_index,
                start_pos=current_start,
                section_title=section_title,
                page_info=page_info
            )
            chunks.append(chunk)

        return chunks

    def _chunk_paragraph(self,
                         content: str,
                         document_id: str,
                         page_info: Optional[List[Dict[str, Any]]]) -> List[DocumentChunk]:
        """Chunk by paragraphs."""
        paragraphs = content.split('\n\n')
        chunks = []
        chunk_index = 0
        current_pos = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            chunk = self._create_chunk(
                content=paragraph,
                document_id=document_id,
                chunk_index=chunk_index,
                start_pos=current_pos,
                page_info=page_info
            )
            chunks.append(chunk)
            chunk_index += 1
            current_pos += len(paragraph) + 2  # +2 for \n\n

        return chunks

    def _chunk_section(self,
                       content: str,
                       document_id: str,
                       page_info: Optional[List[Dict[str, Any]]]) -> List[DocumentChunk]:
        """Chunk by sections (headers)."""
        # Split on markdown headers or numbered sections
        section_pattern = r'\n(#{1,6}\s+.+|\d+\.\s+[A-Z].+)\n'
        parts = re.split(section_pattern, content)

        chunks = []
        chunk_index = 0
        current_pos = 0

        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                section_title = parts[i + 1].strip()
                section_content = parts[i + 2] if i + 2 < len(parts) else ""
            else:
                section_title = None
                section_content = parts[i]

            section_content = section_content.strip()
            if not section_content:
                continue

            chunk = self._create_chunk(
                content=section_content,
                document_id=document_id,
                chunk_index=chunk_index,
                start_pos=current_pos,
                section_title=section_title,
                page_info=page_info
            )
            chunks.append(chunk)
            chunk_index += 1
            current_pos += len(section_content)

        return chunks

    def _chunk_fixed_size(self,
                          content: str,
                          document_id: str,
                          page_info: Optional[List[Dict[str, Any]]]) -> List[DocumentChunk]:
        """Chunk by fixed size with overlap."""
        chunks = []
        chunk_index = 0
        start = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]

            chunk = self._create_chunk(
                content=chunk_text,
                document_id=document_id,
                chunk_index=chunk_index,
                start_pos=start,
                page_info=page_info
            )
            chunks.append(chunk)

            chunk_index += 1
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks

    def _create_chunk(self,
                      content: str,
                      document_id: str,
                      chunk_index: int,
                      start_pos: int,
                      section_title: Optional[str] = None,
                      page_info: Optional[List[Dict[str, Any]]] = None) -> DocumentChunk:
        """Create a DocumentChunk with metadata."""
        end_pos = start_pos + len(content)

        # Determine page number if page_info available
        page_number = None
        if page_info:
            for page in page_info:
                if page['start_position'] <= start_pos < page['end_position']:
                    page_number = page['page_number']
                    break

        return DocumentChunk(
            chunk_id="",  # Will be generated in __post_init__
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            start_position=start_pos,
            end_position=end_pos,
            page_number=page_number,
            section_title=section_title
        )


class BatchDocumentProcessor:
    """
    Batch processor for ingesting multiple documents with progress tracking.
    """

    def __init__(self, reader: DocumentReader):
        """Initialize batch processor."""
        self.reader = reader
        self.total_documents = 0
        self.processed_documents = 0
        self.failed_documents = 0

    def process_directory(self,
                          directory_path: str,
                          recursive: bool = True,
                          file_patterns: Optional[List[str]] = None) -> List[IngestionResult]:
        """
        Process all documents in a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            file_patterns: Optional list of glob patterns (e.g., ['*.pdf', '*.txt'])

        Returns:
            List of IngestionResult objects
        """
        path = Path(directory_path)

        if not path.exists() or not path.is_dir():
            logger.error(f"Invalid directory: {directory_path}")
            return []

        # Default patterns if not specified
        if file_patterns is None:
            file_patterns = ['*.pdf', '*.txt', '*.md', '*.py', '*.js', '*.java']

        # Find all matching files
        files = []
        for pattern in file_patterns:
            if recursive:
                files.extend(path.rglob(pattern))
            else:
                files.extend(path.glob(pattern))

        self.total_documents = len(files)
        self.processed_documents = 0
        self.failed_documents = 0

        logger.info(f"Found {self.total_documents} documents to process")

        results = []
        for file_path in files:
            result = self.reader.ingest_document(str(file_path))
            results.append(result)

            self.processed_documents += 1
            if not result.success:
                self.failed_documents += 1

            # Log progress
            if self.processed_documents % 10 == 0:
                logger.info(f"Progress: {self.processed_documents}/{self.total_documents} "
                           f"({self.failed_documents} failed)")

        logger.info(f"Batch processing complete: {self.processed_documents} processed, "
                   f"{self.failed_documents} failed")

        return results

    def get_progress(self) -> Dict[str, int]:
        """Get current processing progress."""
        return {
            'total': self.total_documents,
            'processed': self.processed_documents,
            'failed': self.failed_documents,
            'remaining': self.total_documents - self.processed_documents
        }
