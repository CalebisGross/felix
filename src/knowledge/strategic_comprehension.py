"""
Strategic Comprehension Engine for Felix Knowledge Brain

Orchestrates efficient document processing by:
1. Building document outline (no LLM) to analyze structure
2. Using basic extraction for simple chunks (no LLM)
3. Using unified LLM call (1 call instead of 2) for complex chunks
4. Batching related chunks for context

This reduces LLM calls by 50-70% while maintaining extraction quality.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.agents.standalone_agent import StandaloneAgent
from src.memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel

from .document_ingest import DocumentChunk
from .document_outline import DocumentOutlineBuilder, DocumentOutline, ChunkAnalysis, ChunkComplexity, ContentType
from .basic_extraction import BasicExtractionEngine, extract_without_llm
from .comprehension import (
    ComprehensionResult, ConceptExtraction, EntityExtraction,
    clean_concept_name, validate_concept
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics from document processing."""
    total_chunks: int
    llm_processed: int
    code_extracted: int
    skipped: int
    total_concepts: int
    total_relationships: int
    llm_calls_saved: int
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_chunks': self.total_chunks,
            'llm_processed': self.llm_processed,
            'code_extracted': self.code_extracted,
            'skipped': self.skipped,
            'total_concepts': self.total_concepts,
            'total_relationships': self.total_relationships,
            'llm_calls_saved': self.llm_calls_saved,
            'processing_time': self.processing_time,
            'llm_call_reduction': f"{(self.llm_calls_saved / max(1, self.total_chunks * 2)) * 100:.1f}%"
        }


class StrategicComprehensionEngine:
    """
    Strategic document comprehension that minimizes LLM calls.

    Processing flow:
    1. Build outline (code-based) - analyze all chunks
    2. For simple chunks: use BasicExtractionEngine (no LLM)
    3. For complex chunks: use unified LLM call (1 call, not 2)
    4. Store results with embeddings

    Compared to original approach (2 LLM calls per chunk):
    - 50-70% fewer LLM calls
    - Faster processing for simple content
    - Same quality for complex content
    """

    # Unified prompt template for single LLM call
    UNIFIED_PROMPT_TEMPLATE = """You are a knowledge extraction expert. Analyze this document excerpt and extract structured knowledge.

DOCUMENT: {document_title}
DOMAIN: {domain}
SECTION: {section_title}

CONTENT:
{content}

---

INSTRUCTIONS:
1. First, provide a 2-3 sentence SUMMARY of the key information
2. Extract 3-10 KEY CONCEPTS with definitions and relationships
3. Identify any ENTITIES (people, organizations, technologies)
4. List 3-5 KEY POINTS or insights

FORMAT YOUR RESPONSE EXACTLY AS:

SUMMARY: [Your 2-3 sentence summary]

CONCEPTS:
[Concept Name] | [Clear definition in one sentence] | [Importance: 0.0-1.0] | [Confidence: 0.0-1.0] | [Related: concept1, concept2]
[Concept Name] | [Definition] | [Importance] | [Confidence] | [Related: ...]
...

ENTITIES:
[Entity Name] | [Type: person/org/tech/concept] | [Brief description]
...

KEY_POINTS:
- [Key insight or takeaway 1]
- [Key insight or takeaway 2]
...

QUALITY: [Self-assessment 0.0-1.0]

IMPORTANT:
- Keep concept names short (2-5 words)
- Definitions should be complete sentences
- Only extract concepts that are actually defined or explained
- Mark importance based on how central the concept is
- Mark confidence based on how clearly it's defined
"""

    def __init__(self,
                 knowledge_store: KnowledgeStore,
                 llm_client,
                 embedding_provider=None,
                 min_quality_threshold: float = 0.5,
                 max_concepts_per_chunk: int = 10):
        """
        Initialize strategic comprehension engine.

        Args:
            knowledge_store: For storing extracted knowledge
            llm_client: LLM client for complex analysis
            embedding_provider: For generating embeddings
            min_quality_threshold: Minimum quality to accept
            max_concepts_per_chunk: Maximum concepts per chunk
        """
        self.knowledge_store = knowledge_store
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.min_quality_threshold = min_quality_threshold
        self.max_concepts_per_chunk = max_concepts_per_chunk

        # Initialize sub-components
        self.outline_builder = DocumentOutlineBuilder()
        self.basic_extractor = BasicExtractionEngine()

        # Lazy-initialized LLM agent
        self._llm_agent = None

    @property
    def llm_agent(self) -> StandaloneAgent:
        """Lazy initialization of LLM agent."""
        if self._llm_agent is None:
            self._llm_agent = StandaloneAgent(
                agent_id="strategic_comprehension",
                mode="analysis",  # Analysis mode for extraction
                llm_client=self.llm_client,
                temperature=0.4,  # Balanced for extraction
                max_tokens=2000   # Room for comprehensive response
            )
        return self._llm_agent

    def process_document(self,
                         chunks: List[DocumentChunk],
                         document_id: str,
                         document_metadata: Dict[str, Any]) -> Tuple[List[ComprehensionResult], ProcessingStats]:
        """
        Process an entire document using strategic approach.

        Args:
            chunks: All document chunks
            document_id: Document identifier
            document_metadata: Metadata about the document

        Returns:
            Tuple of (list of ComprehensionResults, ProcessingStats)
        """
        start_time = time.time()

        if not chunks:
            return [], ProcessingStats(0, 0, 0, 0, 0, 0, 0, 0.0)

        # Step 1: Build document outline (no LLM)
        logger.info(f"Building outline for {document_metadata.get('file_name', 'unknown')}...")
        outline = self.outline_builder.build_outline(
            chunks=chunks,
            document_id=document_id,
            file_name=document_metadata.get('file_name', 'unknown')
        )

        # Log outline stats
        stats = outline.summary_stats
        logger.info(
            f"Outline complete: {stats.get('llm_required_chunks', 0)} LLM chunks, "
            f"{stats.get('skippable_chunks', 0)} skippable ({stats.get('estimated_llm_savings', '0%')} savings)"
        )

        # Update document metadata with detected domain
        if outline.domain_confidence > 0.5:
            document_metadata['domain'] = outline.detected_domain
            logger.info(f"Auto-detected domain: {outline.detected_domain} ({outline.domain_confidence:.2f})")

        # Step 2: Process chunks based on outline recommendations
        results = []
        llm_processed = 0
        code_extracted = 0
        skipped = 0
        total_concepts = 0
        total_relationships = 0

        for chunk, analysis in zip(chunks, outline.chunk_analyses):
            if analysis.skip_llm:
                # Use basic extraction (no LLM)
                if analysis.complexity == ChunkComplexity.TRIVIAL:
                    # Skip entirely
                    skipped += 1
                    logger.debug(f"Skipping trivial chunk {chunk.chunk_index}")
                    continue
                else:
                    # Code-based extraction
                    result = self._process_with_basic_extraction(chunk, analysis, document_metadata)
                    code_extracted += 1
            else:
                # Use unified LLM call
                result = self._process_with_llm(chunk, analysis, document_metadata)
                llm_processed += 1

            if result and result.success:
                results.append(result)
                total_concepts += len(result.concepts)
                # Count relationships from related_concepts
                for concept in result.concepts:
                    total_relationships += len(concept.related_concepts)

        # Calculate LLM calls saved
        # Original: 2 calls per chunk (research + analysis)
        # Strategic: 1 call for complex, 0 for simple/trivial
        original_calls = len(chunks) * 2
        actual_calls = llm_processed
        calls_saved = original_calls - actual_calls

        processing_time = time.time() - start_time

        processing_stats = ProcessingStats(
            total_chunks=len(chunks),
            llm_processed=llm_processed,
            code_extracted=code_extracted,
            skipped=skipped,
            total_concepts=total_concepts,
            total_relationships=total_relationships,
            llm_calls_saved=calls_saved,
            processing_time=processing_time
        )

        logger.info(
            f"Document processing complete: {total_concepts} concepts, "
            f"{calls_saved} LLM calls saved, {processing_time:.2f}s"
        )

        return results, processing_stats

    def _process_with_basic_extraction(self,
                                       chunk: DocumentChunk,
                                       analysis: ChunkAnalysis,
                                       document_metadata: Dict[str, Any]) -> Optional[ComprehensionResult]:
        """Process chunk using basic extraction (no LLM)."""
        try:
            # Map content type to extraction hint
            content_type_map = {
                ContentType.CODE: 'code',
                ContentType.DEFINITION: 'definition',
                ContentType.LIST: 'list',
                ContentType.PROSE: 'prose',
                ContentType.MIXED: 'prose',
                ContentType.HEADER: 'prose',
                ContentType.TABLE: 'prose',
            }
            content_hint = content_type_map.get(analysis.content_type, 'prose')

            result = self.basic_extractor.extract_from_chunk(chunk, content_hint)

            # REMOVED: Fallback that added outline-detected concepts with placeholder definitions
            # This was creating garbage like "A concept mentioned in relation to..."
            # Only extract concepts that have actual definitions from the text

            logger.debug(f"Basic extraction: chunk {chunk.chunk_index} -> {len(result.concepts)} concepts")
            return result

        except Exception as e:
            logger.error(f"Basic extraction failed for chunk {chunk.chunk_index}: {e}")
            return None

    def _process_with_llm(self,
                          chunk: DocumentChunk,
                          analysis: ChunkAnalysis,
                          document_metadata: Dict[str, Any]) -> Optional[ComprehensionResult]:
        """Process chunk using unified LLM call (single call instead of 2)."""
        start_time = time.time()

        try:
            # Build unified prompt
            prompt = self.UNIFIED_PROMPT_TEMPLATE.format(
                document_title=document_metadata.get('title', 'Unknown'),
                domain=document_metadata.get('domain', 'general'),
                section_title=analysis.section_title or chunk.section_title or 'N/A',
                content=chunk.content[:3000]  # Limit content length
            )

            # Single LLM call
            response = self.llm_agent.process_task(prompt)

            # Parse unified response
            result = self._parse_unified_response(response, chunk)

            if result:
                result.processing_time = time.time() - start_time
                logger.debug(f"LLM extraction: chunk {chunk.chunk_index} -> {len(result.concepts)} concepts")

            return result

        except Exception as e:
            logger.error(f"LLM extraction failed for chunk {chunk.chunk_index}: {e}")
            return ComprehensionResult(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                summary="",
                concepts=[],
                entities=[],
                key_points=[],
                quality_score=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _parse_unified_response(self, response: str, chunk: DocumentChunk) -> Optional[ComprehensionResult]:
        """Parse the unified LLM response into ComprehensionResult."""
        try:
            summary = ""
            concepts = []
            entities = []
            key_points = []
            quality = 0.6

            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()

                # Section headers
                if line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
                    current_section = 'summary'
                elif line.startswith('CONCEPTS:'):
                    current_section = 'concepts'
                elif line.startswith('ENTITIES:'):
                    current_section = 'entities'
                elif line.startswith('KEY_POINTS:') or line.startswith('KEY POINTS:'):
                    current_section = 'key_points'
                elif line.startswith('QUALITY:'):
                    try:
                        quality = float(line.replace('QUALITY:', '').strip())
                    except ValueError:
                        quality = 0.6

                # Content parsing
                elif '|' in line and current_section == 'concepts':
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        concept_name = clean_concept_name(parts[0])
                        definition = parts[1]

                        if not validate_concept(concept_name, definition):
                            continue

                        # Parse importance and confidence
                        try:
                            importance = float(parts[2]) if parts[2].replace('.', '').replace('0', '').replace('1', '') == '' else 0.5
                        except:
                            importance = 0.5

                        try:
                            confidence = float(parts[3]) if len(parts) > 3 else 0.5
                        except:
                            confidence = 0.5

                        # Parse related concepts
                        related = []
                        if len(parts) > 4 and 'Related:' in parts[4]:
                            related_str = parts[4].replace('Related:', '').strip()
                            related = [clean_concept_name(r.strip()) for r in related_str.split(',')]
                            related = [r for r in related if r]

                        concepts.append(ConceptExtraction(
                            concept_name=concept_name,
                            definition=definition,
                            examples=[],
                            related_concepts=related,
                            importance=min(1.0, max(0.0, importance)),
                            confidence=min(1.0, max(0.0, confidence)),
                            source_chunk_id=chunk.chunk_id,
                            page_number=chunk.page_number,
                            section_title=chunk.section_title
                        ))

                elif '|' in line and current_section == 'entities':
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 3:
                        entities.append(EntityExtraction(
                            entity_name=parts[0],
                            entity_type=parts[1].replace('Type:', '').strip(),
                            description=parts[2],
                            confidence=0.7
                        ))

                elif line.startswith('-') and current_section == 'key_points':
                    point = line.lstrip('- ').strip()
                    if point:
                        key_points.append(point)

            return ComprehensionResult(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                summary=summary,
                concepts=concepts[:self.max_concepts_per_chunk],
                entities=entities,
                key_points=key_points[:5],
                quality_score=min(1.0, max(0.0, quality)),
                processing_time=0.0,  # Set by caller
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to parse unified response: {e}")
            return None

    def store_results(self,
                      results: List[ComprehensionResult],
                      document_metadata: Dict[str, Any]) -> int:
        """
        Store all comprehension results with embeddings.

        Args:
            results: List of ComprehensionResults to store
            document_metadata: Document metadata for context

        Returns:
            Number of knowledge entries stored
        """
        stored_count = 0
        domain = document_metadata.get('domain', 'general')

        for result in results:
            if not result.success:
                continue

            # Determine confidence level
            if result.quality_score >= 0.8:
                confidence_level = ConfidenceLevel.HIGH
            elif result.quality_score >= 0.5:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW

            # Store each concept
            for concept in result.concepts:
                try:
                    # Generate embedding
                    concept_embedding = None
                    if self.embedding_provider:
                        try:
                            text = f"{concept.concept_name}: {concept.definition}"
                            embed_result = self.embedding_provider.embed(text)
                            concept_embedding = embed_result.embedding if embed_result else None
                        except Exception as e:
                            logger.warning(f"Embedding failed for '{concept.concept_name}': {e}")

                    # Store in knowledge store
                    knowledge_id = self.knowledge_store.store_knowledge(
                        knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
                        content=concept.to_dict(),
                        confidence_level=confidence_level,
                        source_agent="strategic_comprehension",
                        domain=domain,
                        tags=[
                            document_metadata.get('file_type', 'document'),
                            'concept',
                            concept.concept_name.lower().replace(' ', '_')
                        ],
                        embedding=concept_embedding,
                        source_doc_id=result.document_id,
                        chunk_index=int(result.chunk_id.split('_')[-2]) if '_' in result.chunk_id else 0
                    )
                    stored_count += 1

                except Exception as e:
                    logger.error(f"Failed to store concept '{concept.concept_name}': {e}")

            # Store entities
            for entity in result.entities:
                try:
                    knowledge_id = self.knowledge_store.store_knowledge(
                        knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
                        content=entity.to_dict(),
                        confidence_level=confidence_level,
                        source_agent="strategic_comprehension",
                        domain=domain,
                        tags=[
                            document_metadata.get('file_type', 'document'),
                            'entity',
                            entity.entity_type
                        ],
                        source_doc_id=result.document_id
                    )
                    stored_count += 1

                except Exception as e:
                    logger.error(f"Failed to store entity '{entity.entity_name}': {e}")

        logger.info(f"Stored {stored_count} knowledge entries")
        return stored_count

    def process_and_store(self,
                          chunks: List[DocumentChunk],
                          document_id: str,
                          document_metadata: Dict[str, Any]) -> ProcessingStats:
        """
        Convenience method to process and store in one call.

        Args:
            chunks: Document chunks to process
            document_id: Document identifier
            document_metadata: Document metadata

        Returns:
            ProcessingStats with results
        """
        results, stats = self.process_document(chunks, document_id, document_metadata)

        if results:
            stored = self.store_results(results, document_metadata)
            logger.info(f"Processed {stats.total_chunks} chunks, stored {stored} entries")

        return stats
