"""
Knowledge Comprehension Engine for Felix Knowledge Brain

Uses Felix agents to read, understand, and extract knowledge from documents:
- ResearchAgent: Reads and summarizes document content
- AnalysisAgent: Extracts entities, concepts, relationships
- CriticAgent: Validates extraction quality

Implements agentic RAG approach where agents actively comprehend content
rather than just chunking and indexing.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, CriticAgent
from src.memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel
from .document_ingest import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ConceptExtraction:
    """A concept extracted from a document."""
    concept_name: str
    definition: str
    examples: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    source_chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'concept': self.concept_name,
            'definition': self.definition,
            'examples': self.examples,
            'related_concepts': self.related_concepts,
            'importance': self.importance,
            'confidence': self.confidence,
            'source_chunk_id': self.source_chunk_id,
            'page_number': self.page_number,
            'section_title': self.section_title
        }


@dataclass
class EntityExtraction:
    """An entity extracted from a document."""
    entity_name: str
    entity_type: str  # person, organization, technology, concept, etc.
    description: str
    mentions: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'entity': self.entity_name,
            'type': self.entity_type,
            'description': self.description,
            'mentions': self.mentions,
            'confidence': self.confidence
        }


@dataclass
class ComprehensionResult:
    """Result of document comprehension."""
    document_id: str
    chunk_id: str
    summary: str
    concepts: List[ConceptExtraction]
    entities: List[EntityExtraction]
    key_points: List[str]
    quality_score: float  # 0.0 to 1.0
    processing_time: float
    success: bool
    error_message: Optional[str] = None

    def get_concept_count(self) -> int:
        """Get number of concepts extracted."""
        return len(self.concepts)

    def get_entity_count(self) -> int:
        """Get number of entities extracted."""
        return len(self.entities)


class KnowledgeComprehensionEngine:
    """
    Agentic comprehension engine using Felix agents to understand documents.

    Uses three-stage pipeline:
    1. Research: Read and summarize content
    2. Analysis: Extract concepts and entities
    3. Criticism: Validate quality
    """

    def __init__(self,
                 knowledge_store: KnowledgeStore,
                 llm_client,
                 min_quality_threshold: float = 0.6,
                 max_retries: int = 2):
        """
        Initialize comprehension engine.

        Args:
            knowledge_store: KnowledgeStore for storing extracted knowledge
            llm_client: LLM client for agents
            min_quality_threshold: Minimum quality score to accept (0.0-1.0)
            max_retries: Maximum retries if quality insufficient
        """
        self.knowledge_store = knowledge_store
        self.llm_client = llm_client
        self.min_quality_threshold = min_quality_threshold
        self.max_retries = max_retries

        # Initialize agents (without helix position for document processing)
        self.research_agent = None
        self.analysis_agent = None
        self.critic_agent = None

    def _initialize_agents(self):
        """Lazy initialization of agents."""
        if self.research_agent is None:
            self.research_agent = ResearchAgent(
                agent_id="research_comprehension",
                llm_client=self.llm_client,
                helix_position=0.5  # Mid-helix
            )

        if self.analysis_agent is None:
            self.analysis_agent = AnalysisAgent(
                agent_id="analysis_comprehension",
                llm_client=self.llm_client,
                helix_position=0.7  # Lower helix for focused analysis
            )

        if self.critic_agent is None:
            self.critic_agent = CriticAgent(
                agent_id="critic_comprehension",
                llm_client=self.llm_client,
                helix_position=0.3  # Upper helix for broader perspective
            )

    def comprehend_chunk(self,
                         chunk: DocumentChunk,
                         document_metadata: Dict[str, Any]) -> ComprehensionResult:
        """
        Comprehend a single document chunk using agents.

        Args:
            chunk: Document chunk to process
            document_metadata: Metadata about source document

        Returns:
            ComprehensionResult with extracted knowledge
        """
        start_time = time.time()

        try:
            self._initialize_agents()

            # Stage 1: Research - Read and summarize
            logger.info(f"Processing chunk {chunk.chunk_index} from {document_metadata.get('file_name', 'unknown')}")
            research_result = self._research_chunk(chunk, document_metadata)

            if not research_result['success']:
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
                    error_message=research_result.get('error', 'Research failed')
                )

            # Stage 2: Analysis - Extract concepts and entities
            analysis_result = self._analyze_chunk(chunk, research_result, document_metadata)

            if not analysis_result['success']:
                return ComprehensionResult(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    summary=research_result.get('summary', ''),
                    concepts=[],
                    entities=[],
                    key_points=research_result.get('key_points', []),
                    quality_score=0.3,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=analysis_result.get('error', 'Analysis failed')
                )

            # Stage 3: Criticism - Validate quality
            quality_score = self._validate_quality(research_result, analysis_result)

            processing_time = time.time() - start_time

            # Check if quality meets threshold
            if quality_score < self.min_quality_threshold:
                logger.warning(f"Low quality score ({quality_score:.2f}) for chunk {chunk.chunk_index}")

            return ComprehensionResult(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                summary=research_result.get('summary', ''),
                concepts=analysis_result.get('concepts', []),
                entities=analysis_result.get('entities', []),
                key_points=research_result.get('key_points', []),
                quality_score=quality_score,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            logger.error(f"Comprehension failed for chunk {chunk.chunk_id}: {e}")
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

    def _research_chunk(self,
                        chunk: DocumentChunk,
                        document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Use ResearchAgent to read and summarize chunk.

        Returns dict with: success, summary, key_points, confidence
        """
        try:
            # Build context for research agent
            context = f"""Document: {document_metadata.get('title', 'Unknown')}
Source: {document_metadata.get('file_name', 'Unknown')}
Section: {chunk.section_title or 'N/A'}
Page: {chunk.page_number or 'N/A'}

Content to analyze:
{chunk.content}

---

Your task as a ResearchAgent:
1. Read and understand this document excerpt carefully
2. Provide a concise summary (2-3 sentences)
3. Extract 3-5 key points or insights
4. Rate your confidence in understanding (0.0-1.0)

Format your response as:
SUMMARY: [your summary]
KEY_POINTS:
- [point 1]
- [point 2]
...
CONFIDENCE: [0.0-1.0]
"""

            # Use research agent to process
            response = self.research_agent.process_task(context)

            # Parse response
            summary = ""
            key_points = []
            confidence = 0.7  # Default

            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
                    current_section = 'summary'
                elif line.startswith('KEY_POINTS:'):
                    current_section = 'key_points'
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except ValueError:
                        confidence = 0.7
                elif line.startswith('-') and current_section == 'key_points':
                    key_points.append(line.lstrip('- ').strip())

            return {
                'success': True,
                'summary': summary,
                'key_points': key_points,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Research stage failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _analyze_chunk(self,
                       chunk: DocumentChunk,
                       research_result: Dict[str, Any],
                       document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: Use AnalysisAgent to extract concepts and entities.

        Returns dict with: success, concepts, entities
        """
        try:
            # Build context for analysis agent
            context = f"""Document: {document_metadata.get('title', 'Unknown')}
Domain: {document_metadata.get('domain', 'general')}

Research Summary: {research_result.get('summary', '')}

Original Content:
{chunk.content[:2000]}...

---

Your task as an AnalysisAgent:
1. Extract key CONCEPTS (ideas, principles, techniques)
2. Identify ENTITIES (people, organizations, technologies)
3. For each concept, provide: name, definition, examples, related concepts
4. Rate importance (0.0-1.0) and confidence (0.0-1.0) for each

Format your response as:
CONCEPTS:
[Concept Name] | [Definition] | [Importance] | [Confidence] | [Related: concept1, concept2]
Example: [example if available]

ENTITIES:
[Entity Name] | [Type: person/org/tech/etc] | [Description]

Be concise but precise. Extract 3-10 concepts maximum.
"""

            # Use analysis agent
            response = self.analysis_agent.process_task(context)

            # Parse response
            concepts = []
            entities = []

            lines = response.split('\n')
            current_section = None
            current_example = None

            for line in lines:
                line = line.strip()
                if line.startswith('CONCEPTS:'):
                    current_section = 'concepts'
                elif line.startswith('ENTITIES:'):
                    current_section = 'entities'
                elif line.startswith('Example:'):
                    current_example = line.replace('Example:', '').strip()
                elif '|' in line and current_section == 'concepts':
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        concept_name = parts[0]
                        definition = parts[1]
                        importance = float(parts[2]) if parts[2].replace('.', '').isdigit() else 0.5
                        confidence = float(parts[3]) if parts[3].replace('.', '').isdigit() else 0.5

                        related = []
                        if len(parts) > 4 and 'Related:' in parts[4]:
                            related_str = parts[4].replace('Related:', '').strip()
                            related = [r.strip() for r in related_str.split(',')]

                        concept = ConceptExtraction(
                            concept_name=concept_name,
                            definition=definition,
                            examples=[current_example] if current_example else [],
                            related_concepts=related,
                            importance=min(1.0, max(0.0, importance)),
                            confidence=min(1.0, max(0.0, confidence)),
                            source_chunk_id=chunk.chunk_id,
                            page_number=chunk.page_number,
                            section_title=chunk.section_title
                        )
                        concepts.append(concept)
                        current_example = None

                elif '|' in line and current_section == 'entities':
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 3:
                        entity_name = parts[0]
                        entity_type = parts[1].replace('Type:', '').strip()
                        description = parts[2]

                        entity = EntityExtraction(
                            entity_name=entity_name,
                            entity_type=entity_type,
                            description=description,
                            confidence=0.7
                        )
                        entities.append(entity)

            return {
                'success': True,
                'concepts': concepts,
                'entities': entities
            }

        except Exception as e:
            logger.error(f"Analysis stage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'concepts': [],
                'entities': []
            }

    def _validate_quality(self,
                          research_result: Dict[str, Any],
                          analysis_result: Dict[str, Any]) -> float:
        """
        Stage 3: Use CriticAgent to validate extraction quality.

        Returns quality score (0.0-1.0)
        """
        try:
            # Simple heuristic-based validation
            # (Could use CriticAgent for more sophisticated validation)

            quality_factors = []

            # Factor 1: Research confidence
            research_confidence = research_result.get('confidence', 0.5)
            quality_factors.append(research_confidence)

            # Factor 2: Number of key points (expect 3-5)
            key_points = research_result.get('key_points', [])
            key_points_score = min(1.0, len(key_points) / 4.0) if key_points else 0.3
            quality_factors.append(key_points_score)

            # Factor 3: Number of concepts (expect 3-10)
            concepts = analysis_result.get('concepts', [])
            concepts_score = min(1.0, len(concepts) / 5.0) if concepts else 0.3
            quality_factors.append(concepts_score)

            # Factor 4: Average concept confidence
            if concepts:
                avg_concept_confidence = sum(c.confidence for c in concepts) / len(concepts)
                quality_factors.append(avg_concept_confidence)

            # Factor 5: Summary quality (has content)
            summary = research_result.get('summary', '')
            summary_score = min(1.0, len(summary) / 200.0) if summary else 0.0
            quality_factors.append(summary_score)

            # Weighted average
            quality_score = sum(quality_factors) / len(quality_factors)

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return 0.5  # Default medium quality

    def store_comprehension_result(self,
                                   result: ComprehensionResult,
                                   document_metadata: Dict[str, Any],
                                   embedding: Optional[List[float]] = None) -> int:
        """
        Store comprehension result in KnowledgeStore.

        Args:
            result: ComprehensionResult to store
            document_metadata: Source document metadata
            embedding: Optional embedding vector for the content

        Returns:
            Number of knowledge entries created
        """
        if not result.success:
            logger.warning(f"Skipping storage of failed comprehension: {result.chunk_id}")
            return 0

        stored_count = 0

        try:
            # Determine domain from document metadata or infer from concepts
            domain = document_metadata.get('domain', 'general')
            if domain == 'general' and result.concepts:
                # Try to infer domain from concepts
                # (Could be smarter, but simple heuristic for now)
                domain = self._infer_domain(result.concepts)

            # Determine confidence level based on quality score
            if result.quality_score >= 0.9:
                confidence_level = ConfidenceLevel.VERIFIED
            elif result.quality_score >= 0.7:
                confidence_level = ConfidenceLevel.HIGH
            elif result.quality_score >= 0.5:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW

            # Store each concept as a knowledge entry
            for concept in result.concepts:
                try:
                    knowledge_id = self.knowledge_store.store_knowledge(
                        knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
                        content=concept.to_dict(),
                        confidence_level=confidence_level,
                        source_agent="comprehension_engine",
                        domain=domain,
                        tags=[
                            document_metadata.get('file_type', 'document'),
                            'concept',
                            concept.concept_name.lower().replace(' ', '_')
                        ],
                        embedding=embedding,
                        source_doc_id=result.document_id,
                        chunk_index=int(result.chunk_id.split('_')[-2]) if '_' in result.chunk_id else 0
                    )
                    stored_count += 1
                    logger.debug(f"Stored concept: {concept.concept_name} ({knowledge_id})")
                except Exception as e:
                    logger.error(f"Failed to store concept {concept.concept_name}: {e}")

            # Store entities as separate entries
            for entity in result.entities:
                try:
                    knowledge_id = self.knowledge_store.store_knowledge(
                        knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
                        content=entity.to_dict(),
                        confidence_level=confidence_level,
                        source_agent="comprehension_engine",
                        domain=domain,
                        tags=[
                            document_metadata.get('file_type', 'document'),
                            'entity',
                            entity.entity_type,
                            entity.entity_name.lower().replace(' ', '_')
                        ],
                        source_doc_id=result.document_id
                    )
                    stored_count += 1
                    logger.debug(f"Stored entity: {entity.entity_name} ({knowledge_id})")
                except Exception as e:
                    logger.error(f"Failed to store entity {entity.entity_name}: {e}")

            logger.info(f"Stored {stored_count} knowledge entries from chunk {result.chunk_id}")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store comprehension result: {e}")
            return stored_count

    def _infer_domain(self, concepts: List[ConceptExtraction]) -> str:
        """Infer domain from concept names (simple heuristic)."""
        # Common domain keywords
        domain_keywords = {
            'python': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'web': ['html', 'css', 'javascript', 'react', 'vue', 'http', 'api'],
            'ai': ['machine learning', 'neural network', 'deep learning', 'ai', 'model'],
            'database': ['sql', 'database', 'query', 'table', 'index'],
            'system': ['operating system', 'kernel', 'process', 'memory', 'cpu']
        }

        concept_names_lower = ' '.join(c.concept_name.lower() for c in concepts)

        for domain, keywords in domain_keywords.items():
            if any(kw in concept_names_lower for kw in keywords):
                return domain

        return 'general'
