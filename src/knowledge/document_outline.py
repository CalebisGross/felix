"""
Document Outline Builder for Felix Knowledge Brain

Builds strategic document outlines from chunks WITHOUT using the LLM.
Uses code-based analysis to:
1. Detect document structure (headers, sections, code blocks)
2. Identify domain indicators (tech, legal, medical, business)
3. Calculate chunk complexity/importance scores
4. Recommend processing strategy per chunk

This enables the comprehension engine to work smarter:
- Skip LLM for simple chunks (definitions, code comments)
- Use LLM strategically for complex conceptual content
- Batch related chunks for better context
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum

from .document_ingest import DocumentChunk

logger = logging.getLogger(__name__)


class ChunkComplexity(Enum):
    """Complexity level determining processing strategy."""
    TRIVIAL = "trivial"      # Boilerplate, can skip
    SIMPLE = "simple"        # Code-based extraction sufficient
    MODERATE = "moderate"    # Single LLM call sufficient
    COMPLEX = "complex"      # Full LLM comprehension needed


class ContentType(Enum):
    """Type of content in a chunk."""
    PROSE = "prose"              # Natural language paragraphs
    CODE = "code"                # Source code
    LIST = "list"                # Bullet/numbered lists
    DEFINITION = "definition"    # Term: definition format
    TABLE = "table"              # Tabular data
    HEADER = "header"            # Section headers
    MIXED = "mixed"              # Multiple types


@dataclass
class ChunkAnalysis:
    """Analysis result for a single chunk."""
    chunk_id: str
    chunk_index: int
    complexity: ChunkComplexity
    content_type: ContentType
    importance_score: float  # 0.0-1.0
    detected_concepts: List[str]  # Code-extracted concepts
    detected_relationships: List[Tuple[str, str, str]]  # (source, relation, target)
    section_title: Optional[str]
    domain_indicators: List[str]
    processing_recommendation: str
    skip_llm: bool  # True if code-based extraction is sufficient

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'chunk_index': self.chunk_index,
            'complexity': self.complexity.value,
            'content_type': self.content_type.value,
            'importance_score': self.importance_score,
            'detected_concepts': self.detected_concepts,
            'detected_relationships': [
                {'source': r[0], 'relation': r[1], 'target': r[2]}
                for r in self.detected_relationships
            ],
            'section_title': self.section_title,
            'domain_indicators': self.domain_indicators,
            'processing_recommendation': self.processing_recommendation,
            'skip_llm': self.skip_llm
        }


@dataclass
class DocumentOutline:
    """Strategic outline of an entire document."""
    document_id: str
    file_name: str
    total_chunks: int
    detected_domain: str
    domain_confidence: float
    structure: List[Dict[str, Any]]  # Hierarchical section structure
    chunk_analyses: List[ChunkAnalysis]
    summary_stats: Dict[str, Any]

    def get_llm_required_chunks(self) -> List[int]:
        """Get indices of chunks that need LLM processing."""
        return [
            ca.chunk_index for ca in self.chunk_analyses
            if not ca.skip_llm
        ]

    def get_skippable_chunks(self) -> List[int]:
        """Get indices of chunks that can skip LLM."""
        return [
            ca.chunk_index for ca in self.chunk_analyses
            if ca.skip_llm
        ]

    def get_high_importance_chunks(self, threshold: float = 0.7) -> List[int]:
        """Get indices of high-importance chunks."""
        return [
            ca.chunk_index for ca in self.chunk_analyses
            if ca.importance_score >= threshold
        ]


class DocumentOutlineBuilder:
    """
    Builds document outlines using code-based analysis.

    No LLM calls - pure pattern matching and heuristics.
    """

    # Domain indicator patterns (lowercase)
    DOMAIN_PATTERNS = {
        'programming': [
            r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+', r'\bfunction\s+\w+',
            r'\breturn\b', r'\bimport\b', r'\bexport\b', r'//\s*\w+',
            r'\bapi\b', r'\bjson\b', r'\bhttp[s]?\b', r'\bgithub\b'
        ],
        'legal': [
            r'\bwhereas\b', r'\bhereby\b', r'\bparty\b', r'\bcontract\b',
            r'\bliability\b', r'\bindemnif', r'\bjurisdiction\b', r'\bstatute\b',
            r'\bplaintiff\b', r'\bdefendant\b', r'\bcompliance\b'
        ],
        'medical': [
            r'\bpatient\b', r'\bdiagnos', r'\btreatment\b', r'\bsymptom',
            r'\bdosage\b', r'\bprescri', r'\bclinical\b', r'\btherapy\b',
            r'\bprogno', r'\banato', r'\bphysio'
        ],
        'business': [
            r'\brevenue\b', r'\bprofit\b', r'\bmarket\b', r'\bstrategy\b',
            r'\bstakeholder\b', r'\broi\b', r'\bkpi\b', r'\bquarterly\b',
            r'\bforecast\b', r'\binvestment\b'
        ],
        'research': [
            r'\bhypothesis\b', r'\bmethodology\b', r'\bresults\b', r'\bconclusion\b',
            r'\babstract\b', r'\bcitation\b', r'\bpeer.review', r'\bempir',
            r'\bstatistic', r'\bsample\s+size\b', r'\bp\s*[<>=]'
        ],
        'ai_ml': [
            r'\bmodel\b', r'\btraining\b', r'\binference\b', r'\bneural\b',
            r'\bdeep\s*learning\b', r'\bmachine\s*learning\b', r'\bembedding',
            r'\btransformer\b', r'\battention\b', r'\bloss\s*function\b'
        ]
    }

    # Patterns for extracting explicit definitions
    DEFINITION_PATTERNS = [
        r'^([A-Z][A-Za-z\s]+):\s*(.{20,})',  # "Term: definition..."
        r'^([A-Z][A-Za-z\s]+)\s+is\s+(?:defined\s+as\s+)?(.{20,})',  # "Term is defined as..."
        r'^([A-Z][A-Za-z\s]+)\s+refers\s+to\s+(.{20,})',  # "Term refers to..."
        r'^\*\*([^*]+)\*\*:\s*(.{20,})',  # "**Term**: definition..."
    ]

    # Patterns for extracting relationships
    RELATIONSHIP_PATTERNS = [
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:is\s+a\s+type\s+of|extends|inherits\s+from)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'is_type_of'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:uses|utilizes|employs)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'uses'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:contains|includes|has)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'contains'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:depends\s+on|requires)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'depends_on'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:is\s+similar\s+to|resembles)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'similar_to'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:contrasts\s+with|differs\s+from)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'contrasts_with'),
    ]

    # Header patterns
    HEADER_PATTERNS = [
        r'^#{1,6}\s+(.+)$',  # Markdown headers
        r'^([A-Z][A-Z\s]+)$',  # ALL CAPS lines
        r'^\d+\.\s+([A-Z].+)$',  # Numbered sections
        r'^([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s*$',  # Title Case standalone lines
    ]

    def __init__(self,
                 min_concept_length: int = 3,
                 max_concept_length: int = 50,
                 complexity_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize outline builder.

        Args:
            min_concept_length: Minimum characters for a valid concept name
            max_concept_length: Maximum characters for a valid concept name
            complexity_thresholds: Custom thresholds for complexity scoring
        """
        self.min_concept_length = min_concept_length
        self.max_concept_length = max_concept_length
        self.complexity_thresholds = complexity_thresholds or {
            'trivial_max': 0.2,
            'simple_max': 0.4,
            'moderate_max': 0.7
        }

    def build_outline(self,
                      chunks: List[DocumentChunk],
                      document_id: str,
                      file_name: str) -> DocumentOutline:
        """
        Build a strategic outline from document chunks.

        Args:
            chunks: List of document chunks to analyze
            document_id: Document identifier
            file_name: Original file name

        Returns:
            DocumentOutline with analysis and recommendations
        """
        if not chunks:
            return DocumentOutline(
                document_id=document_id,
                file_name=file_name,
                total_chunks=0,
                detected_domain='general',
                domain_confidence=0.0,
                structure=[],
                chunk_analyses=[],
                summary_stats={}
            )

        # Analyze each chunk
        chunk_analyses = []
        all_domain_indicators = []

        for chunk in chunks:
            analysis = self._analyze_chunk(chunk)
            chunk_analyses.append(analysis)
            all_domain_indicators.extend(analysis.domain_indicators)

        # Detect overall domain
        detected_domain, domain_confidence = self._detect_domain(all_domain_indicators)

        # Build hierarchical structure
        structure = self._build_structure(chunks, chunk_analyses)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(chunk_analyses)

        return DocumentOutline(
            document_id=document_id,
            file_name=file_name,
            total_chunks=len(chunks),
            detected_domain=detected_domain,
            domain_confidence=domain_confidence,
            structure=structure,
            chunk_analyses=chunk_analyses,
            summary_stats=summary_stats
        )

    def _analyze_chunk(self, chunk: DocumentChunk) -> ChunkAnalysis:
        """Analyze a single chunk without LLM."""
        content = chunk.content
        content_lower = content.lower()

        # Detect content type
        content_type = self._detect_content_type(content)

        # Extract concepts and relationships using patterns
        detected_concepts = self._extract_concepts(content)
        detected_relationships = self._extract_relationships(content)

        # Detect domain indicators
        domain_indicators = self._extract_domain_indicators(content_lower)

        # Calculate complexity score
        complexity_score = self._calculate_complexity(content, content_type, detected_concepts)
        complexity = self._score_to_complexity(complexity_score)

        # Calculate importance score
        importance_score = self._calculate_importance(
            content,
            content_type,
            detected_concepts,
            detected_relationships
        )

        # Determine if LLM can be skipped
        skip_llm, recommendation = self._should_skip_llm(
            complexity,
            content_type,
            detected_concepts
        )

        return ChunkAnalysis(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            complexity=complexity,
            content_type=content_type,
            importance_score=importance_score,
            detected_concepts=detected_concepts,
            detected_relationships=detected_relationships,
            section_title=chunk.section_title,
            domain_indicators=domain_indicators,
            processing_recommendation=recommendation,
            skip_llm=skip_llm
        )

    def _detect_content_type(self, content: str) -> ContentType:
        """Detect the primary content type of the chunk."""
        lines = content.split('\n')

        # Count indicators
        code_lines = 0
        list_lines = 0
        definition_lines = 0
        header_lines = 0
        prose_lines = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Code indicators
            if (re.match(r'^\s{4,}', line) or  # Indented
                re.match(r'^```', stripped) or  # Code fence
                re.search(r'[{}\[\]();=]', stripped)):  # Code syntax
                code_lines += 1
            # List indicators
            elif re.match(r'^[-*•]\s', stripped) or re.match(r'^\d+\.\s', stripped):
                list_lines += 1
            # Definition indicators
            elif re.match(r'^[A-Z][a-z]+.*:\s+\S', stripped):
                definition_lines += 1
            # Header indicators
            elif any(re.match(p, stripped) for p in self.HEADER_PATTERNS):
                header_lines += 1
            else:
                prose_lines += 1

        total = code_lines + list_lines + definition_lines + header_lines + prose_lines
        if total == 0:
            return ContentType.PROSE

        # Determine primary type (>50% or mixed)
        if code_lines / total > 0.5:
            return ContentType.CODE
        elif list_lines / total > 0.5:
            return ContentType.LIST
        elif definition_lines / total > 0.5:
            return ContentType.DEFINITION
        elif header_lines / total > 0.5:
            return ContentType.HEADER
        elif prose_lines / total > 0.5:
            return ContentType.PROSE
        else:
            return ContentType.MIXED

    def _extract_concepts(self, content: str) -> List[str]:
        """Extract concept names using pattern matching.

        Only extracts concepts that have actual definitions in the text.
        Removed: aggressive capitalized word extraction that created garbage
        like "Confidence", "Priority", "Task" with placeholder definitions.
        """
        concepts = set()

        # ONLY extract from definition patterns (Term: definition format)
        # These have actual definitions, not just names
        for pattern in self.DEFINITION_PATTERNS:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    concept = match[0].strip()
                else:
                    concept = match.strip()
                if self._is_valid_concept(concept):
                    concepts.add(concept)

        # REMOVED: No more capitalized word extraction
        # This was creating garbage like "Confidence", "Priority", "Task"
        # with placeholder definitions "A concept mentioned in relation to..."

        return list(concepts)[:20]  # Limit to prevent explosion

    def _extract_relationships(self, content: str) -> List[Tuple[str, str, str]]:
        """Extract explicit relationships using patterns."""
        relationships = []

        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            matches = re.findall(pattern, content)
            for match in matches:
                source, target = match
                if self._is_valid_concept(source) and self._is_valid_concept(target):
                    relationships.append((source.strip(), rel_type, target.strip()))

        return relationships[:10]  # Limit to prevent explosion

    def _extract_domain_indicators(self, content_lower: str) -> List[str]:
        """Extract domain indicator matches."""
        indicators = []

        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    indicators.append(domain)
                    break  # One match per domain is enough

        return indicators

    def _calculate_complexity(self,
                              content: str,
                              content_type: ContentType,
                              concepts: List[str]) -> float:
        """Calculate complexity score (0.0-1.0)."""
        score = 0.0

        # Factor 1: Length (longer = more complex)
        length_score = min(1.0, len(content) / 3000)
        score += length_score * 0.2

        # Factor 2: Sentence complexity (average words per sentence)
        sentences = re.split(r'[.!?]+', content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            sentence_score = min(1.0, avg_sentence_length / 30)
            score += sentence_score * 0.2

        # Factor 3: Concept density
        concept_density = len(concepts) / (len(content.split()) + 1)
        concept_score = min(1.0, concept_density * 20)
        score += concept_score * 0.2

        # Factor 4: Content type complexity
        type_scores = {
            ContentType.HEADER: 0.1,
            ContentType.LIST: 0.3,
            ContentType.DEFINITION: 0.4,
            ContentType.CODE: 0.5,
            ContentType.PROSE: 0.7,
            ContentType.MIXED: 0.8,
        }
        score += type_scores.get(content_type, 0.5) * 0.2

        # Factor 5: Abstract/technical language density
        technical_terms = len(re.findall(r'\b(?:algorithm|framework|architecture|implementation|abstraction|interface)\b', content.lower()))
        technical_score = min(1.0, technical_terms / 5)
        score += technical_score * 0.2

        return min(1.0, score)

    def _score_to_complexity(self, score: float) -> ChunkComplexity:
        """Convert complexity score to enum."""
        if score <= self.complexity_thresholds['trivial_max']:
            return ChunkComplexity.TRIVIAL
        elif score <= self.complexity_thresholds['simple_max']:
            return ChunkComplexity.SIMPLE
        elif score <= self.complexity_thresholds['moderate_max']:
            return ChunkComplexity.MODERATE
        else:
            return ChunkComplexity.COMPLEX

    def _calculate_importance(self,
                              content: str,
                              content_type: ContentType,
                              concepts: List[str],
                              relationships: List[Tuple[str, str, str]]) -> float:
        """Calculate importance score (0.0-1.0)."""
        score = 0.0

        # Factor 1: Has extractable concepts
        if concepts:
            score += min(0.3, len(concepts) * 0.05)

        # Factor 2: Has explicit relationships (very valuable)
        if relationships:
            score += min(0.3, len(relationships) * 0.1)

        # Factor 3: Definition content is valuable
        if content_type == ContentType.DEFINITION:
            score += 0.2

        # Factor 4: Contains introduction/conclusion indicators
        intro_conclusion = re.search(
            r'\b(introduction|overview|summary|conclusion|key\s+point|takeaway)\b',
            content.lower()
        )
        if intro_conclusion:
            score += 0.1

        # Factor 5: Unique/specific content (not boilerplate)
        boilerplate_indicators = [
            'copyright', 'all rights reserved', 'table of contents',
            'page', 'chapter', 'appendix', 'index'
        ]
        boilerplate_count = sum(1 for ind in boilerplate_indicators if ind in content.lower())
        if boilerplate_count == 0:
            score += 0.1

        return min(1.0, score)

    def _should_skip_llm(self,
                         complexity: ChunkComplexity,
                         content_type: ContentType,
                         concepts: List[str]) -> Tuple[bool, str]:
        """
        Determine if LLM processing can be skipped for this chunk.

        Returns:
            Tuple of (skip_llm: bool, recommendation: str)
        """
        # Trivial chunks: skip entirely
        if complexity == ChunkComplexity.TRIVIAL:
            return True, "Skip - trivial content (boilerplate/headers)"

        # Simple chunks with extracted concepts: use code-based only
        if complexity == ChunkComplexity.SIMPLE and concepts:
            return True, f"Code-based extraction ({len(concepts)} concepts detected)"

        # Definition-heavy content: code extraction often sufficient
        if content_type == ContentType.DEFINITION and len(concepts) >= 3:
            return True, f"Code-based extraction (definition format, {len(concepts)} concepts)"

        # Code content: might not need LLM comprehension
        if content_type == ContentType.CODE:
            return True, "Code-based extraction (source code - extract comments/docstrings)"

        # Everything else needs LLM
        if complexity == ChunkComplexity.MODERATE:
            return False, "Single LLM call (moderate complexity)"
        else:
            return False, "Full LLM comprehension (complex content)"

    def _is_valid_concept(self, concept: str) -> bool:
        """Validate a potential concept name."""
        if not concept:
            return False
        if len(concept) < self.min_concept_length:
            return False
        if len(concept) > self.max_concept_length:
            return False
        # Reject pure numbers or single characters
        if concept.isdigit():
            return False
        # Reject common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out'}
        if concept.lower() in stop_words:
            return False
        return True

    def _detect_domain(self, indicators: List[str]) -> Tuple[str, float]:
        """Detect overall document domain from indicators."""
        if not indicators:
            return 'general', 0.0

        counter = Counter(indicators)
        total = len(indicators)

        if not counter:
            return 'general', 0.0

        # Get most common domain
        domain, count = counter.most_common(1)[0]
        confidence = count / total

        # Require minimum confidence
        if confidence < 0.3:
            return 'general', confidence

        return domain, confidence

    def _build_structure(self,
                         chunks: List[DocumentChunk],
                         analyses: List[ChunkAnalysis]) -> List[Dict[str, Any]]:
        """Build hierarchical document structure."""
        structure = []
        current_section = None

        for chunk, analysis in zip(chunks, analyses):
            # Detect section headers
            if analysis.content_type == ContentType.HEADER or chunk.section_title:
                title = chunk.section_title or self._extract_header(chunk.content)
                if title:
                    current_section = {
                        'title': title,
                        'start_chunk': chunk.chunk_index,
                        'end_chunk': chunk.chunk_index,
                        'subsections': []
                    }
                    structure.append(current_section)
            elif current_section:
                current_section['end_chunk'] = chunk.chunk_index

        return structure

    def _extract_header(self, content: str) -> Optional[str]:
        """Extract header text from content."""
        for pattern in self.HEADER_PATTERNS:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None

    def _calculate_summary_stats(self, analyses: List[ChunkAnalysis]) -> Dict[str, Any]:
        """Calculate summary statistics for the outline."""
        if not analyses:
            return {}

        complexities = Counter(a.complexity for a in analyses)
        content_types = Counter(a.content_type for a in analyses)

        llm_required = sum(1 for a in analyses if not a.skip_llm)
        skippable = sum(1 for a in analyses if a.skip_llm)

        avg_importance = sum(a.importance_score for a in analyses) / len(analyses)

        total_concepts = sum(len(a.detected_concepts) for a in analyses)
        total_relationships = sum(len(a.detected_relationships) for a in analyses)

        return {
            'complexity_distribution': {k.value: v for k, v in complexities.items()},
            'content_type_distribution': {k.value: v for k, v in content_types.items()},
            'llm_required_chunks': llm_required,
            'skippable_chunks': skippable,
            'llm_skip_rate': skippable / len(analyses) if analyses else 0,
            'average_importance': avg_importance,
            'total_code_extracted_concepts': total_concepts,
            'total_code_extracted_relationships': total_relationships,
            'estimated_llm_savings': f"{(skippable / len(analyses) * 100):.1f}%"
        }
