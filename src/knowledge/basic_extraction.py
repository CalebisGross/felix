"""
Basic Knowledge Extraction for Felix Knowledge Brain

Code-based extraction that works WITHOUT LLM calls.
Handles chunks that the outline builder marked as "skip_llm":
- Definition-formatted content (Term: definition)
- Source code (extracts from docstrings, comments)
- Lists (extracts structured items)
- Markdown headers and sections

This provides a fast path for simple content, reserving
LLM calls for complex conceptual analysis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .document_ingest import DocumentChunk
from .comprehension import ConceptExtraction, EntityExtraction, ComprehensionResult, clean_concept_name, validate_concept

logger = logging.getLogger(__name__)


class BasicExtractionEngine:
    """
    Extract knowledge from document chunks using pattern matching.

    No LLM calls - pure regex and heuristics.
    Works on chunks marked as "skip_llm" by the outline builder.
    """

    # Definition patterns (capture term and definition)
    DEFINITION_PATTERNS = [
        # "**Term**: Definition text..."
        (r'^\*\*([^*]+)\*\*\s*[:\-]\s*(.{20,}?)(?:\.|$)', 'markdown_bold'),
        # "Term: Definition text..."
        (r'^([A-Z][A-Za-z\s]{2,40}):\s+([A-Z].{20,}?)(?:\.|$)', 'colon_definition'),
        # "Term - Definition text..."
        (r'^([A-Z][A-Za-z\s]{2,40})\s+-\s+(.{20,}?)(?:\.|$)', 'dash_definition'),
        # "Term is defined as..."
        (r'([A-Z][A-Za-z\s]{2,30})\s+is\s+defined\s+as\s+(.{20,}?)(?:\.|$)', 'explicit_definition'),
        # "Term refers to..."
        (r'([A-Z][A-Za-z\s]{2,30})\s+refers\s+to\s+(.{20,}?)(?:\.|$)', 'refers_to'),
        # "A Term is..."
        (r'[Aa]n?\s+([A-Z][A-Za-z\s]{2,30})\s+is\s+(.{20,}?)(?:\.|$)', 'a_term_is'),
    ]

    # Code documentation patterns
    CODE_DOC_PATTERNS = {
        'python': [
            # Python docstrings
            (r'"""(.+?)"""', 'docstring'),
            (r"'''(.+?)'''", 'docstring'),
            # Python comments with descriptions
            (r'#\s*([A-Z][A-Za-z\s]+):\s*(.+)', 'comment'),
            # Class definitions
            (r'class\s+(\w+)(?:\([^)]*\))?:', 'class'),
            # Function definitions
            (r'def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*\w+)?:', 'function'),
        ],
        'javascript': [
            # JSDoc comments
            (r'/\*\*\s*\n?\s*\*?\s*(.+?)\s*\*/', 'jsdoc'),
            # Single-line comments
            (r'//\s*([A-Z][A-Za-z\s]+):\s*(.+)', 'comment'),
            # Function declarations
            (r'(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:function)?\s*\(', 'function'),
            # Class declarations
            (r'class\s+(\w+)(?:\s+extends\s+\w+)?', 'class'),
        ],
        'generic': [
            # C-style block comments
            (r'/\*\s*(.+?)\s*\*/', 'block_comment'),
            # Shell/Python style comments
            (r'#\s*(.{10,})', 'comment'),
        ]
    }

    # List item patterns
    LIST_PATTERNS = [
        # Bullet lists with term
        (r'^[-*•]\s+\*\*([^*]+)\*\*\s*[:\-]?\s*(.+)', 'bullet_bold'),
        (r'^[-*•]\s+([A-Z][A-Za-z\s]+)\s*[:\-]\s*(.+)', 'bullet_term'),
        # Numbered lists with term
        (r'^\d+\.\s+\*\*([^*]+)\*\*\s*[:\-]?\s*(.+)', 'numbered_bold'),
        (r'^\d+\.\s+([A-Z][A-Za-z\s]+)\s*[:\-]\s*(.+)', 'numbered_term'),
    ]

    # Relationship extraction patterns
    RELATIONSHIP_PATTERNS = [
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:is\s+a|is\s+an)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'is_a'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:extends|inherits\s+from)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'extends'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:contains|includes)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'contains'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:uses|utilizes)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'uses'),
        (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+(?:requires|depends\s+on)\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'depends_on'),
    ]

    def __init__(self,
                 min_definition_length: int = 20,
                 max_concepts_per_chunk: int = 15,
                 confidence_base: float = 0.6):
        """
        Initialize basic extraction engine.

        Args:
            min_definition_length: Minimum chars for a valid definition
            max_concepts_per_chunk: Maximum concepts to extract per chunk
            confidence_base: Base confidence for code-extracted concepts
        """
        self.min_definition_length = min_definition_length
        self.max_concepts_per_chunk = max_concepts_per_chunk
        self.confidence_base = confidence_base

    def extract_from_chunk(self,
                           chunk: DocumentChunk,
                           content_type: str = 'prose') -> ComprehensionResult:
        """
        Extract knowledge from a chunk using pattern matching.

        Args:
            chunk: Document chunk to process
            content_type: Hint about content type (prose/code/definition/list)

        Returns:
            ComprehensionResult with extracted concepts
        """
        import time
        start_time = time.time()

        try:
            content = chunk.content
            concepts = []
            entities = []

            # Try different extraction strategies based on content type
            if content_type == 'code':
                concepts.extend(self._extract_from_code(content))
            elif content_type == 'definition':
                concepts.extend(self._extract_from_definitions(content))
            elif content_type == 'list':
                concepts.extend(self._extract_from_lists(content))
            else:
                # Try all strategies for prose/mixed
                concepts.extend(self._extract_from_definitions(content))
                concepts.extend(self._extract_from_lists(content))

            # Extract relationships (works for all types)
            relationships = self._extract_relationships(content)

            # Add relationships to related_concepts field
            self._link_relationships(concepts, relationships)

            # Deduplicate and limit
            concepts = self._deduplicate_concepts(concepts)[:self.max_concepts_per_chunk]

            # Set chunk metadata
            for concept in concepts:
                concept.source_chunk_id = chunk.chunk_id
                concept.page_number = chunk.page_number
                concept.section_title = chunk.section_title

            # Generate summary from extracted concepts
            summary = self._generate_summary(concepts) if concepts else ""

            # Extract key points from content
            key_points = self._extract_key_points(content)

            # Calculate quality based on extraction success
            quality_score = self._calculate_quality(concepts, relationships)

            return ComprehensionResult(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                summary=summary,
                concepts=concepts,
                entities=entities,
                key_points=key_points,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                success=True
            )

        except Exception as e:
            logger.error(f"Basic extraction failed for chunk {chunk.chunk_id}: {e}")
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

    def _extract_from_definitions(self, content: str) -> List[ConceptExtraction]:
        """Extract concepts from definition-formatted text."""
        concepts = []

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            for pattern, pattern_type in self.DEFINITION_PATTERNS:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    concept_name = clean_concept_name(match.group(1))
                    definition = match.group(2).strip()

                    if validate_concept(concept_name, definition):
                        concepts.append(ConceptExtraction(
                            concept_name=concept_name,
                            definition=definition,
                            examples=[],
                            related_concepts=[],
                            importance=0.6,
                            confidence=self.confidence_base,
                        ))
                    break  # One match per line

        return concepts

    def _extract_from_code(self, content: str) -> List[ConceptExtraction]:
        """Extract concepts from source code (docstrings, comments, class/function names)."""
        concepts = []

        # Detect language
        if 'def ' in content or 'import ' in content:
            patterns = self.CODE_DOC_PATTERNS['python']
        elif 'function ' in content or 'const ' in content or 'var ' in content:
            patterns = self.CODE_DOC_PATTERNS['javascript']
        else:
            patterns = self.CODE_DOC_PATTERNS['generic']

        # Extract from documentation patterns
        for pattern, pattern_type in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                if pattern_type in ('class', 'function'):
                    # Class or function name
                    name = match if isinstance(match, str) else match[0]
                    name = clean_concept_name(name)

                    # Try to find associated docstring
                    definition = self._find_associated_docstring(content, name)
                    if not definition:
                        definition = f"A {pattern_type} named {name}"

                    if validate_concept(name, definition):
                        concepts.append(ConceptExtraction(
                            concept_name=name,
                            definition=definition,
                            examples=[],
                            related_concepts=[],
                            importance=0.5,
                            confidence=self.confidence_base - 0.1,  # Lower confidence for inferred
                        ))

                elif pattern_type == 'docstring':
                    # Parse docstring for concept information
                    docstring = match if isinstance(match, str) else match[0]
                    doc_concepts = self._parse_docstring(docstring)
                    concepts.extend(doc_concepts)

                elif pattern_type == 'comment' and isinstance(match, tuple):
                    # Comment with term: description format
                    name = clean_concept_name(match[0])
                    definition = match[1].strip()
                    if validate_concept(name, definition):
                        concepts.append(ConceptExtraction(
                            concept_name=name,
                            definition=definition,
                            examples=[],
                            related_concepts=[],
                            importance=0.5,
                            confidence=self.confidence_base - 0.1,
                        ))

        return concepts

    def _extract_from_lists(self, content: str) -> List[ConceptExtraction]:
        """Extract concepts from bullet/numbered lists."""
        concepts = []

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            for pattern, pattern_type in self.LIST_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    concept_name = clean_concept_name(match.group(1))
                    definition = match.group(2).strip()

                    if validate_concept(concept_name, definition):
                        concepts.append(ConceptExtraction(
                            concept_name=concept_name,
                            definition=definition,
                            examples=[],
                            related_concepts=[],
                            importance=0.5,
                            confidence=self.confidence_base,
                        ))
                    break

        return concepts

    def _extract_relationships(self, content: str) -> List[Tuple[str, str, str]]:
        """Extract explicit relationships from content."""
        relationships = []

        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            matches = re.findall(pattern, content)
            for match in matches:
                source = clean_concept_name(match[0])
                target = clean_concept_name(match[1])
                if source and target and source != target:
                    relationships.append((source, rel_type, target))

        return relationships

    def _link_relationships(self,
                            concepts: List[ConceptExtraction],
                            relationships: List[Tuple[str, str, str]]):
        """Add relationship info to concept's related_concepts field."""
        # Build name lookup
        concept_names = {c.concept_name.lower() for c in concepts}

        for source, rel_type, target in relationships:
            source_lower = source.lower()
            target_lower = target.lower()

            # Find matching concepts and add related
            for concept in concepts:
                if concept.concept_name.lower() == source_lower:
                    if target not in concept.related_concepts:
                        concept.related_concepts.append(target)
                elif concept.concept_name.lower() == target_lower:
                    if source not in concept.related_concepts:
                        concept.related_concepts.append(source)

    def _find_associated_docstring(self, content: str, name: str) -> Optional[str]:
        """Find docstring associated with a class/function name."""
        # Look for docstring after definition
        patterns = [
            rf'{name}[^:]*:\s*\n\s*"""([^"]+)"""',
            rf'{name}[^:]*:\s*\n\s*\'\'\'([^\']+)\'\'\'',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                docstring = match.group(1).strip()
                # Return first line as definition
                first_line = docstring.split('\n')[0].strip()
                if len(first_line) >= 20:
                    return first_line

        return None

    def _parse_docstring(self, docstring: str) -> List[ConceptExtraction]:
        """Parse a docstring for concepts."""
        concepts = []

        # First line is usually a summary
        lines = docstring.strip().split('\n')
        if not lines:
            return concepts

        # Look for Args/Parameters section
        in_args = False
        for line in lines:
            stripped = line.strip()

            if stripped.lower().startswith(('args:', 'parameters:', 'attributes:')):
                in_args = True
                continue
            elif stripped.lower().startswith(('returns:', 'raises:', 'examples:')):
                in_args = False
                continue

            # Parse argument definitions
            if in_args:
                match = re.match(r'(\w+)\s*(?:\([^)]+\))?\s*:\s*(.+)', stripped)
                if match:
                    param_name = match.group(1)
                    param_desc = match.group(2).strip()

                    if len(param_name) >= 3 and len(param_desc) >= 10:
                        concepts.append(ConceptExtraction(
                            concept_name=param_name,
                            definition=param_desc,
                            examples=[],
                            related_concepts=[],
                            importance=0.4,
                            confidence=self.confidence_base - 0.2,
                        ))

        return concepts

    def _deduplicate_concepts(self, concepts: List[ConceptExtraction]) -> List[ConceptExtraction]:
        """Remove duplicate concepts, keeping highest confidence."""
        seen = {}

        for concept in concepts:
            key = concept.concept_name.lower()
            if key not in seen:
                seen[key] = concept
            elif concept.confidence > seen[key].confidence:
                # Keep higher confidence version
                seen[key] = concept
            elif len(concept.definition) > len(seen[key].definition):
                # Keep longer definition if same confidence
                seen[key] = concept

        return list(seen.values())

    def _generate_summary(self, concepts: List[ConceptExtraction]) -> str:
        """Generate a summary from extracted concepts."""
        if not concepts:
            return ""

        concept_names = [c.concept_name for c in concepts[:5]]

        if len(concept_names) == 1:
            return f"This section covers {concept_names[0]}."
        elif len(concept_names) == 2:
            return f"This section covers {concept_names[0]} and {concept_names[1]}."
        else:
            last = concept_names[-1]
            others = ', '.join(concept_names[:-1])
            return f"This section covers {others}, and {last}."

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        key_points = []

        # Look for explicit key point indicators
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()

            # Key/important indicators
            if re.match(r'^(?:key|important|note|remember|tip):', stripped, re.IGNORECASE):
                point = re.sub(r'^(?:key|important|note|remember|tip):\s*', '', stripped, flags=re.IGNORECASE)
                if len(point) >= 20:
                    key_points.append(point)

            # Numbered important items
            elif re.match(r'^\d+\.\s*(?:key|important)', stripped, re.IGNORECASE):
                key_points.append(stripped)

        return key_points[:5]

    def _calculate_quality(self,
                           concepts: List[ConceptExtraction],
                           relationships: List[Tuple[str, str, str]]) -> float:
        """Calculate quality score for extraction."""
        if not concepts:
            return 0.3  # Low quality if no concepts

        # Factor 1: Number of concepts
        concept_score = min(1.0, len(concepts) / 5)

        # Factor 2: Average definition length
        avg_def_len = sum(len(c.definition) for c in concepts) / len(concepts)
        definition_score = min(1.0, avg_def_len / 100)

        # Factor 3: Relationships found
        relationship_score = min(1.0, len(relationships) / 3) if relationships else 0

        # Factor 4: Related concepts filled in
        related_score = sum(1 for c in concepts if c.related_concepts) / len(concepts)

        # Weighted average
        quality = (
            concept_score * 0.3 +
            definition_score * 0.3 +
            relationship_score * 0.2 +
            related_score * 0.2
        )

        return min(1.0, max(0.0, quality))


def extract_without_llm(chunk: DocumentChunk,
                        content_type: str = 'prose') -> ComprehensionResult:
    """
    Convenience function for basic extraction.

    Args:
        chunk: Document chunk to extract from
        content_type: Hint about content type

    Returns:
        ComprehensionResult with extracted concepts
    """
    engine = BasicExtractionEngine()
    return engine.extract_from_chunk(chunk, content_type)
