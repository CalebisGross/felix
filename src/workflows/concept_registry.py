"""
Shared Conceptual Registry for Felix workflows.

Provides workflow-scoped concept tracking to ensure terminology consistency
across agents, detect duplicate definitions, and prevent conflicting interpretations.

This addresses the workflow recommendation: "Shared Conceptual Registry - Aggregate
insights across agents to enable systemic learning."
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConceptDefinition:
    """A single concept definition with metadata."""
    name: str
    definition: str
    source_agent: str
    confidence: float
    timestamp: float
    related_concepts: List[str] = field(default_factory=list)
    usage_count: int = 0


@dataclass
class ConceptConflict:
    """Detected conflict between concept definitions."""
    concept_name: str
    definition1: ConceptDefinition
    definition2: ConceptDefinition
    conflict_type: str  # 'duplicate', 'contradictory', 'overlapping'
    severity: float  # 0.0 (minor) to 1.0 (critical)


class ConceptRegistry:
    """
    Workflow-scoped registry for tracking concept definitions and relationships.

    Enables agents to:
    - Check if a concept has already been defined
    - Register new concepts with source attribution
    - Detect terminology conflicts and duplicates
    - Query related concepts and concept relationships

    This prevents agents from defining concepts inconsistently and enables
    systemic learning across the agent team.
    """

    def __init__(self, workflow_id: str):
        """
        Initialize concept registry for a specific workflow.

        Args:
            workflow_id: Unique identifier for the workflow
        """
        self.workflow_id = workflow_id
        self.concepts: Dict[str, ConceptDefinition] = {}
        self.conflicts: List[ConceptConflict] = []
        self._concept_aliases: Dict[str, str] = {}  # Alias -> canonical name
        logger.info(f"📚 ConceptRegistry initialized for workflow {workflow_id}")

    def register_concept(self, name: str, definition: str,
                        source_agent: str, confidence: float = 0.7) -> bool:
        """
        Register a new concept or update existing definition.

        Args:
            name: Concept name (case-insensitive)
            definition: Concept definition or description
            source_agent: Agent ID that defined this concept
            confidence: Agent's confidence in this definition (0.0-1.0)

        Returns:
            True if newly registered, False if duplicate detected
        """
        canonical_name = self._canonicalize_name(name)

        # Check if concept already exists
        if canonical_name in self.concepts:
            return self._handle_duplicate(canonical_name, definition, source_agent, confidence)

        # Register new concept
        self.concepts[canonical_name] = ConceptDefinition(
            name=name,
            definition=definition,
            source_agent=source_agent,
            confidence=confidence,
            timestamp=time.time()
        )

        logger.info(f"  ✓ Registered concept '{name}' by {source_agent} (confidence={confidence:.2f})")
        return True

    def get_concept(self, name: str) -> Optional[ConceptDefinition]:
        """
        Retrieve a concept definition by name.

        Args:
            name: Concept name (case-insensitive)

        Returns:
            ConceptDefinition if found, None otherwise
        """
        canonical_name = self._canonicalize_name(name)

        # Check direct match
        if canonical_name in self.concepts:
            self.concepts[canonical_name].usage_count += 1
            return self.concepts[canonical_name]

        # Check aliases
        if canonical_name in self._concept_aliases:
            main_concept = self._concept_aliases[canonical_name]
            self.concepts[main_concept].usage_count += 1
            return self.concepts[main_concept]

        return None

    def find_related_concepts(self, concept_name: str, max_results: int = 5) -> List[str]:
        """
        Find concepts related to the given concept.

        Args:
            concept_name: Concept to find relations for
            max_results: Maximum number of related concepts to return

        Returns:
            List of related concept names
        """
        concept = self.get_concept(concept_name)
        if not concept:
            return []

        # Return explicitly defined related concepts
        return concept.related_concepts[:max_results]

    def link_concepts(self, concept1: str, concept2: str, bidirectional: bool = True) -> bool:
        """
        Create a relationship link between two concepts.

        Args:
            concept1: First concept name
            concept2: Second concept name
            bidirectional: If True, creates link in both directions

        Returns:
            True if link created, False if concepts don't exist
        """
        c1 = self.get_concept(concept1)
        c2 = self.get_concept(concept2)

        if not c1 or not c2:
            return False

        canonical1 = self._canonicalize_name(concept1)
        canonical2 = self._canonicalize_name(concept2)

        # Add relationship
        if canonical2 not in self.concepts[canonical1].related_concepts:
            self.concepts[canonical1].related_concepts.append(canonical2)

        if bidirectional and canonical1 not in self.concepts[canonical2].related_concepts:
            self.concepts[canonical2].related_concepts.append(canonical1)

        logger.debug(f"  🔗 Linked concepts: {concept1} ↔ {concept2}")
        return True

    def find_conflicts(self) -> List[ConceptConflict]:
        """
        Identify conflicts between concept definitions.

        Returns:
            List of detected conflicts
        """
        # Return cached conflicts if already computed
        if self.conflicts:
            return self.conflicts

        self.conflicts = []

        # Check for overlapping or contradictory definitions
        concept_list = list(self.concepts.items())
        for i, (name1, concept1) in enumerate(concept_list):
            for name2, concept2 in concept_list[i+1:]:
                # Check for similar names with different definitions
                if self._names_similar(name1, name2):
                    conflict = self._evaluate_conflict(concept1, concept2)
                    if conflict:
                        self.conflicts.append(conflict)

        if self.conflicts:
            logger.warning(f"⚠️  Found {len(self.conflicts)} concept conflicts in workflow")

        return self.conflicts

    def get_all_concepts(self) -> List[ConceptDefinition]:
        """
        Get all registered concepts.

        Returns:
            List of all concept definitions
        """
        return list(self.concepts.values())

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry state.

        Returns:
            Dictionary with registry statistics
        """
        total_concepts = len(self.concepts)
        total_relationships = sum(len(c.related_concepts) for c in self.concepts.values())
        total_conflicts = len(self.find_conflicts())

        most_used = sorted(self.concepts.values(), key=lambda c: c.usage_count, reverse=True)[:5]

        return {
            'workflow_id': self.workflow_id,
            'total_concepts': total_concepts,
            'total_relationships': total_relationships // 2,  # Bidirectional counted twice
            'total_conflicts': total_conflicts,
            'most_used_concepts': [c.name for c in most_used],
            'agent_contributions': self._count_agent_contributions()
        }

    def _canonicalize_name(self, name: str) -> str:
        """Convert concept name to canonical form (lowercase, stripped)."""
        return name.lower().strip()

    def _handle_duplicate(self, canonical_name: str, definition: str,
                         source_agent: str, confidence: float) -> bool:
        """
        Handle registration of a duplicate concept name.

        Args:
            canonical_name: Canonical concept name
            definition: New definition
            source_agent: Agent proposing the new definition
            confidence: Confidence in new definition

        Returns:
            False (indicating duplicate)
        """
        existing = self.concepts[canonical_name]

        # Check if definitions are similar (might be acceptable)
        similarity = self._definition_similarity(existing.definition, definition)

        if similarity > 0.7:
            # Definitions are similar enough - treat as confirmation
            logger.debug(f"  ✓ Concept '{canonical_name}' confirmed by {source_agent} "
                        f"(similarity={similarity:.2f})")
            # Boost confidence slightly
            existing.confidence = min(1.0, existing.confidence + 0.05)
            return False
        else:
            # Definitions conflict - create conflict record
            conflict = ConceptConflict(
                concept_name=canonical_name,
                definition1=existing,
                definition2=ConceptDefinition(
                    name=canonical_name,
                    definition=definition,
                    source_agent=source_agent,
                    confidence=confidence,
                    timestamp=time.time()
                ),
                conflict_type='contradictory',
                severity=1.0 - similarity
            )
            self.conflicts.append(conflict)

            logger.warning(f"  ⚠️  Conflicting definition for '{canonical_name}' from {source_agent}")
            logger.warning(f"      Existing: {existing.definition[:100]}...")
            logger.warning(f"      New: {definition[:100]}...")

            return False

    def _definition_similarity(self, def1: str, def2: str) -> float:
        """
        Calculate similarity between two definitions (simple word overlap).

        Args:
            def1: First definition
            def2: Second definition

        Returns:
            Similarity score from 0.0 (no overlap) to 1.0 (identical)
        """
        words1 = set(def1.lower().split())
        words2 = set(def2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _names_similar(self, name1: str, name2: str) -> bool:
        """Check if two concept names are similar (potential duplicates)."""
        # Simple check: edit distance or substring match
        if name1 == name2:
            return True

        # Check if one is substring of other
        if name1 in name2 or name2 in name1:
            return True

        return False

    def _evaluate_conflict(self, concept1: ConceptDefinition,
                          concept2: ConceptDefinition) -> Optional[ConceptConflict]:
        """
        Evaluate whether two similar concepts conflict.

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            ConceptConflict if conflict detected, None otherwise
        """
        similarity = self._definition_similarity(concept1.definition, concept2.definition)

        if similarity < 0.5:
            # Significantly different definitions for similar names
            return ConceptConflict(
                concept_name=concept1.name,
                definition1=concept1,
                definition2=concept2,
                conflict_type='contradictory',
                severity=1.0 - similarity
            )

        return None

    def _count_agent_contributions(self) -> Dict[str, int]:
        """Count how many concepts each agent contributed."""
        counts = {}
        for concept in self.concepts.values():
            counts[concept.source_agent] = counts.get(concept.source_agent, 0) + 1
        return counts

    def export_to_markdown(self, filepath: str) -> None:
        """
        Export registry to markdown file as recommended by workflow.

        Args:
            filepath: Path to output markdown file (e.g., analysis/improvement_registry.md)
        """
        summary = self.get_summary()

        with open(filepath, 'w') as f:
            f.write(f"# Concept Registry - Workflow {self.workflow_id}\n\n")
            f.write(f"**Total Concepts**: {summary['total_concepts']}\n")
            f.write(f"**Total Relationships**: {summary['total_relationships']}\n")
            f.write(f"**Total Conflicts**: {summary['total_conflicts']}\n\n")

            f.write("## Most Used Concepts\n\n")
            for concept_name in summary['most_used_concepts']:
                concept = self.get_concept(concept_name)
                f.write(f"- **{concept.name}** (used {concept.usage_count} times)\n")
                f.write(f"  - {concept.definition[:150]}...\n")
                f.write(f"  - Source: {concept.source_agent}\n\n")

            f.write("## Agent Contributions\n\n")
            for agent, count in summary['agent_contributions'].items():
                f.write(f"- {agent}: {count} concepts\n")

            if self.conflicts:
                f.write("\n## Conflicts Detected\n\n")
                for conflict in self.conflicts:
                    f.write(f"### {conflict.concept_name} (severity: {conflict.severity:.2f})\n\n")
                    f.write(f"**Definition 1** ({conflict.definition1.source_agent}):\n")
                    f.write(f"{conflict.definition1.definition}\n\n")
                    f.write(f"**Definition 2** ({conflict.definition2.source_agent}):\n")
                    f.write(f"{conflict.definition2.definition}\n\n")

        logger.info(f"📄 Exported concept registry to {filepath}")
