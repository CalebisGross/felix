"""
Knowledge Tool for querying Felix knowledge base in conversational CLI.
"""

from typing import Dict, Any, List, Optional
from .base_tool import BaseTool, ToolResult


class KnowledgeTool(BaseTool):
    """Tool for querying and exploring the knowledge base."""

    @property
    def name(self) -> str:
        return "knowledge"

    @property
    def description(self) -> str:
        return "Query and explore the Felix knowledge base"

    @property
    def usage(self) -> str:
        return "/knowledge [search|concepts|domains|graph] <query> [options]"

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Execute knowledge command.

        Commands:
            /knowledge search <query>            - Search knowledge base
            /knowledge concepts [domain]         - List concepts by domain
            /knowledge domains                   - List all domains
            /knowledge graph <concept>           - Show concept relationships
        """
        if not args:
            return self.format_error("Usage: /knowledge search <query>")

        command = args[0]

        if command == "search":
            return self._search_knowledge(args[1:], kwargs)
        elif command == "concepts":
            return self._list_concepts(args[1:], kwargs)
        elif command == "domains":
            return self._list_domains(kwargs)
        elif command == "graph":
            return self._show_graph(args[1:], kwargs)
        else:
            # Default to search if not a command
            return self._search_knowledge(args, kwargs)

    def _search_knowledge(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Search the knowledge base."""
        if not args:
            return self.format_error("Usage: /knowledge search <query>")

        query = " ".join(args)

        try:
            # Get knowledge store from context
            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                return self.format_error("Knowledge store not available")

            # Parse options
            top_k = int(kwargs.get('limit', kwargs.get('k', 10)))
            domain_filter = kwargs.get('domain')

            # Try to use knowledge retriever if available
            knowledge_retriever = self.felix_context.get('knowledge_retriever')

            if knowledge_retriever:
                # Use semantic search
                domains = [domain_filter] if domain_filter else None
                retrieval_context = knowledge_retriever.search(
                    query=query,
                    top_k=top_k,
                    domains=domains
                )

                if not retrieval_context.results:
                    return self.format_success(f"No knowledge found matching '{query}'")

                # Format results
                output_lines = [f"Knowledge search results for '{query}':", ""]

                for i, result in enumerate(retrieval_context.results, 1):
                    content = result.content
                    concept = content.get('concept', 'Unknown')
                    definition = content.get('definition', '')
                    domain = result.domain
                    relevance = result.relevance_score

                    output_lines.append(f"{i}. [{domain}] {concept} (relevance: {relevance:.2f})")
                    if definition:
                        output_lines.append(f"   {definition[:100]}{'...' if len(definition) > 100 else ''}")
                    output_lines.append("")

                output_lines.append(f"Found {len(retrieval_context.results)} results in {retrieval_context.processing_time:.2f}s")
                output_lines.append(f"Method: {retrieval_context.retrieval_method}")

            else:
                # Fallback to basic keyword search using correct API
                from src.memory.knowledge_store import KnowledgeQuery

                query_obj = KnowledgeQuery(
                    domains=[domain_filter] if domain_filter else None,
                    limit=100
                )
                entries = knowledge_store.retrieve_knowledge(query_obj)
                # Convert KnowledgeEntry objects to dicts
                entries = [entry.to_dict() for entry in entries]

                if not entries:
                    return self.format_success(f"No knowledge found matching '{query}'")

                # Simple keyword matching
                query_lower = query.lower()
                matches = []
                for entry in entries:
                    content_str = str(entry.get('content', '')).lower()
                    if query_lower in content_str:
                        matches.append(entry)

                matches = matches[:top_k]

                if not matches:
                    return self.format_success(f"No knowledge found matching '{query}'")

                # Format results
                output_lines = [f"Knowledge search results for '{query}':", ""]

                for i, entry in enumerate(matches, 1):
                    content = entry.get('content', {})
                    concept = content.get('concept', 'Unknown')
                    definition = content.get('definition', '')
                    domain = entry.get('domain', 'general')
                    confidence = entry.get('confidence_level', 'unknown')

                    output_lines.append(f"{i}. [{domain}] {concept} (confidence: {confidence})")
                    if definition:
                        output_lines.append(f"   {definition[:100]}{'...' if len(definition) > 100 else ''}")
                    output_lines.append("")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to search knowledge: {str(e)}")

    def _list_concepts(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """List concepts, optionally filtered by domain."""
        try:
            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                return self.format_error("Knowledge store not available")

            domain_filter = args[0] if args else None

            # Get entries using correct API
            from src.memory.knowledge_store import KnowledgeQuery

            query_obj = KnowledgeQuery(
                domains=[domain_filter] if domain_filter else None,
                limit=1000  # High limit for listing
            )
            entries = knowledge_store.retrieve_knowledge(query_obj)
            # Convert KnowledgeEntry objects to dicts
            entries = [entry.to_dict() for entry in entries]

            title = f"Concepts in domain '{domain_filter}'" if domain_filter else "All concepts"

            if not entries:
                return self.format_success(f"No concepts found")

            # Extract concepts
            concepts = []
            for entry in entries:
                content = entry.get('content', {})
                concept = content.get('concept', 'Unknown')
                domain = entry.get('domain', 'general')
                confidence = entry.get('confidence_level', 'unknown')
                concepts.append((concept, domain, confidence))

            # Sort by domain and concept
            concepts.sort(key=lambda x: (x[1], x[0]))

            # Format output
            output_lines = [f"{title}:", ""]

            current_domain = None
            for concept, domain, confidence in concepts[:50]:  # Limit to 50
                if domain != current_domain:
                    output_lines.append(f"\n[{domain}]")
                    current_domain = domain
                output_lines.append(f"  - {concept} (confidence: {confidence})")

            if len(concepts) > 50:
                output_lines.append(f"\n... and {len(concepts) - 50} more")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to list concepts: {str(e)}")

    def _list_domains(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List all knowledge domains."""
        try:
            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                return self.format_error("Knowledge store not available")

            # Get all entries and extract domains using correct API
            from src.memory.knowledge_store import KnowledgeQuery

            query_obj = KnowledgeQuery(limit=10000)  # High limit to get all
            entries = knowledge_store.retrieve_knowledge(query_obj)
            # Convert KnowledgeEntry objects to dicts
            entries = [entry.to_dict() for entry in entries]

            if not entries:
                return self.format_success("No knowledge in database")

            # Count entries per domain
            domain_counts = {}
            for entry in entries:
                domain = entry.get('domain', 'general')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            # Sort by count
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

            # Format output
            output_lines = ["Knowledge domains:", ""]

            for domain, count in sorted_domains:
                output_lines.append(f"  {domain}: {count} entries")

            output_lines.append("")
            output_lines.append(f"Total: {len(sorted_domains)} domains, {len(entries)} entries")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to list domains: {str(e)}")

    def _show_graph(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Show knowledge graph relationships for a concept."""
        if not args:
            return self.format_error("Usage: /knowledge graph <concept>")

        concept = " ".join(args)

        try:
            # Try to use knowledge graph builder if available
            from src.knowledge.graph_builder import KnowledgeGraphBuilder

            knowledge_store = self.felix_context.get('knowledge_store')
            if not knowledge_store:
                return self.format_error("Knowledge store not available")

            graph_builder = KnowledgeGraphBuilder(knowledge_store)

            # Get relationships for concept
            relationships = graph_builder.get_relationships_for_concept(concept)

            if not relationships:
                return self.format_success(f"No relationships found for concept '{concept}'")

            # Format output
            output_lines = [f"Knowledge graph for '{concept}':", ""]

            # Group by relationship type
            by_type = {}
            for rel in relationships:
                rel_type = rel.get('relationship_type', 'unknown')
                if rel_type not in by_type:
                    by_type[rel_type] = []
                by_type[rel_type].append(rel)

            for rel_type, rels in sorted(by_type.items()):
                output_lines.append(f"\n{rel_type.upper()}:")
                for rel in rels[:10]:  # Limit to 10 per type
                    source = rel.get('source_concept', '')
                    target = rel.get('target_concept', '')
                    strength = rel.get('strength', 0.0)

                    if source.lower() == concept.lower():
                        output_lines.append(f"  → {target} (strength: {strength:.2f})")
                    else:
                        output_lines.append(f"  ← {source} (strength: {strength:.2f})")

            return self.format_success("\n".join(output_lines))

        except ImportError:
            return self.format_error("Knowledge graph feature not available")
        except Exception as e:
            return self.format_error(f"Failed to retrieve graph: {str(e)}")
