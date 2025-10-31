"""
Knowledge Brain Demo

Demonstrates Felix's autonomous knowledge brain:
- Document ingestion
- Agentic comprehension
- Knowledge graph construction
- Semantic retrieval
"""

import os
from pathlib import Path

try:
    from src.knowledge import (
        DocumentReader,
        KnowledgeComprehensionEngine,
        KnowledgeGraphBuilder,
        KnowledgeRetriever
    )
    from src.memory.knowledge_store import KnowledgeStore
    from src.llm.router_adapter import create_router_adapter
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False
    print("⚠️  Knowledge Brain modules not available")


def demo_document_ingestion():
    """Show how to ingest a document."""

    print("\n" + "=" * 50)
    print("1. Document Ingestion")
    print("=" * 50)

    # Create a sample document
    sample_doc = """
    # Helical Geometry in Multi-Agent Systems

    Helical geometry provides a novel approach to agent coordination.
    Agents progress along a 3D spiral path from exploration to synthesis.

    ## Key Benefits
    - Adaptive behavior based on helix depth
    - Natural progression from broad to focused
    - Mathematical foundation for agent positioning

    ## Parameters
    - Top radius: Wide exploration (3.0)
    - Bottom radius: Narrow synthesis (0.5)
    - Height: Progression depth (8.0)
    - Turns: Spiral complexity (2)
    """

    # Save to temp file
    temp_file = "temp_knowledge_doc.md"
    with open(temp_file, 'w') as f:
        f.write(sample_doc)

    print(f"✓ Created sample document: {temp_file}")

    # Ingest document
    reader = DocumentReader()
    content = reader.read_document(temp_file)

    print(f"✓ Read document: {len(content)} characters")
    print(f"  First 100 chars: {content[:100]}...")

    # Cleanup
    os.unlink(temp_file)

    return content


def demo_agentic_comprehension(content):
    """Show agentic document comprehension."""

    print("\n" + "=" * 50)
    print("2. Agentic Comprehension")
    print("=" * 50)

    try:
        llm_client = create_router_adapter()

        # Note: This requires setting up helix, central_post, etc.
        # Simplified for demo
        print("💡 Agentic comprehension uses Research, Analysis, and Critic agents")
        print("   to deeply understand documents (not just chunk them)")
        print("\n  Features:")
        print("  - Research agent: Extracts key concepts")
        print("  - Analysis agent: Identifies relationships")
        print("  - Critic agent: Validates understanding")

    except Exception as e:
        print(f"⚠️  Could not initialize comprehension: {e}")


def demo_knowledge_graph():
    """Show knowledge graph construction."""

    print("\n" + "=" * 50)
    print("3. Knowledge Graph")
    print("=" * 50)

    print("Knowledge graph builder discovers relationships through:")
    print("  ✓ Explicit mentions (e.g., 'X relates to Y')")
    print("  ✓ Embedding similarity (threshold: 0.75)")
    print("  ✓ Co-occurrence (within 5-chunk window)")
    print("\nExample graph:")
    print("  [Helical Geometry] --relates-to--> [Agent Coordination]")
    print("  [Top Radius] --parameter-of--> [Helical Geometry]")
    print("  [Exploration] --contrasts-with--> [Synthesis]")


def demo_semantic_retrieval():
    """Show semantic retrieval with meta-learning."""

    print("\n" + "=" * 50)
    print("4. Semantic Retrieval")
    print("=" * 50)

    print("Retrieval uses 3-tier approach:")
    print("  1. LM Studio embeddings (768-dim) - if available")
    print("  2. TF-IDF vectors - pure Python fallback")
    print("  3. SQLite FTS5 (BM25) - always works")
    print("\nMeta-learning boost:")
    print("  Tracks which knowledge helped which workflows")
    print("  Boosts relevance scores for historically useful knowledge")


def demo_knowledge_stats():
    """Show actual knowledge database stats."""

    print("\n" + "=" * 50)
    print("5. Current Knowledge Stats")
    print("=" * 50)

    try:
        knowledge_store = KnowledgeStore()

        # Get counts
        import sqlite3
        conn = sqlite3.connect("felix_knowledge.db")
        cursor = conn.cursor()

        # Knowledge entries
        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        entry_count = cursor.fetchone()[0]

        # Documents
        cursor.execute("SELECT COUNT(*) FROM document_sources")
        doc_count = cursor.fetchone()[0]

        # Relationships
        cursor.execute("SELECT COUNT(*) FROM knowledge_relationships")
        rel_count = cursor.fetchone()[0]

        conn.close()

        print(f"📚 Knowledge entries: {entry_count}")
        print(f"📄 Documents processed: {doc_count}")
        print(f"🔗 Relationships: {rel_count}")

    except Exception as e:
        print(f"⚠️  Could not read database: {e}")
        print("   (Database may not exist yet - run Felix GUI first)")


def main():
    """Run Knowledge Brain demo."""

    print("Felix Knowledge Brain - Demonstration")
    print("=" * 50)

    if not KNOWLEDGE_AVAILABLE:
        print("\n❌ Knowledge Brain modules not available")
        print("   Make sure all dependencies are installed:")
        print("   pip install PyPDF2 watchdog")
        return

    try:
        # Run demos
        content = demo_document_ingestion()
        demo_agentic_comprehension(content)
        demo_knowledge_graph()
        demo_semantic_retrieval()
        demo_knowledge_stats()

        print("\n" + "=" * 50)
        print("✓ Knowledge Brain demo complete!")
        print("\nTo enable Knowledge Brain in workflows:")
        print("  1. Edit config/llm.yaml")
        print("  2. Set: knowledge_brain.enable_knowledge_brain: true")
        print("  3. Set: knowledge_brain.knowledge_watch_dirs: ['./documents']")
        print("  4. Add documents to ./documents/ folder")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
