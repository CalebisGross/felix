#!/usr/bin/env python3
"""
Knowledge Brain API Client Example

Demonstrates how to use the Felix Knowledge Brain API for:
- Document ingestion and processing
- Semantic search
- Knowledge graph operations
- Daemon control
- Concept browsing

Requirements:
    pip install httpx

Usage:
    # Start Felix API server first with Knowledge Brain enabled
    export FELIX_ENABLE_KNOWLEDGE_BRAIN=true
    python -m uvicorn src.api.main:app --port 8000

    # Run this client
    python examples/api_examples/knowledge_brain_client.py
"""

import sys
import time
from typing import Optional, List, Dict, Any

try:
    import httpx
except ImportError:
    print("Error: httpx not installed")
    print("Please install: pip install httpx")
    sys.exit(1)


# Configuration
API_URL = "http://localhost:8000"
API_KEY = "your-secret-api-key-here"  # Change if using authentication


class KnowledgeBrainClient:
    """
    Client for Felix Knowledge Brain API.
    """

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make HTTP request."""
        url = f"{self.api_url}{endpoint}"
        with httpx.Client(timeout=60.0) as client:
            response = client.request(method, url, headers=self.headers, **kwargs)
            return response

    # ========================================================================
    # Document Methods
    # ========================================================================

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document.

        Args:
            file_path: Path to document file

        Returns:
            Ingestion result with document_id and metadata
        """
        print(f"\n📄 Ingesting document: {file_path}")
        response = self._request(
            "POST",
            "/api/v1/knowledge/documents/ingest",
            json={"file_path": file_path, "process_immediately": True}
        )

        if response.status_code == 202:
            result = response.json()
            print(f"✅ Document ingested: {result['document_id']}")
            print(f"   Chunks: {result['chunks_count']}")
            print(f"   Status: {result['status']}")
            return result
        else:
            print(f"❌ Ingestion failed: {response.status_code}")
            print(response.json())
            return {}

    def batch_ingest_documents(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Batch ingest documents from directory.

        Args:
            directory_path: Directory containing documents
            recursive: Process subdirectories

        Returns:
            Batch processing result
        """
        print(f"\n📚 Batch ingesting from: {directory_path}")
        response = self._request(
            "POST",
            "/api/v1/knowledge/documents/batch",
            json={
                "directory_path": directory_path,
                "recursive": recursive,
                "file_patterns": ["*.pdf", "*.txt", "*.md"]
            }
        )

        if response.status_code == 202:
            result = response.json()
            print(f"✅ Batch processing complete:")
            print(f"   Total: {result['total_files']}")
            print(f"   Processed: {result['processed']}")
            print(f"   Failed: {result['failed']}")
            print(f"   Time: {result['processing_time_seconds']:.2f}s")
            return result
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(response.json())
            return {}

    def list_documents(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all documents.

        Args:
            status_filter: Optional filter (processing/complete/failed)

        Returns:
            List of documents
        """
        print(f"\n📋 Listing documents...")
        params = {}
        if status_filter:
            params["status_filter"] = status_filter

        response = self._request("GET", "/api/v1/knowledge/documents", params=params)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total']} documents")
            for doc in result['documents'][:5]:  # Show first 5
                print(f"   - {doc['file_name']} ({doc['status']}) - {doc['chunks_count']} chunks")
            if result['total'] > 5:
                print(f"   ... and {result['total'] - 5} more")
            return result['documents']
        else:
            print(f"❌ List failed: {response.status_code}")
            return []

    def get_document_details(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document details.

        Args:
            document_id: Document ID

        Returns:
            Document details
        """
        print(f"\n🔍 Getting document details: {document_id}")
        response = self._request("GET", f"/api/v1/knowledge/documents/{document_id}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document: {result['metadata']['file_name']}")
            print(f"   Type: {result['metadata']['file_type']}")
            print(f"   Size: {result['metadata']['file_size']} bytes")
            print(f"   Chunks: {result['chunks_count']}")
            print(f"   Concepts: {result['concepts_extracted']}")
            return result
        else:
            print(f"❌ Get details failed: {response.status_code}")
            return None

    # ========================================================================
    # Search Methods
    # ========================================================================

    def search(self, query: str, top_k: int = 10, domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Semantic search across knowledge base.

        Args:
            query: Search query
            top_k: Number of results
            domains: Optional domain filter

        Returns:
            List of search results
        """
        print(f"\n🔎 Searching: {query}")
        request_data = {
            "query": query,
            "top_k": top_k
        }
        if domains:
            request_data["domains"] = domains

        response = self._request("POST", "/api/v1/knowledge/search", json=request_data)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total_results']} results ({result['retrieval_method']})")
            print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
            for i, item in enumerate(result['results'][:3], 1):  # Show top 3
                print(f"\n   {i}. Score: {item['relevance_score']:.3f}")
                print(f"      {item['content'][:150]}...")
            return result['results']
        else:
            print(f"❌ Search failed: {response.status_code}")
            return []

    def augment_context(self, task_description: str, max_concepts: int = 10) -> Optional[str]:
        """
        Get augmented context for a task.

        Args:
            task_description: Task description
            max_concepts: Max concepts to include

        Returns:
            Augmented context string
        """
        print(f"\n🔄 Augmenting context for task...")
        response = self._request(
            "POST",
            "/api/v1/knowledge/search/augment",
            json={
                "task_description": task_description,
                "max_concepts": max_concepts
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Context augmented with {result['concepts_used']} concepts")
            print(f"   Method: {result['retrieval_method']}")
            return result['augmented_context']
        else:
            print(f"❌ Augmentation failed: {response.status_code}")
            return None

    # ========================================================================
    # Knowledge Graph Methods
    # ========================================================================

    def build_graph(self, document_id: Optional[str] = None, max_documents: Optional[int] = None) -> Dict[str, Any]:
        """
        Build knowledge graph.

        Args:
            document_id: Build for specific document, or None for global
            max_documents: Max documents for global graph

        Returns:
            Build result with statistics
        """
        if document_id:
            print(f"\n🕸️  Building graph for document: {document_id}")
        else:
            print(f"\n🕸️  Building global knowledge graph...")

        response = self._request(
            "POST",
            "/api/v1/knowledge/graph/build",
            json={
                "document_id": document_id,
                "max_documents": max_documents,
                "similarity_threshold": 0.75
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Graph built:")
            print(f"   Relationships: {result['relationships_created']}")
            print(f"   Concepts: {result['concepts_processed']}")
            if result.get('documents_processed'):
                print(f"   Documents: {result['documents_processed']}")
            if result.get('entities_linked'):
                print(f"   Entities linked: {result['entities_linked']}")
            print(f"   Time: {result['processing_time_seconds']:.2f}s")
            return result
        else:
            print(f"❌ Graph build failed: {response.status_code}")
            return {}

    def get_relationships(self, concept_id: str, min_strength: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get relationships for a concept.

        Args:
            concept_id: Knowledge ID
            min_strength: Minimum relationship strength

        Returns:
            List of relationships
        """
        print(f"\n🔗 Getting relationships for: {concept_id}")
        response = self._request(
            "POST",
            "/api/v1/knowledge/graph/relationships",
            json={
                "concept_id": concept_id,
                "max_depth": 1,
                "min_strength": min_strength
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total_relationships']} relationships")
            for rel in result['relationships'][:3]:  # Show first 3
                print(f"   - {rel['relationship_type']} (strength: {rel['strength']:.2f})")
                print(f"     {rel['target_content'][:100]}...")
            return result['relationships']
        else:
            print(f"❌ Get relationships failed: {response.status_code}")
            return []

    def get_graph_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get knowledge graph statistics.

        Returns:
            Graph statistics
        """
        print(f"\n📊 Getting graph statistics...")
        response = self._request("GET", "/api/v1/knowledge/graph/statistics")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Graph Statistics:")
            print(f"   Nodes: {result['total_nodes']}")
            print(f"   Relationships: {result['total_relationships']}")
            print(f"   Avg degree: {result['average_degree']:.2f}")
            print(f"   Documents: {result['documents_covered']}")
            return result
        else:
            print(f"❌ Get statistics failed: {response.status_code}")
            return None

    # ========================================================================
    # Daemon Methods
    # ========================================================================

    def start_daemon(self) -> bool:
        """
        Start knowledge daemon.

        Returns:
            True if started successfully
        """
        print(f"\n🤖 Starting knowledge daemon...")
        response = self._request("POST", "/api/v1/knowledge/daemon/start")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        elif response.status_code == 409:
            print(f"ℹ️  Daemon already running")
            return True
        else:
            print(f"❌ Start failed: {response.status_code}")
            return False

    def stop_daemon(self) -> bool:
        """
        Stop knowledge daemon.

        Returns:
            True if stopped successfully
        """
        print(f"\n🛑 Stopping knowledge daemon...")
        response = self._request("POST", "/api/v1/knowledge/daemon/stop")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            if 'final_stats' in result:
                print(f"   Documents processed: {result['final_stats']['documents_processed']}")
            return True
        else:
            print(f"❌ Stop failed: {response.status_code}")
            return False

    def get_daemon_status(self) -> Optional[Dict[str, Any]]:
        """
        Get daemon status.

        Returns:
            Daemon status
        """
        print(f"\n📡 Getting daemon status...")
        response = self._request("GET", "/api/v1/knowledge/daemon/status")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Daemon Status:")
            print(f"   Running: {result['running']}")
            if result['running']:
                print(f"   Batch processor: {result['batch_processor_active']}")
                print(f"   Refiner: {result['refiner_active']}")
                print(f"   File watcher: {result['file_watcher_active']}")
                print(f"   Processed: {result['documents_processed']}")
                print(f"   Pending: {result['documents_pending']}")
                print(f"   Uptime: {result['uptime_seconds']:.0f}s")
            return result
        else:
            print(f"❌ Get status failed: {response.status_code}")
            return None

    # ========================================================================
    # Concept Methods
    # ========================================================================

    def list_concepts(self, domain: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List concepts.

        Args:
            domain: Optional domain filter
            limit: Max results

        Returns:
            List of concepts
        """
        print(f"\n💡 Listing concepts...")
        request_data = {"limit": limit, "offset": 0}
        if domain:
            request_data["domain"] = domain

        response = self._request("POST", "/api/v1/knowledge/concepts", json=request_data)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total']} concepts (showing {len(result['concepts'])})")
            for concept in result['concepts'][:5]:
                print(f"   - {concept['concept_name']}")
                print(f"     {concept['definition'][:100]}...")
            return result['concepts']
        else:
            print(f"❌ List concepts failed: {response.status_code}")
            return []

    def get_concept_details(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get concept details.

        Args:
            knowledge_id: Knowledge entry ID

        Returns:
            Concept details
        """
        print(f"\n🔍 Getting concept details: {knowledge_id}")
        response = self._request("GET", f"/api/v1/knowledge/concepts/{knowledge_id}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Concept: {result['concept_name']}")
            print(f"   Domain: {result.get('domain', 'N/A')}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Access count: {result['access_count']}")
            print(f"   Related: {len(result['related_concept_ids'])}")
            return result
        else:
            print(f"❌ Get concept failed: {response.status_code}")
            return None

    def get_related_concepts(self, knowledge_id: str) -> List[Dict[str, Any]]:
        """
        Get related concepts.

        Args:
            knowledge_id: Knowledge entry ID

        Returns:
            List of related concepts
        """
        print(f"\n🔗 Getting related concepts for: {knowledge_id}")
        response = self._request("GET", f"/api/v1/knowledge/concepts/{knowledge_id}/related")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Found {result['total']} related concepts")
            for related in result['related_concepts'][:3]:
                print(f"   - {related['concept_name']} ({related['relationship_type']})")
            return result['related_concepts']
        else:
            print(f"❌ Get related failed: {response.status_code}")
            return []


# ============================================================================
# Demo Workflows
# ============================================================================

def demo_document_workflow(client: KnowledgeBrainClient):
    """Demonstrate document ingestion and listing."""
    print("\n" + "=" * 60)
    print("DEMO: Document Ingestion Workflow")
    print("=" * 60)

    # List existing documents
    client.list_documents()

    # Note: Actual document ingestion requires valid file paths
    print("\nℹ️  To ingest documents, use:")
    print("   client.ingest_document('/path/to/document.pdf')")
    print("   client.batch_ingest_documents('/path/to/documents')")


def demo_search_workflow(client: KnowledgeBrainClient):
    """Demonstrate semantic search."""
    print("\n" + "=" * 60)
    print("DEMO: Semantic Search Workflow")
    print("=" * 60)

    # Search for concepts
    results = client.search("machine learning algorithms", top_k=5)

    if results:
        # Get details of first result
        first_result = results[0]
        client.get_concept_details(first_result['knowledge_id'])

        # Get related concepts
        client.get_related_concepts(first_result['knowledge_id'])


def demo_graph_workflow(client: KnowledgeBrainClient):
    """Demonstrate knowledge graph operations."""
    print("\n" + "=" * 60)
    print("DEMO: Knowledge Graph Workflow")
    print("=" * 60)

    # Get graph statistics
    client.get_graph_statistics()

    # Note: Building graphs requires documents
    print("\nℹ️  To build knowledge graph:")
    print("   client.build_graph()  # Global graph")
    print("   client.build_graph(document_id='doc_abc123')  # Single document")


def demo_daemon_workflow(client: KnowledgeBrainClient):
    """Demonstrate daemon control."""
    print("\n" + "=" * 60)
    print("DEMO: Daemon Control Workflow")
    print("=" * 60)

    # Get daemon status
    status = client.get_daemon_status()

    if status and not status['running']:
        print("\nℹ️  Daemon is not running. To start:")
        print("   client.start_daemon()")
        print("\n   Then check status periodically:")
        print("   client.get_daemon_status()")


def main():
    """Main demo."""
    print("\n" + "=" * 60)
    print("Felix Knowledge Brain API Client Demo")
    print("=" * 60)

    # Create client
    api_key = API_KEY if API_KEY != "your-secret-api-key-here" else None
    client = KnowledgeBrainClient(API_URL, api_key)

    try:
        # Run demos
        demo_document_workflow(client)
        time.sleep(1)

        demo_search_workflow(client)
        time.sleep(1)

        demo_graph_workflow(client)
        time.sleep(1)

        demo_daemon_workflow(client)

        print("\n" + "=" * 60)
        print("✅ Demo Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Enable Knowledge Brain: export FELIX_ENABLE_KNOWLEDGE_BRAIN=true")
        print("2. Start API: python -m uvicorn src.api.main:app --port 8000")
        print("3. Ingest documents using client.ingest_document()")
        print("4. Explore the knowledge base with search and graph operations")
        print("\nAPI Documentation: http://localhost:8000/docs")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
