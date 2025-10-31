#!/usr/bin/env python3
"""
Felix Memory & History API - Example Python Client

This script demonstrates how to use the Memory & History API to:
1. Query task patterns and get strategy recommendations
2. Record task executions for pattern learning
3. Browse workflow execution history and conversation threads
4. Store and retrieve knowledge entries with meta-learning
5. Compress large contexts using various strategies

Prerequisites:
- Felix API server running (python -m uvicorn src.api.main:app --reload --port 8000)
- httpx installed (pip install httpx)

Usage:
    python memory_history_client.py [--api-key YOUR_KEY]
"""

import sys
import json
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import httpx
except ImportError:
    print("Error: httpx is required")
    print("Install with: pip install httpx")
    sys.exit(1)


class MemoryHistoryClient:
    """Client for Felix Memory & History API."""

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize Memory & History API client.

        Args:
            api_url: Base URL of Felix API (e.g., "http://localhost:8000")
            api_key: Optional API key for authentication
        """
        self.api_url = api_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.api_url}{endpoint}"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    **kwargs
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            try:
                error_json = json.loads(error_body)
                raise Exception(f"API Error ({e.response.status_code}): {error_json.get('message', error_body)}")
            except json.JSONDecodeError:
                raise Exception(f"API Error ({e.response.status_code}): {error_body}")

        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

    # ========================================================================
    # Task Memory API
    # ========================================================================

    def list_task_patterns(
        self,
        task_types: Optional[List[str]] = None,
        complexity_levels: Optional[List[str]] = None,
        min_success_rate: Optional[float] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        List task patterns with optional filtering.

        Args:
            task_types: Filter by task types (e.g., ["research", "analysis"])
            complexity_levels: Filter by complexity (e.g., ["complex"])
            min_success_rate: Minimum success rate (0.0-1.0)
            limit: Maximum results

        Returns:
            Dictionary with patterns list and total count
        """
        params = {"limit": limit}

        if task_types:
            params["task_types"] = task_types
        if complexity_levels:
            params["complexity_levels"] = complexity_levels
        if min_success_rate is not None:
            params["min_success_rate"] = min_success_rate

        return self._request("GET", "/api/v1/memory/tasks/patterns", params=params)

    def get_task_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """Get specific task pattern by ID."""
        return self._request("GET", f"/api/v1/memory/tasks/patterns/{pattern_id}")

    def list_task_executions(
        self,
        task_types: Optional[List[str]] = None,
        outcomes: Optional[List[str]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        List task execution records.

        Args:
            task_types: Filter by task types
            outcomes: Filter by outcomes (e.g., ["success", "failure"])
            limit: Maximum results

        Returns:
            Dictionary with executions list and total count
        """
        params = {"limit": limit}

        if task_types:
            params["task_types"] = task_types
        if outcomes:
            params["outcomes"] = outcomes

        return self._request("GET", "/api/v1/memory/tasks/executions", params=params)

    def get_task_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get specific task execution by ID."""
        return self._request("GET", f"/api/v1/memory/tasks/executions/{execution_id}")

    def record_task_execution(
        self,
        task_description: str,
        task_type: str,
        complexity: str,
        outcome: str,
        duration: float,
        agents_used: Optional[List[str]] = None,
        strategies_used: Optional[List[str]] = None,
        context_size: int = 0,
        error_messages: Optional[List[str]] = None,
        success_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record a new task execution.

        This creates an execution record and updates relevant patterns.

        Args:
            task_description: Description of the task
            task_type: Type (e.g., "research", "analysis")
            complexity: Complexity level (simple/moderate/complex/very_complex)
            outcome: Outcome (success/partial_success/failure/timeout/error)
            duration: Duration in seconds
            agents_used: List of agent IDs used
            strategies_used: List of strategies applied
            context_size: Size of context used
            error_messages: List of errors encountered
            success_metrics: Custom success metrics

        Returns:
            Response with execution_id and patterns matched
        """
        payload = {
            "task_description": task_description,
            "task_type": task_type,
            "complexity": complexity,
            "outcome": outcome,
            "duration": duration,
            "agents_used": agents_used or [],
            "strategies_used": strategies_used or [],
            "context_size": context_size,
            "error_messages": error_messages or [],
            "success_metrics": success_metrics or {}
        }

        return self._request("POST", "/api/v1/memory/tasks/executions", json=payload)

    def recommend_strategy(
        self,
        task_description: str,
        task_type: str,
        complexity: str
    ) -> Dict[str, Any]:
        """
        Get strategy recommendation for a task.

        Uses historical patterns to recommend optimal strategies, agents,
        and estimate success probability.

        Args:
            task_description: Description of the task
            task_type: Type (e.g., "research")
            complexity: Complexity level (simple/moderate/complex/very_complex)

        Returns:
            Recommendations including strategies, agents, duration, success probability
        """
        payload = {
            "task_description": task_description,
            "task_type": task_type,
            "complexity": complexity
        }

        return self._request("POST", "/api/v1/memory/tasks/recommend-strategy", json=payload)

    def get_task_memory_summary(self) -> Dict[str, Any]:
        """Get task memory statistics and summary."""
        return self._request("GET", "/api/v1/memory/tasks/summary")

    # ========================================================================
    # Workflow History API
    # ========================================================================

    def list_workflows(
        self,
        status: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List workflow execution records.

        Args:
            status: Filter by status (e.g., "completed", "failed")
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
            search_query: Search in task_input and synthesis
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            Dictionary with workflows list, total, offset, limit
        """
        params = {"limit": limit, "offset": offset}

        if status:
            params["status_filter"] = status
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        if search_query:
            params["search_query"] = search_query

        return self._request("GET", "/api/v1/memory/workflows/", params=params)

    def get_workflow(self, workflow_id: int) -> Dict[str, Any]:
        """Get specific workflow by ID."""
        return self._request("GET", f"/api/v1/memory/workflows/{workflow_id}")

    def get_conversation_thread(self, workflow_id: int) -> Dict[str, Any]:
        """
        Get complete conversation thread for a workflow.

        Returns the root workflow and all related workflows in the thread.

        Args:
            workflow_id: Workflow ID to get thread for

        Returns:
            Dictionary with thread_id, root_workflow, child_workflows, total, depth
        """
        return self._request("GET", f"/api/v1/memory/workflows/{workflow_id}/thread")

    def save_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a new workflow to history.

        Args:
            workflow_data: Workflow data (task_input, status, synthesis, etc.)

        Returns:
            Saved workflow with assigned workflow_id
        """
        return self._request("POST", "/api/v1/memory/workflows/", json=workflow_data)

    def delete_workflow(self, workflow_id: int) -> Dict[str, Any]:
        """Delete a workflow from history."""
        return self._request("DELETE", f"/api/v1/memory/workflows/{workflow_id}")

    def search_workflows(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Full-text search across workflow task inputs and synthesis results.

        Args:
            query: Search query
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            Dictionary with matching workflows
        """
        params = {"query": query, "limit": limit, "offset": offset}
        return self._request("GET", "/api/v1/memory/workflows/search/", params=params)

    def get_workflow_analytics(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get workflow analytics and performance metrics.

        Args:
            from_date: Start date (ISO format)
            to_date: End date (ISO format)

        Returns:
            Analytics including completion rates, average metrics, distributions
        """
        params = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return self._request("GET", "/api/v1/memory/workflows/analytics/summary", params=params)

    # ========================================================================
    # Knowledge Memory API
    # ========================================================================

    def retrieve_knowledge(
        self,
        knowledge_types: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Retrieve knowledge entries with filtering and meta-learning boost.

        Args:
            knowledge_types: Filter by types (e.g., ["agent_insight"])
            domains: Filter by domains
            tags: Filter by tags
            min_confidence: Minimum confidence (low/medium/high/verified)
            task_type: Task type for meta-learning boost
            limit: Maximum results

        Returns:
            Dictionary with knowledge entries
        """
        params = {"limit": limit}

        if knowledge_types:
            params["knowledge_types"] = knowledge_types
        if domains:
            params["domains"] = domains
        if tags:
            params["tags"] = tags
        if min_confidence:
            params["min_confidence"] = min_confidence
        if task_type:
            params["task_type"] = task_type

        return self._request("GET", "/api/v1/memory/knowledge/", params=params)

    def get_knowledge_entry(self, knowledge_id: str) -> Dict[str, Any]:
        """Get specific knowledge entry by ID."""
        return self._request("GET", f"/api/v1/memory/knowledge/{knowledge_id}")

    def store_knowledge(
        self,
        knowledge_type: str,
        content: Dict[str, Any],
        confidence_level: str,
        source_agent: str,
        domain: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Store a new knowledge entry.

        Args:
            knowledge_type: Type (task_result/agent_insight/pattern_recognition/etc.)
            content: Content dictionary
            confidence_level: Confidence (low/medium/high/verified)
            source_agent: Agent that created this knowledge
            domain: Domain classification
            tags: Optional tags

        Returns:
            Response with knowledge_id and stored/updated status
        """
        payload = {
            "knowledge_type": knowledge_type,
            "content": content,
            "confidence_level": confidence_level,
            "source_agent": source_agent,
            "domain": domain,
            "tags": tags or []
        }

        return self._request("POST", "/api/v1/memory/knowledge/", json=payload)

    def record_knowledge_usage(
        self,
        knowledge_id: str,
        workflow_id: str,
        task_type: str,
        task_complexity: str,
        useful_score: float,
        retrieval_method: str
    ) -> Dict[str, Any]:
        """
        Record knowledge usage for meta-learning.

        Args:
            knowledge_id: Knowledge entry ID
            workflow_id: Workflow that used this knowledge
            task_type: Type of task
            task_complexity: Complexity level
            useful_score: How useful was it (0.0-1.0)
            retrieval_method: Method used (sql/semantic/hybrid)

        Returns:
            Confirmation of usage recording
        """
        payload = {
            "knowledge_id": knowledge_id,
            "workflow_id": workflow_id,
            "task_type": task_type,
            "task_complexity": task_complexity,
            "useful_score": useful_score,
            "retrieval_method": retrieval_method
        }

        return self._request("POST", f"/api/v1/memory/knowledge/{knowledge_id}/usage", json=payload)

    def update_knowledge_success_rate(
        self,
        knowledge_id: str,
        new_success_rate: float
    ) -> Dict[str, Any]:
        """Update the success rate of a knowledge entry."""
        payload = {"new_success_rate": new_success_rate}
        return self._request("PATCH", f"/api/v1/memory/knowledge/{knowledge_id}/success-rate", json=payload)

    def get_related_knowledge(
        self,
        knowledge_id: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Get knowledge entries related to the specified entry.

        Args:
            knowledge_id: Knowledge entry ID
            max_results: Maximum related entries

        Returns:
            Dictionary with relationships
        """
        params = {"max_results": max_results}
        return self._request("GET", f"/api/v1/memory/knowledge/{knowledge_id}/related", params=params)

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get knowledge memory statistics and summary."""
        return self._request("GET", "/api/v1/memory/knowledge/summary/stats")

    # ========================================================================
    # Context Compression API
    # ========================================================================

    def compress_context(
        self,
        context: Dict[str, Any],
        strategy: str = "progressive_refinement",
        level: str = "moderate",
        preserve_keywords: Optional[List[str]] = None,
        topic_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compress context using specified strategy and level.

        Args:
            context: Context to compress
            strategy: Strategy (extractive_summary/abstractive_summary/keyword_extraction/
                     hierarchical_summary/relevance_filtering/progressive_refinement)
            level: Level (light/moderate/heavy/extreme)
            preserve_keywords: Keywords to preserve
            topic_keywords: Topic keywords for relevance filtering

        Returns:
            Compressed context with metrics
        """
        payload = {
            "context": context,
            "strategy": strategy,
            "level": level,
            "preserve_keywords": preserve_keywords or [],
            "preserve_structure": True
        }

        if topic_keywords:
            payload["topic_keywords"] = topic_keywords

        return self._request("POST", "/api/v1/memory/compression/compress", json=payload)

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression system statistics and configuration."""
        return self._request("GET", "/api/v1/memory/compression/stats")


# ============================================================================
# Demo Functions
# ============================================================================

def demo_task_memory(client: MemoryHistoryClient):
    """Demonstrate Task Memory API."""
    print("\n" + "=" * 70)
    print("DEMO: Task Memory API")
    print("=" * 70)

    # 1. Get strategy recommendation
    print("\n1. Getting strategy recommendation for a complex research task...")
    recommendation = client.recommend_strategy(
        task_description="Research the latest developments in quantum computing and AI integration",
        task_type="research",
        complexity="complex"
    )

    print(f"   Recommended strategies: {', '.join(recommendation['recommended_strategies'])}")
    print(f"   Recommended agents: {', '.join(recommendation['recommended_agents'])}")
    print(f"   Success probability: {recommendation['success_probability']:.1%}")
    print(f"   Confidence: {recommendation['confidence']:.1%}")

    # 2. Record a task execution
    print("\n2. Recording a successful task execution...")
    execution_response = client.record_task_execution(
        task_description="Research quantum computing fundamentals for technical report",
        task_type="research",
        complexity="complex",
        outcome="success",
        duration=42.5,
        agents_used=["research_001", "analysis_002", "critic_003"],
        strategies_used=["web-search", "multi-agent", "knowledge-augment"],
        context_size=3500,
        success_metrics={"quality": 0.92, "completeness": 0.88}
    )

    print(f"   ✓ Execution recorded: {execution_response['execution_id']}")
    print(f"   Patterns matched: {len(execution_response['patterns_matched'])}")

    # 3. List task patterns
    print("\n3. Listing task patterns for research tasks...")
    patterns = client.list_task_patterns(
        task_types=["research"],
        min_success_rate=0.8,
        limit=5
    )

    print(f"   Found {patterns['total']} patterns")
    for pattern in patterns['patterns'][:3]:
        print(f"   - {pattern['pattern_id']}: Success rate {pattern['success_rate']:.1%}, "
              f"Used {pattern['usage_count']} times")

    # 4. Get task memory summary
    print("\n4. Getting task memory summary...")
    summary = client.get_task_memory_summary()

    print(f"   Total patterns: {summary['total_patterns']}")
    print(f"   Total executions: {summary['total_executions']}")
    print(f"   Average success rate: {summary['average_success_rate']:.1%}")


def demo_workflow_history(client: MemoryHistoryClient):
    """Demonstrate Workflow History API."""
    print("\n" + "=" * 70)
    print("DEMO: Workflow History API")
    print("=" * 70)

    # 1. List recent workflows
    print("\n1. Listing recent workflows...")
    workflows = client.list_workflows(limit=5)

    print(f"   Found {workflows['total']} workflows")
    for workflow in workflows['workflows'][:3]:
        print(f"   - Workflow {workflow['workflow_id']}: {workflow['status']}, "
              f"{workflow['agents_count']} agents, {workflow['processing_time']:.1f}s")

    # 2. Get workflow analytics
    print("\n2. Getting workflow analytics...")
    analytics = client.get_workflow_analytics()

    print(f"   Total workflows: {analytics['total_workflows']}")
    print(f"   Completed: {analytics['completed_workflows']}")
    print(f"   Failed: {analytics['failed_workflows']}")
    print(f"   Average confidence: {analytics['average_confidence']:.1%}")
    print(f"   Average processing time: {analytics['average_processing_time']:.1f}s")

    # 3. Search workflows
    if workflows['workflows']:
        print("\n3. Searching workflows...")
        search_results = client.search_workflows(query="quantum", limit=3)
        print(f"   Found {search_results['total']} workflows matching 'quantum'")

        # 4. Get conversation thread (if any workflow has a parent)
        for workflow in workflows['workflows']:
            if workflow.get('conversation_thread_id'):
                print(f"\n4. Getting conversation thread for workflow {workflow['workflow_id']}...")
                try:
                    thread = client.get_conversation_thread(workflow['workflow_id'])
                    print(f"   Thread {thread['thread_id']}: {thread['total_workflows']} workflows, "
                          f"depth {thread['thread_depth']}")
                    break
                except:
                    pass


def demo_knowledge_memory(client: MemoryHistoryClient):
    """Demonstrate Knowledge Memory API."""
    print("\n" + "=" * 70)
    print("DEMO: Knowledge Memory API")
    print("=" * 70)

    # 1. Store knowledge entry
    print("\n1. Storing a knowledge entry...")
    store_response = client.store_knowledge(
        knowledge_type="domain_expertise",
        content={
            "concept": "quantum entanglement",
            "definition": "A phenomenon where quantum states of particles become correlated",
            "applications": ["quantum computing", "quantum cryptography"]
        },
        confidence_level="high",
        source_agent="research_001",
        domain="physics",
        tags=["quantum", "physics", "advanced"]
    )

    print(f"   ✓ Knowledge stored: {store_response['knowledge_id']}")
    print(f"   {'Updated' if store_response['updated'] else 'Created'}")

    # 2. Retrieve knowledge
    print("\n2. Retrieving knowledge entries...")
    knowledge = client.retrieve_knowledge(
        domains=["physics", "computer_science"],
        min_confidence="high",
        limit=5
    )

    print(f"   Found {knowledge['total']} entries")
    for entry in knowledge['entries'][:3]:
        print(f"   - {entry['knowledge_id']}: {entry['domain']}, "
              f"confidence={entry['confidence_level']}, "
              f"success_rate={entry['success_rate']:.1%}")

    # 3. Get knowledge summary
    print("\n3. Getting knowledge summary...")
    summary = client.get_knowledge_summary()

    print(f"   Total entries: {summary['total_entries']}")
    print(f"   Average success rate: {summary['average_success_rate']:.1%}")
    print(f"   Total accesses: {summary['total_access_count']}")


def demo_compression(client: MemoryHistoryClient):
    """Demonstrate Context Compression API."""
    print("\n" + "=" * 70)
    print("DEMO: Context Compression API")
    print("=" * 70)

    # 1. Compress context
    print("\n1. Compressing a large context...")
    test_context = {
        "agent_outputs": [
            "Research Agent: Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform computation. Key applications include cryptography, optimization, and simulation of quantum systems.",
            "Analysis Agent: The current state shows significant progress in error correction and qubit stability. IBM and Google lead in hardware development.",
            "Critic Agent: While promising, practical applications remain limited by decoherence and error rates. More research needed in error correction."
        ],
        "synthesis": "Quantum computing shows promise but faces technical challenges"
    }

    compressed = client.compress_context(
        context=test_context,
        strategy="progressive_refinement",
        level="moderate",
        preserve_keywords=["quantum", "error correction"]
    )

    print(f"   Original size: {compressed['original_size']} chars")
    print(f"   Compressed size: {compressed['compressed_size']} chars")
    print(f"   Compression ratio: {compressed['compression_ratio']:.1%}")
    print(f"   Processing time: {compressed['processing_time_ms']:.1f}ms")
    print(f"   Strategy: {compressed['strategy_used']}")

    # 2. Get compression stats
    print("\n2. Getting compression stats...")
    stats = client.get_compression_stats()

    print(f"   Max context size: {stats['max_context_size']} chars")
    print(f"   Default strategy: {stats['default_strategy']}")
    print(f"   Available strategies: {len(stats['available_strategies'])}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Felix Memory & History API Example Client")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Felix API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (optional)"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "task", "workflow", "knowledge", "compression"],
        default="all",
        help="Which demo to run (default: all)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Felix Memory & History API - Example Client")
    print("=" * 70)
    print(f"API URL: {args.api_url}")
    print(f"Authentication: {'Enabled' if args.api_key else 'Disabled'}")

    # Create client
    client = MemoryHistoryClient(args.api_url, args.api_key)

    # Test API connection
    print("\nTesting API connection...")
    try:
        response = httpx.get(f"{args.api_url}/health", timeout=5.0)
        response.raise_for_status()
        print("✓ API is reachable")
    except Exception as e:
        print(f"✗ Cannot reach API: {e}")
        print("\nMake sure the Felix API server is running:")
        print("  python -m uvicorn src.api.main:app --reload --port 8000")
        sys.exit(1)

    # Run demos
    try:
        if args.demo in ["all", "task"]:
            demo_task_memory(client)

        if args.demo in ["all", "workflow"]:
            demo_workflow_history(client)

        if args.demo in ["all", "knowledge"]:
            demo_knowledge_memory(client)

        if args.demo in ["all", "compression"]:
            demo_compression(client)

        print("\n" + "=" * 70)
        print("✓ All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
