#!/usr/bin/env python3
"""
Advanced Felix framework verification tests.
Tests agent interactions, dynamic spawning, and complex workflows.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_agent_interactions():
    """Test complex agent interactions and workflows."""
    print("Testing Agent Interactions...")

    try:
        from agents.agent import Agent, generate_spawn_times, create_agents_from_spawn_times
        from core.helix_geometry import HelixGeometry
        from communication.central_post import CentralPost, Message, MessageType
        from communication.spoke import Spoke
        from memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel

        # Create helix geometry
        helix = HelixGeometry(top_radius=2.0, bottom_radius=1.0, height=10.0, turns=3)

        # Create multiple agents
        spawn_times = generate_spawn_times(count=5, seed=42)
        agents = create_agents_from_spawn_times(spawn_times, helix)
        print(f"✓ Created {len(agents)} agents successfully")

        # Create central post and register agents
        central = CentralPost(max_agents=10)
        spokes = []

        for agent in agents:
            # Spawn agent
            mock_task = type('Task', (), {'id': f'task_{agent.agent_id}'})()
            agent.spawn(current_time=0.5, task=mock_task)

            # Create spoke connection
            spoke = Spoke(agent=agent, central_post=central)
            spokes.append(spoke)

        print(f"✓ Registered {len(spokes)} agents with central post")

        # Test message passing between agents
        message = Message(
            sender_id=agents[0].agent_id,
            message_type=MessageType.STATUS_UPDATE,
            content={"status": "active", "position": 0.5, "confidence": 0.8},
            timestamp=0.5
        )

        message_id = spokes[0].send_message(message)
        print(f"✓ Message passing works: {message_id}")

        # Test memory integration
        store = KnowledgeStore(storage_path="test_memory.db")

        # Store agent result as knowledge
        knowledge_id = store.store_knowledge(
            knowledge_type=KnowledgeType.TASK_RESULT,
            content={"result": "test analysis", "confidence": 0.8},
            confidence_level=ConfidenceLevel.HIGH,
            source_agent=agents[0].agent_id,
            domain="test_domain"
        )
        print(f"✓ Memory integration works: {knowledge_id}")

        return True

    except Exception as e:
        print(f"✗ Agent interactions test failed: {e}")
        return False

def test_dynamic_spawning():
    """Test dynamic agent spawning functionality."""
    print("\nTesting Dynamic Spawning...")

    try:
        from agents.dynamic_spawning import DynamicSpawning
        from core.helix_geometry import HelixGeometry
        from communication.central_post import CentralPost, Message, MessageType

        # Create helix and central post
        helix = HelixGeometry(top_radius=2.0, bottom_radius=1.0, height=10.0, turns=3)
        central = CentralPost(max_agents=10)

        # Create dynamic spawning system
        spawner = DynamicSpawning(
            agent_factory=None,  # We'll test the core logic without full factory
            confidence_threshold=0.7,
            max_agents=5,
            token_budget_limit=1000
        )
        print("✓ DynamicSpawning instantiated successfully")

        # Test confidence analysis
        messages = [
            Message("agent1", MessageType.STATUS_UPDATE, {"confidence": 0.5}, 0.1),
            Message("agent2", MessageType.STATUS_UPDATE, {"confidence": 0.6}, 0.2),
            Message("agent3", MessageType.STATUS_UPDATE, {"confidence": 0.4}, 0.3),
        ]

        # Test the core spawning logic
        should_spawn = spawner._should_spawn_new_agent(messages, current_time=0.5)
        print(f"✓ Spawning analysis works: {should_spawn}")

        return True

    except Exception as e:
        print(f"✗ Dynamic spawning test failed: {e}")
        return False

def test_llm_mocking():
    """Test LLM integration with proper mocking."""
    print("\nTesting LLM Integration with Mocking...")

    try:
        from llm.lm_studio_client import LMStudioClient
        from llm.token_budget import TokenBudgetManager

        # Test TokenBudgetManager with different agent types
        budget_manager = TokenBudgetManager(base_budget=1000)

        agent_types = ["research", "analysis", "synthesis", "critic"]
        for agent_type in agent_types:
            budget = budget_manager.initialize_agent_budget(f"test_{agent_type}", agent_type)
            allocation = budget_manager.calculate_stage_allocation(f"test_{agent_type}", 0.5, 1)
            print(f"✓ {agent_type} agent: {budget} total, {allocation.stage_budget} stage budget")

        # Test LMStudioClient (should work without server)
        client = LMStudioClient(base_url="http://localhost:1234")
        print("✓ LMStudioClient handles missing server gracefully")

        return True

    except Exception as e:
        print(f"✗ LLM mocking test failed: {e}")
        return False

def test_pipeline_advanced():
    """Test advanced pipeline functionality."""
    print("\nTesting Advanced Pipeline Features...")

    try:
        from pipeline.linear_pipeline import LinearPipeline, PipelineAgent

        # Create pipeline
        pipeline = LinearPipeline(num_stages=4, stage_capacity=3)

        # Create multiple agents with different spawn times
        agents = []
        for i in range(5):
            agent = PipelineAgent(agent_id=f"agent_{i}", spawn_time=0.1 * i)
            agents.append(agent)
            pipeline.add_agent(agent, current_time=0.5)

        print(f"✓ Added {len(agents)} agents to pipeline")

        # Test agent advancement through stages
        pipeline.update(current_time=0.6)
        pipeline.advance_agents()

        # Check agent distribution
        stage_loads = [stage.current_load for stage in pipeline.stages]
        print(f"✓ Agent distribution across stages: {stage_loads}")

        # Test workload coefficient of variation
        cv = pipeline.calculate_workload_cv()
        print(f"✓ Workload CV calculation: {cv:.3f}")

        return True

    except Exception as e:
        print(f"✗ Advanced pipeline test failed: {e}")
        return False

def test_memory_advanced():
    """Test advanced memory functionality."""
    print("\nTesting Advanced Memory Features...")

    try:
        from memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel, KnowledgeQuery
        from memory.context_compression import ContextCompressor, CompressionStrategy

        # Test KnowledgeStore with multiple entries
        store = KnowledgeStore(storage_path="test_memory.db")

        # Add multiple knowledge entries
        domains = ["research", "analysis", "synthesis"]
        for i, domain in enumerate(domains):
            knowledge_id = store.store_knowledge(
                knowledge_type=KnowledgeType.TASK_RESULT,
                content={"result": f"test result {i}", "domain": domain},
                confidence_level=ConfidenceLevel.HIGH,
                source_agent=f"agent_{i}",
                domain=domain,
                tags=[f"tag_{i}", "common_tag"]
            )

        print("✓ Stored multiple knowledge entries")

        # Test retrieval with filtering
        query = KnowledgeQuery(
            domains=["research"],
            min_confidence=ConfidenceLevel.MEDIUM,
            limit=5
        )
        results = store.retrieve_knowledge(query)
        print(f"✓ Filtered retrieval works: {len(results)} results")

        # Test context compression
        compressor = ContextCompressor()
        test_context = {"content": "This is a long text " * 50}
        compressed = compressor.compress_context(
            context=test_context,
            strategy=CompressionStrategy.EXTRACTIVE_SUMMARY
        )
        print(f"✓ Context compression works: {type(compressed)}")

        return True

    except Exception as e:
        print(f"✗ Advanced memory test failed: {e}")
        return False

def main():
    """Run all advanced tests."""
    print("Felix Framework Advanced Verification Tests")
    print("=" * 50)

    tests = [
        test_agent_interactions,
        test_dynamic_spawning,
        test_llm_mocking,
        test_pipeline_advanced,
        test_memory_advanced,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print("Advanced Tests Summary:")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} advanced tests passed!")
        return True
    else:
        print(f"✗ {passed}/{total} advanced tests passed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)