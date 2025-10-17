#!/usr/bin/env python3
"""
Temporary test script to verify Felix framework components.
This script tests core functionality without requiring external LLM servers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")

    try:
        from core import helix_geometry
        print("✓ HelixGeometry imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import HelixGeometry: {e}")
        return False

    try:
        from agents import agent
        print("✓ Agent module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Agent module: {e}")
        return False

    try:
        from communication import central_post, spoke
        print("✓ Communication modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Communication modules: {e}")
        return False

    try:
        from memory import knowledge_store
        print("✓ Memory modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Memory modules: {e}")
        return False

    try:
        from pipeline import linear_pipeline
        print("✓ Pipeline modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pipeline modules: {e}")
        return False

    return True

def test_helix_geometry():
    """Test HelixGeometry functionality."""
    print("\nTesting HelixGeometry...")

    try:
        from core.helix_geometry import HelixGeometry

        # Test instantiation with correct parameters
        helix = HelixGeometry(top_radius=2.0, bottom_radius=1.0, height=10.0, turns=3)
        print("✓ HelixGeometry instantiated successfully")

        # Test position calculation
        position = helix.get_position(0.5)
        print(f"✓ Position calculation works: {position}")

        # Test radius calculation
        radius = helix.get_radius(5.0)
        print(f"✓ Radius calculation works: {radius}")

        return True

    except Exception as e:
        print(f"✗ HelixGeometry test failed: {e}")
        return False

def test_agent_system():
    """Test basic agent functionality."""
    print("\nTesting Agent system...")

    try:
        from agents.agent import Agent
        from core.helix_geometry import HelixGeometry

        # Create helix for agent
        helix = HelixGeometry(top_radius=2.0, bottom_radius=1.0, height=10.0, turns=3)

        # Test agent instantiation with correct parameters
        test_agent = Agent(agent_id="test_agent_1", spawn_time=0.2, helix=helix)
        print("✓ Agent instantiated successfully")

        # Test agent spawning
        mock_task = type('Task', (), {'id': 'test_task'})()
        test_agent.spawn(current_time=0.3, task=mock_task)
        print("✓ Agent spawning works")

        # Test position update
        test_agent.update_position(current_time=0.5)
        position = test_agent.get_position(current_time=0.5)
        print(f"✓ Agent position update works: {position}")

        return True

    except Exception as e:
        print(f"✗ Agent system test failed: {e}")
        return False

def test_communication():
    """Test communication system."""
    print("\nTesting Communication system...")

    try:
        from communication.central_post import CentralPost, Message, MessageType
        from communication.spoke import Spoke
        from agents.agent import Agent
        from core.helix_geometry import HelixGeometry

        # Create helix and agent for testing
        helix = HelixGeometry(top_radius=2.0, bottom_radius=1.0, height=10.0, turns=3)
        test_agent = Agent(agent_id="test_agent_1", spawn_time=0.2, helix=helix)

        # Test CentralPost
        central = CentralPost(max_agents=10)
        print("✓ CentralPost instantiated successfully")

        # Test Spoke
        spoke = Spoke(agent=test_agent, central_post=central)
        print("✓ Spoke instantiated successfully")

        # Test message creation and routing
        message = Message(
            sender_id="test_agent_1",
            message_type=MessageType.STATUS_UPDATE,
            content={"status": "active", "position": 0.5},
            timestamp=0.5
        )
        message_id = spoke.send_message(message)
        print(f"✓ Message sending works: {message_id}")

        return True

    except Exception as e:
        print(f"✗ Communication test failed: {e}")
        return False

def test_memory():
    """Test memory system."""
    print("\nTesting Memory system...")

    try:
        from memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel

        # Test KnowledgeStore
        store = KnowledgeStore(storage_path="test_memory.db")
        print("✓ KnowledgeStore instantiated successfully")

        # Test knowledge storage with correct method
        knowledge_id = store.store_knowledge(
            knowledge_type=KnowledgeType.TASK_RESULT,
            content={"data": "test_value", "timestamp": "now"},
            confidence_level=ConfidenceLevel.HIGH,
            source_agent="test_agent",
            domain="test"
        )
        print(f"✓ Knowledge storage works: {knowledge_id}")

        # Test retrieval
        from memory.knowledge_store import KnowledgeQuery
        query = KnowledgeQuery(limit=10)
        results = store.retrieve_knowledge(query)
        print(f"✓ Knowledge retrieval works: {len(results)} entries")

        return True

    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return False

def test_pipeline():
    """Test pipeline functionality."""
    print("\nTesting Pipeline system...")

    try:
        from pipeline.linear_pipeline import LinearPipeline, PipelineAgent

        # Test LinearPipeline with correct parameters
        pipeline = LinearPipeline(num_stages=3, stage_capacity=5)
        print("✓ LinearPipeline instantiated successfully")

        # Test adding agents to pipeline
        test_agent = PipelineAgent(agent_id="test_agent_1", spawn_time=0.2)
        pipeline.add_agent(test_agent, current_time=0.3)
        print("✓ Agent addition works")

        # Test pipeline update
        pipeline.update(current_time=0.5)
        print("✓ Pipeline update works")

        # Test performance metrics
        metrics = pipeline.get_performance_metrics()
        print(f"✓ Performance metrics work: {metrics}")

        return True

    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False

def test_llm_integration():
    """Test LLM client classes (without actual server calls)."""
    print("\nTesting LLM integration...")

    try:
        from llm.lm_studio_client import LMStudioClient
        from llm.token_budget import TokenBudgetManager

        # Test TokenBudgetManager
        budget_manager = TokenBudgetManager(base_budget=1000)
        print("✓ TokenBudgetManager instantiated successfully")

        # Test agent budget initialization
        total_budget = budget_manager.initialize_agent_budget("test_agent", "research")
        print(f"✓ Agent budget initialization works: {total_budget} tokens")

        # Test LMStudioClient (without server connection)
        try:
            client = LMStudioClient(base_url="http://localhost:1234")
            print("✓ LMStudioClient instantiated successfully (no server connection attempted)")
        except Exception as e:
            print(f"✗ LMStudioClient instantiation failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"✗ LLM integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Felix Framework Verification Tests")
    print("=" * 40)

    tests = [
        test_core_imports,
        test_helix_geometry,
        test_agent_system,
        test_communication,
        test_memory,
        test_pipeline,
        test_llm_integration,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 40)
    print("Summary:")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ All {total} tests passed!")
        return True
    else:
        print(f"✗ {passed}/{total} tests passed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)