#!/usr/bin/env python3
"""
Test script to verify agents integration without GUI display.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lm_studio_connection():
    """Test LM Studio connection."""
    try:
        from src.llm.lm_studio_client import LMStudioClient
        client = LMStudioClient()
        connected = client.test_connection()
        print(f"LM Studio connection test: {'SUCCESS' if connected else 'FAILED'}")
        return connected
    except Exception as e:
        print(f"LM Studio connection error: {e}")
        return False

def test_helix_geometry():
    """Test helix geometry creation."""
    try:
        from src.core.helix_geometry import HelixGeometry
        helix = HelixGeometry(top_radius=5.0, bottom_radius=1.0, height=10.0, turns=3)
        position = helix.get_position(0.5)
        print(f"Helix geometry test: SUCCESS - Position at t=0.5: {position}")
        return True
    except Exception as e:
        print(f"Helix geometry error: {e}")
        return False

def test_llm_agent_creation():
    """Test LLM agent creation."""
    try:
        from src.agents.llm_agent import LLMAgent
        from src.core.helix_geometry import HelixGeometry
        from src.llm.lm_studio_client import LMStudioClient

        # Create dependencies
        helix = HelixGeometry(top_radius=5.0, bottom_radius=1.0, height=10.0, turns=3)
        lm_client = LMStudioClient()

        # Create LLM agent
        agent = LLMAgent(
            agent_id="test_agent",
            spawn_time=0.1,
            helix=helix,
            llm_client=lm_client,
            agent_type="general"
        )

        print(f"LLM agent creation test: SUCCESS - Agent {agent.agent_id} created")
        return True
    except Exception as e:
        print(f"LLM agent creation error: {e}")
        return False

def test_agent_task_processing():
    """Test LLM agent task processing."""
    try:
        from src.agents.llm_agent import LLMAgent, LLMTask
        from src.core.helix_geometry import HelixGeometry
        from src.llm.lm_studio_client import LMStudioClient

        # Create dependencies
        helix = HelixGeometry(top_radius=5.0, bottom_radius=1.0, height=10.0, turns=3)
        lm_client = LMStudioClient()

        # Create agent
        agent = LLMAgent(
            agent_id="test_agent",
            spawn_time=0.1,
            helix=helix,
            llm_client=lm_client,
            agent_type="general"
        )

        # Create task
        task = LLMTask(
            task_id="test_task",
            description="Say hello and introduce yourself as a test agent.",
            context="This is a test of the LLM agent system."
        )

        # Process task
        result = agent.process_task_with_llm(task, current_time=0.1)

        print(f"Agent task processing test: SUCCESS")
        print(f"Response: {result.content[:100]}...")
        print(f"Confidence: {result.confidence}")
        return True
    except Exception as e:
        print(f"Agent task processing error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Felix agents integration...")
    print("=" * 50)

    tests = [
        test_lm_studio_connection,
        test_helix_geometry,
        test_llm_agent_creation,
        test_agent_task_processing
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All tests passed! Agents integration is working correctly.")
        print("The GUI agents tab should now properly interact with LM Studio.")
    else:
        print("❌ Some tests failed. Check the errors above.")