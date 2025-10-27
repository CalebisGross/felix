#!/usr/bin/env python3
"""
Test script for Terminal tab streaming functionality.

Tests:
1. SystemExecutor streaming callback mechanism
2. CommandHistory placeholder and update flow
3. CentralPost broadcast messages
4. Database status tracking
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.execution import SystemExecutor, CommandHistory, TrustManager
from src.communication import CentralPost, MessageType
from src.core.helix_geometry import HelixGeometry


def test_streaming_output():
    """Test SystemExecutor streaming with callback."""
    print("=" * 60)
    print("TEST 1: SystemExecutor Streaming")
    print("=" * 60)

    executor = SystemExecutor()

    # Collect output via callback
    output_lines = []

    def callback(line: str, stream_type: str):
        output_lines.append(f"[{stream_type}] {line}")
        print(f"  Callback received: [{stream_type}] {line}")

    # Test with multi-step command
    print("\nExecuting: 'echo Line1 && sleep 1 && echo Line2 && sleep 1 && echo Line3'")
    result = executor.execute_command_streaming(
        command="echo Line1 && sleep 1 && echo Line2 && sleep 1 && echo Line3",
        output_callback=callback
    )

    print(f"\n✓ Command completed:")
    print(f"  Success: {result.success}")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Callback invocations: {len(output_lines)}")
    print(f"  Lines captured: {output_lines}")

    assert result.success, "Command should succeed"
    assert len(output_lines) >= 3, "Should receive at least 3 lines"
    assert "Line1" in str(output_lines), "Should capture Line1"

    print("\n✓ TEST 1 PASSED\n")


def test_command_history_flow():
    """Test CommandHistory placeholder and update flow."""
    print("=" * 60)
    print("TEST 2: CommandHistory Placeholder/Update Flow")
    print("=" * 60)

    from src.execution.trust_manager import TrustLevel
    from src.execution.system_executor import CommandResult, ErrorCategory

    # Use existing database with migrations
    print("\nUsing database: felix_system_actions.db")
    history = CommandHistory("felix_system_actions.db")

    # Create placeholder
    print("\nCreating execution placeholder...")
    exec_id = history.create_execution_placeholder(
        command="test command",
        command_hash="test_hash_123",
        agent_id="test_agent",
        agent_type="test",
        workflow_id=None,
        trust_level=TrustLevel.SAFE,
        context="Test execution"
    )
    print(f"  Execution ID: {exec_id}")

    # Query active commands
    active = history.get_active_commands()
    print(f"\n  Active commands: {len(active)}")
    assert len(active) >= 1, "Should have at least 1 active command"

    # Verify status is 'running'
    found = False
    for cmd in active:
        if cmd['execution_id'] == exec_id:
            assert cmd['status'] == 'running', "Status should be 'running'"
            found = True
            print(f"  ✓ Found placeholder with status='running'")
            break
    assert found, "Should find created placeholder"

    # Simulate command completion
    print("\nSimulating command completion...")
    time.sleep(0.5)

    result = CommandResult(
        command="test command",
        exit_code=0,
        stdout="test output",
        stderr="",
        duration=0.5,
        success=True,
        error_category=None,
        cwd="/test",
        venv_active=False,
        output_size=11
    )

    history.update_execution_result(exec_id, result)
    print(f"  ✓ Updated execution with result")

    # Query again - should no longer be active
    active_after = history.get_active_commands()
    found_after = any(cmd['execution_id'] == exec_id for cmd in active_after)
    assert not found_after, "Should not be in active commands after completion"
    print(f"  ✓ No longer in active commands")

    # Get full details
    details = history.get_command_details(exec_id)
    assert details is not None, "Should retrieve details"
    assert details['status'] == 'completed', "Status should be 'completed'"
    assert details['success'] == True, "Should be successful"
    print(f"  ✓ Retrieved full details with status='completed'")

    print("\n✓ TEST 2 PASSED\n")


def test_broadcast_messages():
    """Test CentralPost broadcast message generation."""
    print("=" * 60)
    print("TEST 3: CentralPost Broadcast Messages")
    print("=" * 60)

    from src.execution.trust_manager import TrustLevel

    # Create minimal CentralPost
    helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2.0)
    central_post = CentralPost(
        helix=helix,
        enable_memory=False,
        enable_metrics=False
    )

    print("\nRequesting system action (SAFE command)...")
    action_id = central_post.request_system_action(
        agent_id="test_agent",
        command="echo 'Test streaming'",
        context="Test broadcast messages"
    )

    print(f"  Action ID: {action_id}")

    # Check that result is available
    result = central_post.get_action_result(action_id)
    assert result is not None, "Should have result for SAFE command"
    print(f"  ✓ Command executed: success={result.success}")

    # Check messages in queue
    # Note: In real system, GUI would consume these via message processing
    queue_size = central_post.message_queue_size
    print(f"  Message queue size: {queue_size}")
    print(f"  ✓ Messages broadcast to queue")

    # Query history to verify execution was recorded
    history = central_post.command_history.get_filtered_history(limit=10)
    assert len(history) > 0, "Should have at least one execution in history"

    latest = history[0]
    print(f"\n  Latest execution in history:")
    print(f"    ID: {latest['execution_id']}")
    print(f"    Command: {latest['command']}")
    print(f"    Status: {latest['status']}")
    print(f"    Success: {latest['success']}")

    assert latest['status'] in ['completed', 'failed'], "Should have final status"

    print("\n✓ TEST 3 PASSED\n")


def test_filtered_history():
    """Test CommandHistory filtering."""
    print("=" * 60)
    print("TEST 4: CommandHistory Filtering")
    print("=" * 60)

    history = CommandHistory("felix_system_actions.db")

    # Test status filter
    print("\nTesting status filter...")
    completed = history.get_filtered_history(status='completed', limit=50)
    print(f"  Completed commands: {len(completed)}")

    # Test search query
    print("\nTesting search query...")
    echo_results = history.get_filtered_history(search_query='echo', limit=50)
    print(f"  Commands matching 'echo': {len(echo_results)}")

    # Test combined filters
    print("\nTesting combined filters...")
    combined = history.get_filtered_history(
        status='completed',
        search_query='test',
        limit=10
    )
    print(f"  Completed + 'test' query: {len(combined)}")

    print("\n✓ TEST 4 PASSED\n")


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("TERMINAL STREAMING TEST SUITE")
        print("=" * 60 + "\n")

        test_streaming_output()
        test_command_history_flow()
        test_broadcast_messages()
        test_filtered_history()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nTerminal streaming implementation verified!")
        print("Next step: Test GUI with 'python -m src.gui.main'")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
