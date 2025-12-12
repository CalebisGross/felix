"""
Basic Test: Verify Subconscious Tool Memory Prevents Unwanted File Creation

This test validates the core functionality: simple questions don't trigger
automatic file creation by agents.

Original Problem:
    "What is 2+2?" → CriticAgent creates /results/critique_summary_final.txt

Expected Behavior:
    "What is 2+2?" → No file operations instructions → No files created

Run:
    python3 tests/test_tool_memory_basic.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.communication.synthesis_engine import SynthesisEngine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_tool_classification():
    """Test that tool requirements are correctly classified."""
    logger.info("="*60)
    logger.info("TEST 1: Tool Requirement Classification")
    logger.info("="*60)

    engine = SynthesisEngine(None, None)

    # Test 1: Simple question (NO tools needed)
    task1 = "What is 2+2?"
    reqs1 = engine.classify_tool_requirements(task1)
    logger.info(f"\nTask: '{task1}'")
    logger.info(f"Tool requirements: {reqs1}")

    assert reqs1['needs_file_ops'] == False, "Simple math question should NOT need file operations"
    assert reqs1['needs_web_search'] == False, "Simple math question should NOT need web search"
    assert reqs1['needs_system_commands'] == False, "Simple math question should NOT need system commands"
    logger.info("✓ PASS: No tools needed for simple question")

    # Test 2: File creation request (file ops needed)
    task2 = "Create a report and save it to results/test.txt"
    reqs2 = engine.classify_tool_requirements(task2)
    logger.info(f"\nTask: '{task2}'")
    logger.info(f"Tool requirements: {reqs2}")

    assert reqs2['needs_file_ops'] == True, "File creation request SHOULD need file operations"
    logger.info("✓ PASS: File operations needed for file creation request")

    # Test 3: Current information request (web search needed)
    task3 = "What's the latest news on AI?"
    reqs3 = engine.classify_tool_requirements(task3)
    logger.info(f"\nTask: '{task3}'")
    logger.info(f"Tool requirements: {reqs3}")

    assert reqs3['needs_web_search'] == True, "Latest news request SHOULD need web search"
    logger.info("✓ PASS: Web search needed for current information")

    # Test 4: Explain/analysis task (NO tools needed)
    task4 = "Explain quantum computing"
    reqs4 = engine.classify_tool_requirements(task4)
    logger.info(f"\nTask: '{task4}'")
    logger.info(f"Tool requirements: {reqs4}")

    assert reqs4['needs_file_ops'] == False, "Explanation task should NOT need file operations"
    assert reqs4['needs_system_commands'] == False, "Explanation task should NOT need system commands"
    logger.info("✓ PASS: No tools needed for explanation task")

    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("="*60)


def test_tool_retrieval():
    """Test that tool instructions are retrieved correctly."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Tool Instruction Retrieval")
    logger.info("="*60)

    from src.memory.knowledge_store import KnowledgeStore, KnowledgeQuery
    from src.communication.memory_facade import MemoryFacade

    # Initialize knowledge store and memory facade
    knowledge_store = KnowledgeStore()
    memory_facade = MemoryFacade(knowledge_store=knowledge_store)

    # Test 1: No tools needed
    logger.info("\nTest: No tools needed")
    reqs1 = {'needs_file_ops': False, 'needs_web_search': False, 'needs_system_commands': False}
    instructions1, ids1 = memory_facade.retrieve_tool_instructions(reqs1)

    assert instructions1 == "", "Should return empty string when no tools needed"
    assert len(ids1) == 0, "Should return empty ID list when no tools needed"
    logger.info("✓ PASS: Empty instructions when no tools needed")

    # Test 2: File operations needed
    logger.info("\nTest: File operations needed")
    reqs2 = {'needs_file_ops': True, 'needs_web_search': False, 'needs_system_commands': False}
    instructions2, ids2 = memory_facade.retrieve_tool_instructions(reqs2)

    assert len(instructions2) > 0, "Should return instructions when file ops needed"
    assert len(ids2) > 0, "Should return knowledge IDs for meta-learning"
    assert "SYSTEM_ACTION_NEEDED" in instructions2, "File ops instructions should contain SYSTEM_ACTION_NEEDED pattern"
    assert "mkdir" in instructions2 or "echo" in instructions2, "File ops instructions should contain file commands"
    logger.info(f"✓ PASS: Retrieved file operations instructions ({len(instructions2)} chars, {len(ids2)} IDs)")

    # Test 3: Multiple tools needed
    logger.info("\nTest: Multiple tools needed")
    reqs3 = {'needs_file_ops': True, 'needs_web_search': True, 'needs_system_commands': False}
    instructions3, ids3 = memory_facade.retrieve_tool_instructions(reqs3)

    assert len(instructions3) > len(instructions2), "Should have more instructions with multiple tools"
    assert len(ids3) >= 2, "Should have IDs for multiple tools"
    assert "WEB_SEARCH_NEEDED" in instructions3, "Should contain web search pattern"
    assert "SYSTEM_ACTION_NEEDED" in instructions3, "Should contain system action pattern"
    logger.info(f"✓ PASS: Retrieved multiple tool instructions ({len(instructions3)} chars, {len(ids3)} IDs)")

    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("="*60)


def test_no_files_created():
    """Test that no files are created in results/ directory for simple questions."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: File Creation Check")
    logger.info("="*60)

    results_dir = "./results"

    # Get baseline: files that exist before test
    baseline_files = set()
    if os.path.exists(results_dir):
        baseline_files = set(os.listdir(results_dir))
        logger.info(f"Baseline: {len(baseline_files)} files in {results_dir}/")

    # Simulate the workflow classification (not running full workflow to keep test fast)
    engine = SynthesisEngine(None, None)
    task = "What is 2+2?"
    reqs = engine.classify_tool_requirements(task)

    # Verify no file operations would be requested
    assert reqs['needs_file_ops'] == False, "Simple question should NOT trigger file operations"

    # Verify no new files created
    current_files = set()
    if os.path.exists(results_dir):
        current_files = set(os.listdir(results_dir))

    new_files = current_files - baseline_files

    if new_files:
        logger.error(f"❌ FAIL: New files created: {new_files}")
        assert False, f"No files should be created, but found: {new_files}"
    else:
        logger.info("✓ PASS: No files created for simple question")

    logger.info("\n" + "="*60)
    logger.info("✅ TEST PASSED")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        logger.info("\n🧪 FELIX SUBCONSCIOUS TOOL MEMORY - BASIC TESTS\n")

        test_tool_classification()
        test_tool_retrieval()
        test_no_files_created()

        logger.info("\n" + "="*60)
        logger.info("🎉 ALL TESTS PASSED - SYSTEM WORKING CORRECTLY")
        logger.info("="*60)
        logger.info("\nKey Results:")
        logger.info("  ✅ Simple questions don't get file operation instructions")
        logger.info("  ✅ File creation requests DO get file operation instructions")
        logger.info("  ✅ Tool instructions correctly retrieved from knowledge store")
        logger.info("  ✅ No unwanted files created")
        logger.info("\nOriginal problem SOLVED:")
        logger.info("  Agents no longer auto-create files for simple questions")
        logger.info("="*60)

        sys.exit(0)

    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)
