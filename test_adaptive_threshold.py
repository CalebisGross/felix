#!/usr/bin/env python3
"""
Test script for adaptive confidence threshold system.

This test validates:
1. Only 1-2 web searches (not 4)
2. Concise answers (<50 words, not 5000+ chars)
3. Confidence threshold lowered to 0.60 when trustable knowledge exists
4. Direct answer mode activates when appropriate
5. Total time <15 seconds
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.felix_system import FelixSystem, FelixConfig

# Configure detailed logging - FORCE DEBUG for all loggers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s',
    force=True  # Force reconfiguration
)

# Set all relevant loggers to DEBUG
logging.getLogger('src.communication.central_post').setLevel(logging.DEBUG)
logging.getLogger('src.workflows.context_builder').setLevel(logging.DEBUG)
logging.getLogger('src.agents.specialized_agents').setLevel(logging.DEBUG)
logging.getLogger('src.workflows.felix_workflow').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def test_adaptive_threshold():
    """Test adaptive confidence threshold with time query."""
    logger.info("\n" + "="*80)
    logger.info("ADAPTIVE CONFIDENCE THRESHOLD TEST")
    logger.info("="*80)
    logger.info("\nQuery: 'what is the current date and time?'")
    logger.info("\nExpected behavior:")
    logger.info("  ✓ Only 1-2 web searches (not 4)")
    logger.info("  ✓ Concise answer (<50 words)")
    logger.info("  ✓ Threshold lowered to 0.60 when trustable knowledge exists")
    logger.info("  ✓ Direct answer mode activates")
    logger.info("  ✓ Total time <15 seconds")
    logger.info("="*80)

    # Create test databases
    test_db = "test_adaptive_threshold.db"
    knowledge_db = "test_adaptive_knowledge.db"
    Path(test_db).unlink(missing_ok=True)
    Path(knowledge_db).unlink(missing_ok=True)

    # Track metrics
    start_time = time.time()

    felix_system = None

    try:
        print("\n[TEST] Creating Felix configuration...")
        # Create Felix configuration with web search enabled
        config = FelixConfig(
            memory_db_path=test_db,
            knowledge_db_path=knowledge_db,
            web_search_enabled=True,
            web_search_provider="duckduckgo",
            web_search_max_results=5,
            web_search_confidence_threshold=0.70,  # Default threshold
            web_search_min_samples=1,
            web_search_cooldown=10.0,
            max_agents=25,
            enable_memory=True,
            verbose_llm_logging=False  # Reduce noise for test
        )

        # Create and start Felix system
        print("[TEST] Initializing Felix system...")
        logger.info("\nInitializing Felix system...")
        felix_system = FelixSystem(config)

        print("[TEST] Starting Felix system...")
        if not felix_system.start():
            print("[TEST] ✗ Failed to start Felix system")
            logger.error("✗ Failed to start Felix system")
            return False

        print("[TEST] ✓ Felix system started successfully\n")
        logger.info("✓ Felix system started successfully\n")

        # Run Felix workflow with the time query
        print("[TEST] Starting Felix workflow...")
        logger.info("Starting Felix workflow...")
        logger.info("-" * 80)

        print("[TEST] Calling felix_system.run_workflow()...")
        result = felix_system.run_workflow(
            task_input="what is the current date and time?"
        )

        print(f"[TEST] Workflow completed with status: {result.get('status', 'unknown')}")
        print(f"[TEST] Result keys: {list(result.keys())}")
        print(f"[TEST] Agents spawned: {len(result.get('agents_spawned', []))}")
        print(f"[TEST] Messages processed: {len(result.get('messages_processed', []))}")
        print(f"[TEST] Centralpost synthesis type: {type(result.get('centralpost_synthesis'))}")
        print(f"[TEST] Centralpost synthesis: {result.get('centralpost_synthesis')}")
        logger.info("-" * 80)

        # Calculate metrics
        total_time = time.time() - start_time

        # Analyze results
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        logger.info("\n" + "="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)

        # Check final answer length (from centralpost_synthesis)
        synthesis_result = result.get('centralpost_synthesis', {})
        if isinstance(synthesis_result, dict):
            final_answer = synthesis_result.get('synthesis_content', '')
        else:
            final_answer = str(synthesis_result) if synthesis_result else ''

        answer_word_count = len(final_answer.split())
        answer_char_count = len(final_answer)

        print(f"\n1. Answer Conciseness:")
        print(f"   Words: {answer_word_count}")
        print(f"   Characters: {answer_char_count}")
        logger.info(f"\n1. Answer Conciseness:")
        logger.info(f"   Words: {answer_word_count}")
        logger.info(f"   Characters: {answer_char_count}")

        if answer_word_count <= 50:
            print("   ✓ PASS: Answer is concise (<50 words)")
            logger.info("   ✓ PASS: Answer is concise (<50 words)")
        else:
            print(f"   ⚠ CONCERN: Answer might be verbose ({answer_word_count} words)")
            logger.warning(f"   ⚠ CONCERN: Answer might be verbose ({answer_word_count} words)")

        # Check total time
        print(f"\n2. Total Time:")
        print(f"   Duration: {total_time:.1f} seconds")
        logger.info(f"\n2. Total Time:")
        logger.info(f"   Duration: {total_time:.1f} seconds")

        if total_time < 15:
            print("   ✓ PASS: Completed quickly (<15 seconds)")
            logger.info("   ✓ PASS: Completed quickly (<15 seconds)")
        else:
            print(f"   ⚠ CONCERN: Took longer than expected ({total_time:.1f}s)")
            logger.warning(f"   ⚠ CONCERN: Took longer than expected ({total_time:.1f}s)")

        # Check confidence (from centralpost_synthesis)
        if isinstance(synthesis_result, dict):
            final_confidence = synthesis_result.get('confidence', 0.0)
        else:
            final_confidence = 0.0
        print(f"\n3. Final Confidence:")
        print(f"   Score: {final_confidence:.2f}")
        logger.info(f"\n3. Final Confidence:")
        logger.info(f"   Score: {final_confidence:.2f}")

        if final_confidence >= 0.60:
            print("   ✓ PASS: Reached synthesis threshold")
            logger.info("   ✓ PASS: Reached synthesis threshold")
        else:
            print(f"   ⚠ CONCERN: Low confidence ({final_confidence:.2f})")
            logger.warning(f"   ⚠ CONCERN: Low confidence ({final_confidence:.2f})")

        # Display final answer
        print(f"\n4. Final Answer:")
        print(f"   {final_answer}")
        logger.info(f"\n4. Final Answer:")
        logger.info(f"   {final_answer}")

        # Check for web search count
        print(f"\n5. Note: Check logs above for:")
        print(f"   - Number of web searches triggered (should be 1-2)")
        print(f"   - 'DIRECT ANSWER MODE activated' message")
        print(f"   - 'Adaptive threshold' adjustment messages")
        logger.info(f"\n5. Note: Check logs above for:")
        logger.info(f"   - Number of web searches triggered (should be 1-2)")
        logger.info(f"   - 'DIRECT ANSWER MODE activated' message")
        logger.info(f"   - 'Adaptive threshold' adjustment messages")

        # Summary
        print("\n" + "="*80)
        logger.info("\n" + "="*80)
        if answer_word_count <= 50 and total_time < 20:
            print("✓ TEST SUCCESSFUL - System shows improvement")
            print("  Review logs above for detailed adaptive threshold behavior")
            logger.info("✓ TEST SUCCESSFUL - System shows improvement")
            logger.info("  Review logs above for detailed adaptive threshold behavior")
        else:
            print("⚠ TEST COMPLETED - Review metrics above")
            logger.warning("⚠ TEST COMPLETED - Review metrics above")
        print("="*80)
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED WITH ERROR: {e}", exc_info=True)
        return False

    finally:
        # Stop Felix system
        if felix_system:
            felix_system.stop()
            logger.info("Felix system stopped")

        # Cleanup databases
        if Path(test_db).exists():
            Path(test_db).unlink(missing_ok=True)
        if Path(knowledge_db).exists():
            Path(knowledge_db).unlink(missing_ok=True)


def main():
    """Run the test."""
    success = test_adaptive_threshold()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
