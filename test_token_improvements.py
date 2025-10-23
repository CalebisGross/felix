#!/usr/bin/env python3
"""
Test script to verify token management and prompt improvements in Felix framework.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.token_budget import TokenBudgetManager
from src.agents.llm_agent import LLMAgent, LLMTask
from src.communication.central_post import CentralPost
from src.memory.knowledge_store import KnowledgeStore, KnowledgeEntry, KnowledgeType, ConfidenceLevel
from src.memory.context_compression import ContextCompressor, CompressionStrategy, CompressionConfig
from src.workflows.context_builder import CollaborativeContextBuilder
from src.core.helix_geometry import HelixGeometry

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_token_allocations():
    """Test that token allocations are properly fixed."""
    logger.info("\n=== Testing Token Allocations ===")

    # Create token budget manager
    budget_manager = TokenBudgetManager(base_budget=3000, strict_mode=True)

    # Test allocations for each agent type
    agent_types = ["research", "analysis", "synthesis", "critic"]
    expected_budgets = {
        "research": 4000,   # Fixed: was 2000
        "analysis": 2500,   # Fixed: was 2000
        "synthesis": 1500,  # Fixed: was 20000!
        "critic": 1000      # Fixed: was 2000
    }

    for agent_type in agent_types:
        agent_id = f"test_{agent_type}_001"
        budget = budget_manager.initialize_agent_budget(agent_id, agent_type)

        logger.info(f"  {agent_type.upper()} agent:")
        logger.info(f"    Allocated budget: {budget} tokens")
        logger.info(f"    Expected budget: {expected_budgets[agent_type]} tokens")

        if budget == expected_budgets[agent_type]:
            logger.info(f"    ✓ PASS - Budget correctly allocated")
        else:
            logger.error(f"    ✗ FAIL - Budget mismatch!")

        # Test compression ratios (should be inverted now)
        allocation = budget_manager.calculate_stage_allocation(agent_id, 0.5, 1)
        logger.info(f"    Compression ratio: {allocation.compression_ratio:.2f}")

        # Research should have LOW compression, synthesis HIGH
        if agent_type == "research" and allocation.compression_ratio > 0.4:
            logger.error(f"    ✗ FAIL - Research compression too high!")
        elif agent_type == "synthesis" and allocation.compression_ratio < 0.6:
            logger.error(f"    ✗ FAIL - Synthesis compression too low!")
        else:
            logger.info(f"    ✓ PASS - Compression ratio appropriate")


def test_knowledge_integration():
    """Test that knowledge entries are properly included in prompts."""
    logger.info("\n=== Testing Knowledge Integration ===")

    # Create mock knowledge store with some entries
    knowledge_store = KnowledgeStore("test_knowledge.db")

    # Add test knowledge entry
    knowledge_store.store_knowledge(
        knowledge_type=KnowledgeType.AGENT_INSIGHT,
        content={"text": "Helical progression allows agents to naturally evolve from exploration to synthesis"},
        confidence_level=ConfidenceLevel.HIGH,
        source_agent="test_research_001",
        domain="helical_theory"
    )

    # Create a task with knowledge entries
    task = LLMTask(
        task_id="test_001",
        description="Test task for knowledge integration",
        context="Testing context",
        knowledge_entries=[
            KnowledgeEntry(
                knowledge_id="test_k002",
                knowledge_type=KnowledgeType.TASK_RESULT,
                content={"text": "Test knowledge from memory system"},
                confidence_level=ConfidenceLevel.MEDIUM,
                source_agent="system",
                domain="test"
            )
        ]
    )

    # Create a real LLM client (LM Studio)
    from src.llm.lm_studio_client import LMStudioClient

    # Use default LM Studio settings
    llm_client = LMStudioClient(base_url="http://localhost:1234/v1")

    # Create helix geometry for agent
    helix = HelixGeometry(
        top_radius=3.0,
        bottom_radius=0.5,
        height=8.0,
        turns=2.0
    )

    agent = LLMAgent(
        agent_id="test_agent",
        agent_type="research",
        spawn_time=0.0,
        helix=helix,
        llm_client=llm_client
    )

    # Create prompt with knowledge
    prompt, budget = agent.create_position_aware_prompt(task, 0.5)

    # Check if knowledge is in prompt
    if "Relevant Knowledge from Memory" in prompt:
        logger.info("  ✓ PASS - Knowledge section found in prompt")
        if "Test knowledge from memory system" in prompt:
            logger.info("  ✓ PASS - Knowledge content included")
        else:
            logger.error("  ✗ FAIL - Knowledge content missing")
    else:
        logger.error("  ✗ FAIL - Knowledge section not found in prompt")

    # Check if metadata is included
    if "=== Processing Parameters ===" in prompt:
        logger.info("  ✓ PASS - Metadata section found")
        if "Temperature:" in prompt and "Token Budget:" in prompt:
            logger.info("  ✓ PASS - Temperature and token budget included")
        else:
            logger.error("  ✗ FAIL - Metadata incomplete")
    else:
        logger.error("  ✗ FAIL - Metadata section missing")

    # Cleanup
    # Note: KnowledgeStore doesn't have a close method, SQLite handles cleanup automatically
    Path("test_knowledge.db").unlink(missing_ok=True)


def test_compression_implementation():
    """Test that real compression is now implemented."""
    logger.info("\n=== Testing Compression Implementation ===")

    # Create context compressor with config
    config = CompressionConfig(
        strategy=CompressionStrategy.ABSTRACTIVE_SUMMARY,
        max_context_size=1000
    )
    compressor = ContextCompressor(config=config)

    # Create context builder
    context_builder = CollaborativeContextBuilder(
        central_post=None,  # We'll use mock messages
        context_compressor=compressor
    )

    # Test compression with large context (as dict since that's what it expects)
    large_text = "This is a test of the compression system. " * 100
    context_dict = {"content": large_text}
    compressed = compressor.compress_context(context_dict, target_size=50)

    if compressed and compressed.content:
        logger.info("  ✓ PASS - Compression produced output")
        logger.info(f"    Original size: {len(large_text)} chars")
        compressed_str = str(compressed.content)
        logger.info(f"    Compressed size: {compressed.compressed_size} chars")
        logger.info(f"    Compression ratio: {compressed.compression_ratio:.2f}")

        if compressed.compressed_size < len(large_text):
            logger.info("  ✓ PASS - Text was actually compressed")
        else:
            logger.error("  ✗ FAIL - Compression did not reduce size")
    else:
        logger.error("  ✗ FAIL - Compression failed to produce output")


def test_prompt_optimization():
    """Test that prompts are more concise."""
    logger.info("\n=== Testing Prompt Optimization ===")

    # Create real LLM agent with LM Studio client
    from src.llm.lm_studio_client import LMStudioClient

    llm_client = LMStudioClient(base_url="http://localhost:1234/v1")
    helix = HelixGeometry(
        top_radius=3.0,
        bottom_radius=0.5,
        height=8.0,
        turns=2.0
    )

    agent = LLMAgent(
        agent_id="test_agent",
        agent_type="research",
        spawn_time=0.0,
        helix=helix,
        llm_client=llm_client
    )

    # Build collaborative prompt
    context_history = [
        {"agent_type": "research", "response": "Previous finding 1", "confidence": 0.7},
        {"agent_type": "analysis", "response": "Previous finding 2", "confidence": 0.8}
    ]

    prompt = agent._build_collaborative_prompt(
        "Test task",
        context_history,
        "research"
    )

    # Check if prompt is concise
    prompt_lines = prompt.split('\n')
    instruction_lines = [l for l in prompt_lines if l.startswith("**")]

    logger.info(f"  Prompt has {len(prompt_lines)} total lines")
    logger.info(f"  Instruction lines: {len(instruction_lines)}")

    # Should be much shorter than the old 35+ lines
    if len(instruction_lines) < 10:
        logger.info("  ✓ PASS - Prompt instructions are concise")
    else:
        logger.error("  ✗ FAIL - Prompt still too verbose")

    # Check for key concise instructions
    if "DO NOT repeat - ADD NEW insights" in prompt:
        logger.info("  ✓ PASS - Concise collaboration instruction found")
    else:
        logger.error("  ✗ FAIL - Missing collaboration instruction")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("FELIX TOKEN MANAGEMENT & PROMPT IMPROVEMENTS TEST")
    logger.info("=" * 60)

    try:
        test_token_allocations()
        test_knowledge_integration()
        test_compression_implementation()
        test_prompt_optimization()

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS COMPLETED")
        logger.info("Check the output above for any FAIL messages")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED WITH ERROR: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())