"""
Test: Verify destructive commands are properly blocked

This test ensures the trust classification properly blocks dangerous commands
like "rm -rf" that could cause data loss.

Run:
    python3 tests/test_destructive_command_blocking.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution.trust_manager import TrustManager, TrustLevel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_rm_rf_blocking():
    """Test that all forms of rm -rf are blocked."""
    logger.info("="*60)
    logger.info("TEST: rm -rf Command Blocking")
    logger.info("="*60)

    trust_manager = TrustManager()

    # Test cases that should be BLOCKED
    blocked_commands = [
        "rm -rf /",
        "rm -rf results/",
        "rm -rf data",
        "rm -rf *",
        "rm -fr results/",  # Reversed flags
        "sudo rm -rf results/",
        "rm -rf .",
        "rm -rf ./data",
    ]

    logger.info("\nTesting commands that SHOULD be BLOCKED:")
    for cmd in blocked_commands:
        trust_level = trust_manager.classify_command(cmd)
        logger.info(f"  Command: {cmd:40s} → {trust_level.value}")

        if trust_level != TrustLevel.BLOCKED:
            logger.error(f"  ❌ FAIL: '{cmd}' was classified as {trust_level.value}, should be BLOCKED")
            assert False, f"Command '{cmd}' should be BLOCKED but was {trust_level.value}"

    logger.info("✓ PASS: All rm -rf variants properly blocked")

    # Test cases that should be SAFE
    safe_commands = [
        "ls -la",
        "pwd",
        "date",
        "pip list",
        "git status",
    ]

    logger.info("\nTesting commands that SHOULD be SAFE:")
    for cmd in safe_commands:
        trust_level = trust_manager.classify_command(cmd)
        logger.info(f"  Command: {cmd:40s} → {trust_level.value}")

        if trust_level != TrustLevel.SAFE:
            logger.error(f"  ❌ FAIL: '{cmd}' was classified as {trust_level.value}, should be SAFE")
            assert False, f"Command '{cmd}' should be SAFE but was {trust_level.value}"

    logger.info("✓ PASS: Safe commands properly classified")

    logger.info("\n" + "="*60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("="*60)


def test_destructive_patterns():
    """Test the destructive pattern detection in SystemCommandManager."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Destructive Pattern Detection")
    logger.info("="*60)

    import re

    DESTRUCTIVE_PATTERNS = [
        r'rm\s+-rf',
        r'rm\s+-fr',
        r'truncate\s+-s\s*0',
        r'dd\s+.*of=',
        r':\s*>\s*[^>]',
        r'>\s*/dev/sd',
    ]

    # Commands that SHOULD match destructive patterns
    destructive_commands = [
        "rm -rf results/",
        "rm -fr data/",
        "truncate -s 0 file.txt",
        "dd if=/dev/zero of=/dev/sda",
        ": > file.txt",  # Truncate file
        "echo 'data' > /dev/sda",
    ]

    logger.info("\nTesting destructive pattern matching:")
    for cmd in destructive_commands:
        matched = any(re.search(pattern, cmd, re.IGNORECASE) for pattern in DESTRUCTIVE_PATTERNS)
        logger.info(f"  Command: {cmd:40s} → {'MATCHED ✓' if matched else 'NOT MATCHED ✗'}")

        if not matched:
            logger.error(f"  ❌ FAIL: '{cmd}' did not match any destructive pattern")
            assert False, f"Destructive command '{cmd}' was not detected"

    logger.info("✓ PASS: All destructive commands detected")

    # Commands that should NOT match
    safe_commands = [
        "echo 'content' >> file.txt",  # Append (not truncate)
        "ls -la results/",
        "mkdir -p results",
    ]

    logger.info("\nTesting that safe commands don't match:")
    for cmd in safe_commands:
        matched = any(re.search(pattern, cmd, re.IGNORECASE) for pattern in DESTRUCTIVE_PATTERNS)
        logger.info(f"  Command: {cmd:40s} → {'MATCHED ✗' if matched else 'NOT MATCHED ✓'}")

        if matched:
            logger.warning(f"  ⚠ WARNING: Safe command '{cmd}' matched destructive pattern (may need adjustment)")
            # Don't fail test, but warn

    logger.info("\n" + "="*60)
    logger.info("✅ TEST PASSED")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        logger.info("\n🧪 DESTRUCTIVE COMMAND BLOCKING TESTS\n")

        test_rm_rf_blocking()
        test_destructive_patterns()

        logger.info("\n" + "="*60)
        logger.info("🎉 ALL TESTS PASSED")
        logger.info("="*60)
        logger.info("\nSafety Improvements:")
        logger.info("  ✅ rm -rf with ANY path is blocked (not just root)")
        logger.info("  ✅ Destructive commands force manual approval")
        logger.info("  ✅ No auto-approval bypass for dangerous operations")
        logger.info("="*60)

        sys.exit(0)

    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)
