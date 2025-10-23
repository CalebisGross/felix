#!/usr/bin/env python3
"""
Simple test to check Felix system initialization.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.gui.felix_system import FelixSystem, FelixConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_init():
    """Test basic initialization."""
    print("=== Starting test ===")
    logger.info("Creating Felix config...")

    config = FelixConfig(
        memory_db_path="test_init.db",
        knowledge_db_path="test_init_knowledge.db",
        web_search_enabled=True,
        verbose_llm_logging=False  # Reduce noise
    )

    logger.info("Creating Felix system...")
    felix_system = FelixSystem(config)

    logger.info("Starting Felix system...")
    success = felix_system.start()

    if success:
        print("✓ Felix system started successfully")
        logger.info("Stopping Felix system...")
        felix_system.stop()
        print("✓ Felix system stopped")
        return True
    else:
        print("✗ Felix system failed to start")
        return False

if __name__ == "__main__":
    try:
        result = test_init()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        for db in ["test_init.db", "test_init_knowledge.db"]:
            if Path(db).exists():
                Path(db).unlink()
