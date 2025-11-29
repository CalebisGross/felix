"""
Migration Script: Convert AGENT_TOOLS_HEADER to Knowledge Store Entries

This script extracts tool-specific instructions from the static AGENT_TOOLS_HEADER
and stores them as retrievable knowledge entries in the knowledge store.

This enables the "subconscious memory" pattern where agents only receive
instructions for tools they actually need.

Usage:
    python scripts/migrate_tool_instructions.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Tool instruction sections extracted from AGENT_TOOLS_HEADER
TOOL_INSTRUCTIONS = {
    'web_search': {
        'name': 'Web Search',
        'keywords': ['web_search', 'WEB_SEARCH_NEEDED', 'current', 'real-time'],
        'instructions': """🔍 WEB SEARCH - USE THIS FOR CURRENT INFORMATION:
If you need current/real-time data (dates, times, recent events, latest stats), write EXACTLY:
WEB_SEARCH_NEEDED: [your query]

EXAMPLES:
✓ "WEB_SEARCH_NEEDED: current date and time"
✓ "WEB_SEARCH_NEEDED: 2024 election results"

⚠️ CRITICAL FORMATTING:
1. Write the pattern ON ITS OWN LINE or at the START of your response
2. Write ONLY the query after the colon - no explanation
3. DO NOT embed the pattern in sentences

✓ CORRECT:
"I need current information.
WEB_SEARCH_NEEDED: latest AI news"

✗ WRONG:
"I will use WEB_SEARCH_NEEDED: latest AI news to find..."
"""
    },

    'file_operations': {
        'name': 'File Operations',
        'keywords': ['file_operations', 'file', 'read', 'create', 'write', 'mkdir', 'echo', 'cat'],
        'instructions': """📁 FILE OPERATIONS - READ AND WRITE FILES:

⚠️ **CRITICAL: ALWAYS USE RELATIVE PATHS, NEVER ABSOLUTE PATHS**
   Use: results/file.txt ✅
   NOT: /results/file.txt ❌ (requires root permissions, will fail!)

📖 READ FILE CONTENTS:
✓ "SYSTEM_ACTION_NEEDED: cat [filepath]"  # Read entire file
✓ "SYSTEM_ACTION_NEEDED: head -n [N] [filepath]"  # Read first N lines (replace [N] with number)
✓ "SYSTEM_ACTION_NEEDED: tail -n [N] [filepath]"  # Read last N lines (replace [N] with number)
✓ "SYSTEM_ACTION_NEEDED: wc -l [filepath]"  # Count lines

🔍 DISCOVER FILE LOCATIONS (When path is unknown):
When you need a file but don't know its full path, DISCOVER it first using find:

✓ "SYSTEM_ACTION_NEEDED: find . -name 'filename.py' -type f"  # Find Python file
✓ "SYSTEM_ACTION_NEEDED: find . -name 'config.yaml' -type f"  # Find config file
✓ "SYSTEM_ACTION_NEEDED: find . -name '*.md' -type f"  # Find all markdown files

The find command will return the file's location. Then use that path with cat/head/tail.

Example reasoning flow:
"User wants central_post.py but I don't know the path.
SYSTEM_ACTION_NEEDED: find . -name 'central_post.py' -type f

After system returns './src/communication/central_post.py', I can read it:
SYSTEM_ACTION_NEEDED: cat ./src/communication/central_post.py"

📝 CHECK FILE/DIRECTORY EXISTS:
✓ "SYSTEM_ACTION_NEEDED: test -f src/agents/prompt_optimization.py && echo 'File exists' || echo 'File not found'"
✓ "SYSTEM_ACTION_NEEDED: test -d results && echo 'Directory exists' || echo 'Directory not found'"
✓ "SYSTEM_ACTION_NEEDED: ls -la src/agents/"  # List directory contents

CREATE DIRECTORY:
✓ "SYSTEM_ACTION_NEEDED: mkdir -p results/data"

CREATE FILE WITH CONTENT:
✓ 'SYSTEM_ACTION_NEEDED: echo "your content" > results/file.txt'  # Use double quotes!

APPEND TO FILE:
✓ 'SYSTEM_ACTION_NEEDED: echo "more content" >> results/log.txt'  # Use double quotes!

📝 SHELL QUOTING RULES - CRITICAL:

✅ CORRECT - Use DOUBLE QUOTES for content with apostrophes:
'SYSTEM_ACTION_NEEDED: echo "Testing agent\\'s work" > file.txt'

❌ WRONG - Single quotes break on apostrophes:
"SYSTEM_ACTION_NEEDED: echo 'agent's work' > file.txt"  # SYNTAX ERROR!

⚠️ ESCAPING RULES:
- Inside double quotes: escape $ ` \\\\ " with backslash
- Simple text: use double quotes
- Complex text: use printf

🧠 INTELLIGENT COMMAND PATTERNS:

✅ GOOD - Combine operations with && (ensures both happen):
"SYSTEM_ACTION_NEEDED: mkdir -p results && echo \\"content\\" > results/file.txt"

⚠️ FILE OVERWRITES - Preserve data:
❌ BAD: "SYSTEM_ACTION_NEEDED: echo \\"new\\" > existing_file.txt"  # Data loss!
✅ GOOD: "SYSTEM_ACTION_NEEDED: echo \\"content\\" >> file.txt"  # Append

KEY PRINCIPLES:
1. Check state before modifying (test -f, test -d)
2. Use idempotent operations (mkdir -p is safe)
3. Prefer append >> over overwrite >
4. Always use double quotes for text
5. Combine related operations with &&
"""
    },

    'system_commands': {
        'name': 'System Commands',
        'keywords': ['system_commands', 'SYSTEM_ACTION_NEEDED', 'command', 'execute'],
        'instructions': """🖥️ SYSTEM COMMANDS - EXECUTE SYSTEM OPERATIONS:

If you need to CHECK state, RUN COMMANDS, or MODIFY SYSTEM, write EXACTLY:
SYSTEM_ACTION_NEEDED: [command]

⚠️ CRITICAL FORMATTING RULES:
1. Write the pattern ON ITS OWN LINE or at the START of your response
2. Write ONLY the command after the colon - no explanation
3. DO NOT embed the pattern in sentences

✓ CORRECT FORMAT:
"I need to check the directory.
SYSTEM_ACTION_NEEDED: pwd"

OR:
"SYSTEM_ACTION_NEEDED: pwd
This will show the current directory."

✗ WRONG - DO NOT DO THIS:
"I will use SYSTEM_ACTION_NEEDED: pwd to check..."
"The command (SYSTEM_ACTION_NEEDED: pwd) will help..."

EXAMPLES:
✓ "SYSTEM_ACTION_NEEDED: date"     # Get current time/date
✓ "SYSTEM_ACTION_NEEDED: pwd"      # Get current directory
✓ "SYSTEM_ACTION_NEEDED: ls -la"   # List files
✓ "SYSTEM_ACTION_NEEDED: pip list" # Check installed packages

📝 MULTI-STEP WORKFLOWS:
For tasks requiring multiple commands, output multiple lines:

"SYSTEM_ACTION_NEEDED: cd /project/dir
SYSTEM_ACTION_NEEDED: ls -la
SYSTEM_ACTION_NEEDED: pwd"

NOTE: Commands requiring approval (mkdir, writes) will prompt user first.

SAFETY: Commands are classified as SAFE, REVIEW, or BLOCKED.
You'll receive results after execution.
"""
    }
}


def migrate_tool_instructions():
    """
    Migrate tool instructions from static header to knowledge store.

    Creates knowledge entries for each tool category with appropriate
    domain, keywords, and confidence levels.
    """
    logger.info("="*60)
    logger.info("TOOL INSTRUCTION MIGRATION")
    logger.info("="*60)

    # Initialize knowledge store
    try:
        knowledge_store = KnowledgeStore()
        logger.info("✓ Knowledge store initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize knowledge store: {e}")
        return False

    # Check if migration already done
    from src.memory.knowledge_store import KnowledgeQuery
    existing_entries = knowledge_store.retrieve_knowledge(
        KnowledgeQuery(domains=["tool_instructions"], limit=10)
    )

    if existing_entries:
        logger.warning(f"⚠ Found {len(existing_entries)} existing tool instruction entries")
        response = input("Do you want to overwrite them? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Migration cancelled by user")
            return False

        # Delete existing entries
        logger.info("Deleting existing tool instruction entries...")
        for entry in existing_entries:
            try:
                knowledge_store.delete_knowledge(entry.knowledge_id)
                logger.info(f"  ✓ Deleted entry {entry.knowledge_id}")
            except Exception as e:
                logger.warning(f"  ⚠ Could not delete entry {entry.knowledge_id}: {e}")

    # Migrate each tool instruction
    logger.info("")
    logger.info("Migrating tool instructions to knowledge store...")
    logger.info("")

    success_count = 0
    for tool_name, tool_data in TOOL_INSTRUCTIONS.items():
        logger.info(f"Migrating: {tool_data['name']} ({tool_name})")

        try:
            # Store instruction as knowledge entry
            entry_id = knowledge_store.store_knowledge(
                knowledge_type=KnowledgeType.TOOL_INSTRUCTION,
                content={
                    'tool_name': tool_name,
                    'instructions': tool_data['instructions'],
                    'keywords': tool_data['keywords']
                },
                confidence_level=ConfidenceLevel.HIGH,  # Tool instructions are always high confidence
                source_agent="migration_script",
                domain="tool_instructions",
                tags=tool_data['keywords']
            )

            if entry_id:
                logger.info(f"  ✓ Stored as knowledge entry #{entry_id}")
                logger.info(f"    Keywords: {', '.join(tool_data['keywords'])}")
                logger.info(f"    Size: {len(tool_data['instructions'])} characters")
                success_count += 1
            else:
                logger.error(f"  ✗ Failed to store {tool_name} (returned None)")

        except Exception as e:
            logger.error(f"  ✗ Error storing {tool_name}: {e}")

        logger.info("")

    # Summary
    logger.info("="*60)
    logger.info("MIGRATION COMPLETE")
    logger.info(f"  Success: {success_count}/{len(TOOL_INSTRUCTIONS)} tool instructions migrated")
    logger.info("="*60)

    return success_count == len(TOOL_INSTRUCTIONS)


if __name__ == "__main__":
    success = migrate_tool_instructions()
    sys.exit(0 if success else 1)
