"""
Prompt Loader for Felix Chat System

Loads system prompts from external files with:
- Template variable substitution
- File caching with modification time check
- Graceful fallback on errors
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Cache for loaded prompts
_prompt_cache: dict = {}
_cache_mtime: dict = {}


def get_config_path() -> Path:
    """Get the config directory path."""
    # Navigate from src/gui_ctk/tabs/chat/ to project root, then to config/
    return Path(__file__).parent.parent.parent.parent.parent / "config"


def load_system_prompt(
    filename: str = "chat_system_prompt.md",
    variables: Optional[dict] = None
) -> str:
    """
    Load and process system prompt from file.

    Args:
        filename: Name of the prompt file in config directory
        variables: Optional dict of template variables to substitute

    Returns:
        Processed prompt string with variables substituted
    """
    filepath = get_config_path() / filename

    # Check cache validity
    if filepath in _prompt_cache:
        try:
            current_mtime = filepath.stat().st_mtime
            if current_mtime == _cache_mtime.get(filepath):
                template = _prompt_cache[filepath]
                return substitute_variables(template, variables or {})
        except OSError:
            # File may have been deleted, clear cache
            _prompt_cache.pop(filepath, None)
            _cache_mtime.pop(filepath, None)

    # Load from file
    try:
        template = filepath.read_text(encoding='utf-8')
        _prompt_cache[filepath] = template
        _cache_mtime[filepath] = filepath.stat().st_mtime
        logger.info(f"Loaded system prompt from {filepath} ({len(template)} chars)")
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {filepath}")
        return get_fallback_prompt()
    except Exception as e:
        logger.warning(f"Failed to load prompt from {filepath}: {e}")
        return get_fallback_prompt()

    return substitute_variables(template, variables or {})


def substitute_variables(template: str, variables: dict) -> str:
    """
    Replace {{variable}} placeholders with values.

    Args:
        template: Prompt template with {{variable}} placeholders
        variables: Dict of variable names to values

    Returns:
        Template with all variables substituted
    """
    # Default variables for Felix
    defaults = {
        "currentDateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "userLocation": "Local (air-gapped)",
    }

    # Merge with provided variables (provided takes precedence)
    all_vars = {**defaults, **variables}

    result = template
    for key, value in all_vars.items():
        # Replace {{key}} with value
        result = result.replace(f"{{{{{key}}}}}", str(value))

    return result


def get_fallback_prompt() -> str:
    """
    Minimal fallback prompt if file loading fails.

    Returns:
        Basic system prompt for Felix
    """
    return """You are Felix, an air-gapped multi-agent AI assistant.

IDENTITY:
- You operate entirely offline with zero external dependencies
- You are NOT ChatGPT, GPT, Claude, or any OpenAI/Anthropic product. You are Felix.

CONSTRAINTS:
- OFFLINE-ONLY: No internet, no external APIs, no cloud services
- NO HALLUCINATION: Never fabricate information
- Be helpful, accurate, and concise

SYSTEM COMMAND EXECUTION:
To execute system commands, output: SYSTEM_ACTION_NEEDED: <command>
"""


def clear_cache() -> None:
    """Clear the prompt cache (useful for testing or hot-reloading)."""
    global _prompt_cache, _cache_mtime
    _prompt_cache.clear()
    _cache_mtime.clear()
    logger.debug("Prompt cache cleared")


def get_prompt_info(filename: str = "chat_system_prompt.md") -> dict:
    """
    Get information about a prompt file.

    Args:
        filename: Name of the prompt file

    Returns:
        Dict with file info (exists, size, cached, etc.)
    """
    filepath = get_config_path() / filename

    info = {
        "filepath": str(filepath),
        "exists": filepath.exists(),
        "cached": filepath in _prompt_cache,
        "size_bytes": None,
        "size_chars": None,
        "estimated_tokens": None,
    }

    if filepath.exists():
        try:
            stat = filepath.stat()
            info["size_bytes"] = stat.st_size

            # If cached, get char count
            if filepath in _prompt_cache:
                chars = len(_prompt_cache[filepath])
                info["size_chars"] = chars
                # Rough estimate: ~4 chars per token
                info["estimated_tokens"] = chars // 4
        except OSError:
            pass

    return info
