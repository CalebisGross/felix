"""
Centralized .felixignore support for Felix Framework.

Provides gitignore-style file exclusion across all Felix components:
- Knowledge Brain file scanning
- Workflow file discovery
- CLI document ingestion
- Chat file operations

Usage:
    from src.core.felixignore import should_ignore, load_felixignore

    # Load patterns (call once at startup)
    load_felixignore('/path/to/project')

    # Check if file should be ignored
    if should_ignore('/path/to/file.py'):
        skip_file()
"""

import fnmatch
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Singleton state
_ignore_patterns: Optional[List[str]] = None
_ignore_file_path: Optional[Path] = None

# Default patterns (always applied even without .felixignore file)
DEFAULT_PATTERNS = [
    # Virtual environments
    '.venv/',
    'venv/',
    'env/',
    # Python
    '__pycache__/',
    '*.pyc',
    '*.pyo',
    '.pytest_cache/',
    '.mypy_cache/',
    '.coverage',
    'htmlcov/',
    '*.egg-info/',
    'site-packages/',
    # Node.js
    'node_modules/',
    # Version control
    '.git/',
    # IDE
    '.vscode/',
    '.idea/',
    '*.swp',
    '*.swo',
    # Build artifacts
    'dist/',
    'build/',
    # OS files
    '.DS_Store',
    'Thumbs.db',
    # Logs
    '*.log',
]

# Protected paths (NEVER ignore these - system-critical files)
PROTECTED_PATTERNS = [
    'felix_*.db',        # System databases
    'config/*',          # Configuration files
    'pyproject.toml',    # Project config
    'requirements.txt',  # Dependencies
    '.felixignore',      # The ignore file itself
    'CLAUDE.md',         # Project instructions
]


def load_felixignore(root_path: str = '.') -> List[str]:
    """
    Load .felixignore file from root path.

    Args:
        root_path: Directory to look for .felixignore file

    Returns:
        List of all ignore patterns (defaults + user patterns)
    """
    global _ignore_patterns, _ignore_file_path

    root = Path(root_path).resolve()
    felixignore_path = root / '.felixignore'
    _ignore_file_path = felixignore_path

    # Start with default patterns
    patterns = list(DEFAULT_PATTERNS)

    # Load user patterns from .felixignore if exists
    if felixignore_path.exists():
        try:
            with open(felixignore_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    patterns.append(line)
            logger.info(f"Loaded .felixignore from {felixignore_path} ({len(patterns) - len(DEFAULT_PATTERNS)} user patterns)")
        except Exception as e:
            logger.warning(f"Failed to load .felixignore: {e}")
    else:
        logger.debug(f"No .felixignore found at {felixignore_path}, using defaults")

    _ignore_patterns = patterns
    return patterns


def should_ignore(file_path) -> bool:
    """
    Check if a file path should be ignored.

    Args:
        file_path: Path to check (str or Path object)

    Returns:
        True if file should be ignored, False otherwise
    """
    global _ignore_patterns

    # Lazy load patterns if not initialized
    if _ignore_patterns is None:
        load_felixignore()

    path_str = str(file_path)
    path_obj = Path(path_str)
    path_parts = path_obj.parts

    # Check protected patterns first (NEVER ignore these)
    for pattern in PROTECTED_PATTERNS:
        # Match against filename or path
        if fnmatch.fnmatch(path_obj.name, pattern):
            return False
        if fnmatch.fnmatch(path_str, f'*/{pattern}'):
            return False
        if fnmatch.fnmatch(path_str, pattern):
            return False

    # Check ignore patterns
    for pattern in _ignore_patterns:
        # Handle directory patterns (trailing /)
        if pattern.endswith('/'):
            dir_name = pattern.rstrip('/')
            # Check if any path component matches
            if dir_name in path_parts:
                logger.debug(f"Ignoring {path_str} (dir pattern: {pattern})")
                return True
        else:
            # Handle file patterns
            # Match against full path
            if fnmatch.fnmatch(path_str, f'*/{pattern}'):
                logger.debug(f"Ignoring {path_str} (pattern: {pattern})")
                return True
            if fnmatch.fnmatch(path_str, pattern):
                logger.debug(f"Ignoring {path_str} (pattern: {pattern})")
                return True
            # Match against filename only
            if fnmatch.fnmatch(path_obj.name, pattern):
                logger.debug(f"Ignoring {path_str} (filename pattern: {pattern})")
                return True
            # Match against any path component
            if any(fnmatch.fnmatch(part, pattern) for part in path_parts):
                logger.debug(f"Ignoring {path_str} (component pattern: {pattern})")
                return True

    return False


def get_patterns() -> List[str]:
    """Get current ignore patterns (defaults + user patterns)."""
    if _ignore_patterns is None:
        load_felixignore()
    return list(_ignore_patterns or [])


def get_default_patterns() -> List[str]:
    """Get default ignore patterns."""
    return list(DEFAULT_PATTERNS)


def get_protected_patterns() -> List[str]:
    """Get protected patterns (files that are never ignored)."""
    return list(PROTECTED_PATTERNS)


def reload(root_path: str = '.') -> List[str]:
    """
    Reload .felixignore from disk.

    Args:
        root_path: Directory to look for .felixignore file

    Returns:
        List of all ignore patterns after reload
    """
    global _ignore_patterns
    _ignore_patterns = None
    return load_felixignore(root_path)


def is_loaded() -> bool:
    """Check if patterns have been loaded."""
    return _ignore_patterns is not None


def get_ignore_file_path() -> Optional[Path]:
    """Get path to the .felixignore file (if loaded)."""
    return _ignore_file_path


def filter_command_output(stdout: str) -> str:
    """
    Filter file paths from command output that match .felixignore patterns.

    Used to prevent ignored paths from being stored in knowledge base
    when commands like 'find' or 'ls' are executed. This prevents data
    poisoning from .venv, __pycache__, and other ignored directories.

    Args:
        stdout: Raw command output string

    Returns:
        Filtered output with ignored paths removed
    """
    if not stdout:
        return stdout

    lines = stdout.split('\n')
    filtered_lines = []

    for line in lines:
        line_stripped = line.strip()

        # Check if line looks like a file path
        if line_stripped and (
            line_stripped.startswith('./') or
            line_stripped.startswith('/') or
            ('/' in line_stripped and not line_stripped.startswith('#'))
        ):
            # Filter out ignored paths
            if not should_ignore(line_stripped):
                filtered_lines.append(line)
            else:
                logger.debug(f"Filtered from command output: {line_stripped}")
        else:
            # Non-path lines pass through unchanged
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)
