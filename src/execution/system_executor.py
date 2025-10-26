"""
System Command Executor with Safety Controls.

Handles actual command execution with:
- Timeout management
- Output streaming and capture
- Virtual environment detection
- Resource limits
- Error categorization
"""

import subprocess
import time
import os
import signal
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when command execution fails."""
    pass


class ErrorCategory(Enum):
    """Categories of execution errors."""
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    RESOURCE_LIMIT = "resource_limit"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass
class CommandResult:
    """Result of command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    success: bool
    error_category: Optional[ErrorCategory] = None
    cwd: str = ""
    venv_active: bool = False
    output_size: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'command': self.command,
            'exit_code': self.exit_code,
            'stdout': self.stdout[:1000] if self.stdout else "",  # Preview only
            'stderr': self.stderr[:1000] if self.stderr else "",
            'duration': self.duration,
            'success': self.success,
            'error_category': self.error_category.value if self.error_category else None,
            'cwd': self.cwd,
            'venv_active': self.venv_active,
            'output_size': self.output_size,
            'timestamp': self.timestamp
        }


class SystemExecutor:
    """
    Executes system commands with safety controls.

    Features:
    - Timeout management (default 5 minutes)
    - Output size limits (default 100MB)
    - Virtual environment detection
    - Error categorization
    - Environment variable management
    """

    def __init__(self,
                 default_timeout: float = 300.0,  # 5 minutes
                 max_output_size: int = 100 * 1024 * 1024,  # 100MB
                 default_cwd: Optional[Path] = None):
        """
        Initialize system executor.

        Args:
            default_timeout: Default command timeout in seconds
            max_output_size: Maximum output size in bytes
            default_cwd: Default working directory
        """
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.default_cwd = default_cwd or Path.cwd()

        logger.info(f"SystemExecutor initialized:")
        logger.info(f"  Default timeout: {default_timeout}s")
        logger.info(f"  Max output size: {max_output_size / (1024*1024):.1f}MB")
        logger.info(f"  Default CWD: {self.default_cwd}")

    def execute_command(self,
                       command: str,
                       timeout: Optional[float] = None,
                       cwd: Optional[Path] = None,
                       env: Optional[Dict[str, str]] = None,
                       context: str = "") -> CommandResult:
        """
        Execute a system command with safety controls.

        Args:
            command: Command to execute
            timeout: Command timeout in seconds (None = default)
            cwd: Working directory (None = default)
            env: Environment variables (None = inherit)
            context: Context/reason for command execution

        Returns:
            CommandResult with execution details

        Raises:
            ExecutionError: If execution fails critically
        """
        timeout = timeout or self.default_timeout
        cwd = cwd or self.default_cwd

        logger.info(f"Executing command: {command}")
        if context:
            logger.info(f"  Context: {context}")
        logger.info(f"  CWD: {cwd}")
        logger.info(f"  Timeout: {timeout}s")

        # Detect if venv is active
        venv_active = self.is_venv_active()
        if venv_active:
            logger.info(f"  Virtual environment: ACTIVE")

        start_time = time.time()

        try:
            # Prepare environment
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            # Execute command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
                env=exec_env,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None  # Unix: new process group
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
                exit_code = process.returncode

            except subprocess.TimeoutExpired:
                logger.warning(f"Command timed out after {timeout}s")

                # Kill process group
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()

                stdout, stderr = process.communicate()

                duration = time.time() - start_time

                return CommandResult(
                    command=command,
                    exit_code=-1,
                    stdout=stdout or "",
                    stderr=stderr or "Command timed out",
                    duration=duration,
                    success=False,
                    error_category=ErrorCategory.TIMEOUT,
                    cwd=str(cwd),
                    venv_active=venv_active,
                    output_size=len(stdout or "") + len(stderr or "")
                )

            duration = time.time() - start_time
            output_size = len(stdout) + len(stderr)

            # Check output size
            if output_size > self.max_output_size:
                logger.warning(f"Output size ({output_size} bytes) exceeds limit ({self.max_output_size} bytes)")
                stdout = stdout[:self.max_output_size // 2]
                stderr = stderr[:self.max_output_size // 2]

            # Determine success and categorize errors
            success = exit_code == 0
            error_category = None

            if not success:
                error_category = self._categorize_error(exit_code, stderr, stdout)
                logger.warning(f"Command failed: exit_code={exit_code}, category={error_category.value}")
            else:
                logger.info(f"✓ Command succeeded in {duration:.2f}s")

            return CommandResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                success=success,
                error_category=error_category,
                cwd=str(cwd),
                venv_active=venv_active,
                output_size=output_size
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Execution failed with exception: {e}")

            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=duration,
                success=False,
                error_category=ErrorCategory.UNKNOWN,
                cwd=str(cwd),
                venv_active=venv_active,
                output_size=0
            )

    def _categorize_error(self, exit_code: int, stderr: str, stdout: str) -> ErrorCategory:
        """
        Categorize error based on exit code and output.

        Args:
            exit_code: Process exit code
            stderr: Standard error output
            stdout: Standard output

        Returns:
            ErrorCategory enum
        """
        error_text = (stderr + stdout).lower()

        # Permission errors
        if 'permission denied' in error_text or 'access denied' in error_text:
            return ErrorCategory.PERMISSION

        # Command not found
        if 'command not found' in error_text or 'not recognized' in error_text:
            return ErrorCategory.NOT_FOUND

        # Syntax errors
        if 'syntax error' in error_text or 'invalid syntax' in error_text:
            return ErrorCategory.SYNTAX_ERROR

        # Network errors
        if 'connection refused' in error_text or 'network' in error_text or 'timeout' in error_text:
            return ErrorCategory.NETWORK_ERROR

        # Resource limits
        if 'out of memory' in error_text or 'disk full' in error_text:
            return ErrorCategory.RESOURCE_LIMIT

        # Default to runtime error
        if exit_code != 0:
            return ErrorCategory.RUNTIME_ERROR

        return ErrorCategory.UNKNOWN

    def is_venv_active(self) -> bool:
        """
        Check if a virtual environment is currently active.

        Returns:
            True if venv is active, False otherwise
        """
        # Check VIRTUAL_ENV environment variable
        if os.environ.get('VIRTUAL_ENV'):
            return True

        # Check if sys.prefix differs from sys.base_prefix (Python 3.3+)
        import sys
        return hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

    def detect_venv_path(self, cwd: Optional[Path] = None) -> Optional[Path]:
        """
        Detect virtual environment in current or parent directories.

        Args:
            cwd: Directory to start search from

        Returns:
            Path to venv activate script, or None if not found
        """
        cwd = cwd or self.default_cwd

        # Common venv names
        venv_names = ['.venv', 'venv', 'env', 'virtualenv']

        # Check current directory and parents
        current = Path(cwd).absolute()

        for _ in range(5):  # Check up to 5 levels up
            for venv_name in venv_names:
                venv_path = current / venv_name

                # Check for activate script
                if os.name != 'nt':
                    activate_script = venv_path / 'bin' / 'activate'
                else:
                    activate_script = venv_path / 'Scripts' / 'activate.bat'

                if activate_script.exists():
                    logger.info(f"Found venv at: {venv_path}")
                    return activate_script

            # Move to parent
            parent = current.parent
            if parent == current:  # Reached root
                break
            current = parent

        return None

    def get_venv_activation_command(self, cwd: Optional[Path] = None) -> Optional[str]:
        """
        Get command to activate virtual environment.

        Args:
            cwd: Directory to search from

        Returns:
            Activation command string, or None if no venv found
        """
        activate_script = self.detect_venv_path(cwd)

        if not activate_script:
            return None

        if os.name != 'nt':
            return f"source {activate_script}"
        else:
            return str(activate_script)

    def get_system_state(self) -> Dict[str, Any]:
        """
        Get current system state information.

        Returns:
            Dictionary with system state details
        """
        return {
            'cwd': str(Path.cwd()),
            'venv_active': self.is_venv_active(),
            'venv_path': os.environ.get('VIRTUAL_ENV', None),
            'user': os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
            'home': str(Path.home()),
            'python_executable': os.sys.executable
        }

    def compute_command_hash(self, command: str) -> str:
        """
        Compute hash of command for deduplication.

        Args:
            command: Command string

        Returns:
            SHA256 hash (first 16 characters)
        """
        return hashlib.sha256(command.encode()).hexdigest()[:16]
