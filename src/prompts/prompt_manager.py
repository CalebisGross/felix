"""
Prompt Manager for Felix Framework.

Hybrid YAML + Database system for managing agent prompts with:
- Default prompts from YAML (version controlled)
- Custom overrides in SQLite database
- Version history and performance tracking
- Runtime caching for performance
"""

import sqlite3
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    key: str
    template: str
    description: str = ""
    source: str = "yaml"  # "yaml", "database", or "fallback"
    version: int = 1
    depth_range: Optional[Tuple[float, float]] = None
    strict_mode: Optional[bool] = None


@dataclass
class PromptPerformance:
    """Performance metrics for a prompt version."""
    prompt_key: str
    version: int
    avg_confidence: float
    avg_tokens: float
    avg_processing_time: float
    usage_count: int


class PromptManager:
    """
    Manages agent prompts with hybrid YAML + Database storage.

    Lookup priority:
    1. Runtime cache (in-memory)
    2. Database custom overrides
    3. YAML default templates
    4. Python fallback (hardcoded)
    """

    # Known agent types for robust key parsing
    # Keys like "research_exploration_normal" are parsed as agent_type="research", sub_key="exploration_normal"
    # This registry prevents issues with agent types that might contain underscores
    KNOWN_AGENT_TYPES = frozenset({
        "research",
        "analysis",
        "critic",
        "system",
        "synthesis",
        "direct",     # For direct mode
        "workflow",   # For workflow-level prompts
    })

    def __init__(self, yaml_path: str = "config/prompts.yaml",
                 db_path: str = "prompts/felix_prompts.db"):
        """
        Initialize PromptManager.

        Args:
            yaml_path: Path to YAML file with default prompts
            db_path: Path to SQLite database for custom prompts
        """
        self.yaml_path = Path(yaml_path)
        self.db_path = Path(db_path)
        self._cache: Dict[str, PromptTemplate] = {}
        self._yaml_data: Optional[Dict[str, Any]] = None

        # Initialize database
        self._init_database()

        # Load YAML on initialization
        self._load_yaml()

        logger.info(f"PromptManager initialized with YAML: {self.yaml_path}, DB: {self.db_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Custom prompt overrides with version history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_key TEXT NOT NULL,
                template_text TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)

        # Prompt performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_key TEXT NOT NULL,
                version INTEGER NOT NULL,
                agent_id TEXT,
                confidence REAL,
                tokens_used INTEGER,
                processing_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Prompt version history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_key TEXT NOT NULL,
                version INTEGER NOT NULL,
                template_text TEXT NOT NULL,
                action TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt_key ON custom_prompts(prompt_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON custom_prompts(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance ON prompt_performance(prompt_key, version)")

        conn.commit()
        conn.close()

        logger.debug(f"Database initialized at {self.db_path}")

    def _load_yaml(self) -> None:
        """Load default prompts from YAML file."""
        if not self.yaml_path.exists():
            logger.warning(f"YAML file not found: {self.yaml_path}")
            self._yaml_data = {}
            return

        try:
            with open(self.yaml_path, 'r') as f:
                self._yaml_data = yaml.safe_load(f)
            logger.info(f"Loaded prompts from {self.yaml_path}")
        except Exception as e:
            logger.error(f"Failed to load YAML: {e}")
            self._yaml_data = {}

    def get_prompt(self, prompt_key: str) -> Optional[PromptTemplate]:
        """
        Get prompt template with priority: cache → DB → YAML → fallback.

        Args:
            prompt_key: Key identifying the prompt (e.g., "research_exploration_normal")

        Returns:
            PromptTemplate or None if not found
        """
        # 1. Check cache
        if prompt_key in self._cache:
            logger.debug(f"Prompt '{prompt_key}' found in cache")
            return self._cache[prompt_key]

        # 2. Check database for custom override
        custom = self._get_from_database(prompt_key)
        if custom:
            logger.debug(f"Prompt '{prompt_key}' loaded from database")
            self._cache[prompt_key] = custom
            return custom

        # 3. Load from YAML default
        default = self._get_from_yaml(prompt_key)
        if default:
            logger.debug(f"Prompt '{prompt_key}' loaded from YAML")
            self._cache[prompt_key] = default
            return default

        # 4. Not found
        logger.warning(f"Prompt '{prompt_key}' not found in any source")
        return None

    def _get_from_database(self, prompt_key: str) -> Optional[PromptTemplate]:
        """Get custom prompt from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT prompt_key, template_text, version, notes
            FROM custom_prompts
            WHERE prompt_key = ? AND is_active = 1
            ORDER BY version DESC
            LIMIT 1
        """, (prompt_key,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return PromptTemplate(
                key=row[0],
                template=row[1],
                version=row[2],
                description=row[3] or "",
                source="database"
            )
        return None

    def _parse_prompt_key(self, prompt_key: str) -> Optional[Tuple[str, str]]:
        """
        Parse a prompt key into (agent_type, sub_key) using known types registry.

        This method uses KNOWN_AGENT_TYPES to safely parse keys, avoiding issues
        with agent types that might contain underscores.

        Args:
            prompt_key: Key like "research_exploration_normal"

        Returns:
            Tuple of (agent_type, sub_key) or None if parsing fails
        """
        parts = prompt_key.split('_')
        if len(parts) < 2:
            return None

        # Try to match against known agent types (prefer longest match)
        # This handles cases where future agent types might have underscores
        for i in range(len(parts), 0, -1):
            potential_type = '_'.join(parts[:i])
            if potential_type in self.KNOWN_AGENT_TYPES:
                agent_type = potential_type
                sub_key = '_'.join(parts[i:]) if i < len(parts) else ""
                if sub_key:  # Must have a sub_key
                    return (agent_type, sub_key)

        # Fall back to simple split (first part is agent_type)
        # This maintains backward compatibility with unknown agent types
        agent_type = parts[0]
        sub_key = '_'.join(parts[1:])

        return (agent_type, sub_key) if sub_key else None

    def _get_from_yaml(self, prompt_key: str) -> Optional[PromptTemplate]:
        """Get default prompt from YAML."""
        logger.debug(f"Looking up YAML key '{prompt_key}'")

        if not self._yaml_data:
            logger.debug(f"YAML data is empty, returning None")
            return None

        # Parse prompt_key using known types registry
        parsed = self._parse_prompt_key(prompt_key)
        if not parsed:
            logger.debug(f"Key '{prompt_key}' could not be parsed")
            return None

        agent_type, sub_key = parsed
        logger.debug(f"Parsed key: agent_type='{agent_type}', sub_key='{sub_key}'")

        # Navigate YAML structure
        if agent_type not in self._yaml_data:
            available_types = list(self._yaml_data.keys()) if self._yaml_data else []
            logger.debug(f"Agent type '{agent_type}' not in YAML. Available: {available_types}")
            return None

        agent_prompts = self._yaml_data[agent_type]
        if sub_key not in agent_prompts:
            available_subkeys = list(agent_prompts.keys()) if isinstance(agent_prompts, dict) else []
            logger.debug(f"Sub-key '{sub_key}' not in {agent_type}. Available: {available_subkeys}")
            return None

        logger.debug(f"Found {agent_type}.{sub_key} in YAML")

        prompt_data = agent_prompts[sub_key]
        if isinstance(prompt_data, dict) and 'template' in prompt_data:
            # Extract metadata if available
            depth_range = prompt_data.get('depth_range')
            if depth_range:
                depth_range = tuple(depth_range)

            return PromptTemplate(
                key=prompt_key,
                template=prompt_data['template'],
                description=prompt_data.get('description', ''),
                source="yaml",
                depth_range=depth_range,
                strict_mode=prompt_data.get('strict_mode')
            )

        return None

    def save_custom_prompt(self, prompt_key: str, template: str, notes: str = "") -> int:
        """
        Save custom prompt to database with versioning.

        Args:
            prompt_key: Key identifying the prompt
            template: Prompt template text
            notes: Optional notes about this version

        Returns:
            Version number of saved prompt
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next version number
        cursor.execute("""
            SELECT COALESCE(MAX(version), 0) + 1
            FROM custom_prompts
            WHERE prompt_key = ?
        """, (prompt_key,))
        next_version = cursor.fetchone()[0]

        # Deactivate previous versions
        cursor.execute("""
            UPDATE custom_prompts
            SET is_active = 0
            WHERE prompt_key = ? AND is_active = 1
        """, (prompt_key,))

        # Insert new version
        cursor.execute("""
            INSERT INTO custom_prompts (prompt_key, template_text, version, notes)
            VALUES (?, ?, ?, ?)
        """, (prompt_key, template, next_version, notes))

        # Record in history
        cursor.execute("""
            INSERT INTO prompt_history (prompt_key, version, template_text, action, notes)
            VALUES (?, ?, ?, 'edited', ?)
        """, (prompt_key, next_version, template, notes))

        conn.commit()
        conn.close()

        # Invalidate cache
        if prompt_key in self._cache:
            del self._cache[prompt_key]

        logger.info(f"Saved custom prompt '{prompt_key}' version {next_version}")
        return next_version

    def reset_to_default(self, prompt_key: str) -> bool:
        """
        Reset prompt to YAML default by deactivating custom versions.

        Args:
            prompt_key: Key identifying the prompt

        Returns:
            True if reset successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Deactivate all custom versions
        cursor.execute("""
            UPDATE custom_prompts
            SET is_active = 0
            WHERE prompt_key = ?
        """, (prompt_key,))

        # Record reset in history
        cursor.execute("""
            INSERT INTO prompt_history (prompt_key, version, template_text, action)
            VALUES (?, 0, '', 'reset_to_default')
        """, (prompt_key,))

        conn.commit()
        conn.close()

        # Invalidate cache
        if prompt_key in self._cache:
            del self._cache[prompt_key]

        logger.info(f"Reset prompt '{prompt_key}' to YAML default")
        return True

    def list_all_prompts(self) -> List[Tuple[str, str, str]]:
        """
        List all available prompts from YAML and database.

        Returns:
            List of tuples: (prompt_key, description, source)
        """
        prompts = []

        # Get YAML prompts
        if self._yaml_data:
            for agent_type, agent_prompts in self._yaml_data.items():
                if agent_type in ['version', 'last_updated', 'variables', 'metadata']:
                    continue

                for sub_key, prompt_data in agent_prompts.items():
                    if isinstance(prompt_data, dict) and 'template' in prompt_data:
                        prompt_key = f"{agent_type}_{sub_key}"
                        description = prompt_data.get('description', '')

                        # Check if has custom override
                        has_custom = self.has_custom_prompt(prompt_key)
                        source = "custom" if has_custom else "yaml"

                        prompts.append((prompt_key, description, source))

        return sorted(prompts)

    def has_custom_prompt(self, prompt_key: str) -> bool:
        """Check if prompt has custom override in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM custom_prompts
            WHERE prompt_key = ? AND is_active = 1
        """, (prompt_key,))

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def get_prompt_history(self, prompt_key: str) -> List[Dict[str, Any]]:
        """
        Get version history for a prompt.

        Args:
            prompt_key: Key identifying the prompt

        Returns:
            List of history entries with version, action, timestamp
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT version, template_text, action, timestamp, notes
            FROM prompt_history
            WHERE prompt_key = ?
            ORDER BY timestamp DESC
        """, (prompt_key,))

        history = []
        for row in cursor.fetchall():
            history.append({
                'version': row[0],
                'template_text': row[1],
                'action': row[2],
                'timestamp': row[3],
                'notes': row[4] or ''
            })

        conn.close()
        return history

    def record_performance(self, prompt_key: str, version: int, agent_id: str,
                          confidence: float, tokens_used: int, processing_time: float) -> None:
        """
        Record prompt performance metrics for tracking.

        Args:
            prompt_key: Key identifying the prompt
            version: Version number of prompt used
            agent_id: ID of agent that used this prompt
            confidence: Confidence score achieved
            tokens_used: Tokens consumed
            processing_time: Processing time in seconds
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO prompt_performance
            (prompt_key, version, agent_id, confidence, tokens_used, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (prompt_key, version, agent_id, confidence, tokens_used, processing_time))

        conn.commit()
        conn.close()

        logger.debug(f"Recorded performance for '{prompt_key}' v{version}: confidence={confidence:.3f}")

    def get_performance_stats(self, prompt_key: str) -> List[PromptPerformance]:
        """
        Get performance statistics for all versions of a prompt.

        Args:
            prompt_key: Key identifying the prompt

        Returns:
            List of PromptPerformance objects per version
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                version,
                AVG(confidence) as avg_confidence,
                AVG(tokens_used) as avg_tokens,
                AVG(processing_time) as avg_time,
                COUNT(*) as count
            FROM prompt_performance
            WHERE prompt_key = ?
            GROUP BY version
            ORDER BY version DESC
        """, (prompt_key,))

        stats = []
        for row in cursor.fetchall():
            stats.append(PromptPerformance(
                prompt_key=prompt_key,
                version=row[0],
                avg_confidence=row[1],
                avg_tokens=row[2],
                avg_processing_time=row[3],
                usage_count=row[4]
            ))

        conn.close()
        return stats

    def render_template(self, template: str, **variables) -> str:
        """
        Render prompt template with variable substitution.

        Args:
            template: Template string with {variable} placeholders
            **variables: Variables to substitute

        Returns:
            Rendered prompt string
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}")
            return template
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return template

    def clear_cache(self) -> None:
        """Clear runtime cache."""
        self._cache.clear()
        logger.debug("Prompt cache cleared")

    def reload_yaml(self) -> None:
        """Reload YAML file and clear cache."""
        self._load_yaml()
        self.clear_cache()
        logger.info("YAML reloaded and cache cleared")

    def get_system_chat_identity(self, variables: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Get Felix's chat identity system prompt.

        Priority:
        1. Database custom override (key: "system_chat_identity")
        2. Markdown file (config/chat_system_prompt.md)
        3. Fallback minimal identity

        Args:
            variables: Optional template variables (e.g., {"currentDateTime": "2025-12-18 10:00:00"})

        Returns:
            System prompt text with variables substituted, or None if not found
        """
        # 1. Check database for custom override
        custom = self._get_from_database("system_chat_identity")
        if custom:
            logger.debug("Chat identity loaded from database override")
            template = custom.template
            if variables:
                return self.render_template(template, **variables)
            return template

        # 2. Load from markdown file (default)
        project_root = self.yaml_path.parent.parent
        prompt_path = project_root / "config" / "chat_system_prompt.md"

        if prompt_path.exists():
            try:
                content = prompt_path.read_text(encoding='utf-8')
                logger.debug(f"Chat identity loaded from {prompt_path}")

                # Apply template variables if provided
                if variables:
                    # Replace {{variable}} style placeholders (markdown style)
                    for key, value in variables.items():
                        content = content.replace(f"{{{{{key}}}}}", str(value))

                return content
            except Exception as e:
                logger.warning(f"Failed to load chat identity from file: {e}")

        # 3. Not found
        logger.warning("Chat identity not found in database or file")
        return None
