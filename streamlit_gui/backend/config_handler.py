"""
Configuration handler with error handling for Felix Framework.

Provides robust configuration management with validation and fallbacks.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigHandler:
    """
    Handle configuration loading, validation, and management
    with comprehensive error handling.
    """

    DEFAULT_CONFIG = {
        'helix': {
            'top_radius': 3.0,
            'bottom_radius': 0.5,
            'height': 8.0,
            'turns': 2
        },
        'lm_host': '127.0.0.1',
        'lm_port': 1234,
        'max_agents': 25,
        'base_token_budget': 2500,
        'spawning': {
            'confidence_threshold': 0.75,
            'max_depth': 5,
            'spawn_delay': 0.5
        },
        'temperature': {
            'top': 1.0,
            'bottom': 0.2
        },
        'timeout': 30,
        'log_level': 'INFO'
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config handler.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file with error handling.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded configuration
        """
        try:
            path = Path(config_path)

            if not path.exists():
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return self.config

            with open(path, 'r') as f:
                if path.suffix == '.json':
                    loaded_config = json.load(f)
                elif path.suffix in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported config format: {path.suffix}")
                    return self.config

            # Merge with defaults
            self.config = self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
            logger.info(f"Successfully loaded config from: {config_path}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

        return self.config

    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge override config with defaults.

        Args:
            default: Default configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = default.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate current configuration.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required fields
        required = ['helix', 'lm_host', 'lm_port']
        for field in required:
            if field not in self.config:
                issues.append(f"Missing required field: {field}")

        # Validate helix parameters
        if 'helix' in self.config:
            helix = self.config['helix']

            if helix.get('top_radius', 0) <= helix.get('bottom_radius', 1):
                issues.append("Helix top_radius must be greater than bottom_radius")

            if helix.get('height', 0) <= 0:
                issues.append("Helix height must be positive")

            if helix.get('turns', 0) <= 0:
                issues.append("Helix turns must be positive")

        # Validate network parameters
        if not isinstance(self.config.get('lm_port'), int):
            issues.append("lm_port must be an integer")
        elif not 1 <= self.config.get('lm_port', 0) <= 65535:
            issues.append("lm_port must be between 1 and 65535")

        # Validate agent parameters
        if self.config.get('max_agents', 0) < 1:
            issues.append("max_agents must be at least 1")

        if self.config.get('base_token_budget', 0) < 100:
            issues.append("base_token_budget should be at least 100")

        is_valid = len(issues) == 0
        return is_valid, issues

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            parts = key.split('.')
            value = self.config

            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                    if value is None:
                        return default
                else:
                    return default

            return value

        except Exception as e:
            logger.debug(f"Error getting config value for {key}: {e}")
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        try:
            parts = key.split('.')
            config = self.config

            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]

            config[parts[-1]] = value
            logger.debug(f"Set config {key} = {value}")

        except Exception as e:
            logger.error(f"Error setting config value for {key}: {e}")

    def save_config(self, filepath: str, format: str = 'yaml'):
        """
        Save current configuration to file.

        Args:
            filepath: Path to save configuration
            format: Format to save ('yaml' or 'json')
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                if format == 'json':
                    json.dump(self.config, f, indent=2)
                else:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved configuration to: {filepath}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        logger.info("Configuration reset to defaults")

    def get_safe_config(self) -> Dict[str, Any]:
        """
        Get configuration with sensitive values masked.

        Returns:
            Safe configuration for display
        """
        safe_config = self.config.copy()

        # Mask sensitive fields
        sensitive_fields = ['api_key', 'secret', 'password', 'token']

        def mask_sensitive(d: dict):
            for key, value in d.items():
                if any(field in key.lower() for field in sensitive_fields):
                    d[key] = '***MASKED***'
                elif isinstance(value, dict):
                    mask_sensitive(value)

        mask_sensitive(safe_config)
        return safe_config