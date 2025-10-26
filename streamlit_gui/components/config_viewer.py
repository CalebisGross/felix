"""Configuration viewer component for Felix Framework."""

import streamlit as st
import yaml
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import difflib


def display_config_tree(config: Dict[str, Any], path: str = "", level: int = 0):
    """
    Display configuration as an expandable tree structure.

    Args:
        config: Configuration dictionary
        path: Current path in configuration
        level: Current nesting level
    """
    indent = "  " * level

    for key, value in config.items():
        current_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            # Nested dictionary - create expandable section
            with st.expander(f"{indent}📁 {key}"):
                display_config_tree(value, current_path, level + 1)
        elif isinstance(value, list):
            # List - display items
            st.markdown(f"{indent}📋 **{key}**: [{len(value)} items]")
            if st.checkbox(f"Show items", key=f"show_{current_path}"):
                for idx, item in enumerate(value):
                    st.text(f"{indent}  [{idx}]: {item}")
        else:
            # Leaf value - display directly
            st.markdown(f"{indent}🔹 **{key}**: `{value}`")


def validate_configuration(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate configuration against expected schema.

    Args:
        config: Configuration to validate

    Returns:
        List of validation issues
    """
    issues = []

    # Check required fields
    required_fields = [
        'helix.top_radius',
        'helix.bottom_radius',
        'helix.height',
        'helix.turns',
        'lm_host',
        'lm_port'
    ]

    for field in required_fields:
        parts = field.split('.')
        current = config
        found = True

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break

        if not found:
            issues.append({
                'level': 'error',
                'field': field,
                'message': f"Required field '{field}' is missing"
            })

    # Validate value ranges
    validations = [
        ('helix.top_radius', 0.5, 10.0, 'Helix top radius'),
        ('helix.bottom_radius', 0.1, 5.0, 'Helix bottom radius'),
        ('helix.height', 1.0, 20.0, 'Helix height'),
        ('helix.turns', 0.5, 5.0, 'Helix turns'),
        ('max_agents', 1, 100, 'Maximum agents'),
        ('base_token_budget', 100, 10000, 'Token budget')
    ]

    for field_path, min_val, max_val, name in validations:
        parts = field_path.split('.')
        current = config
        found = True

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break

        if found and current is not None:
            try:
                value = float(current)
                if value < min_val or value > max_val:
                    issues.append({
                        'level': 'warning',
                        'field': field_path,
                        'message': f"{name} ({value}) is outside recommended range [{min_val}, {max_val}]"
                    })
            except (TypeError, ValueError):
                issues.append({
                    'level': 'error',
                    'field': field_path,
                    'message': f"{name} must be a number"
                })

    # Check logical constraints
    helix = config.get('helix', {})
    if helix.get('top_radius', 0) <= helix.get('bottom_radius', 1):
        issues.append({
            'level': 'warning',
            'field': 'helix',
            'message': "Top radius should be larger than bottom radius for proper tapering"
        })

    return issues


def display_validation_results(issues: List[Dict[str, Any]]):
    """
    Display configuration validation results.

    Args:
        issues: List of validation issues
    """
    if not issues:
        st.success("✅ Configuration is valid")
        return

    # Group by severity
    errors = [i for i in issues if i['level'] == 'error']
    warnings = [i for i in issues if i['level'] == 'warning']

    if errors:
        with st.expander(f"❌ Errors ({len(errors)})", expanded=True):
            for issue in errors:
                st.error(f"**{issue['field']}**: {issue['message']}")

    if warnings:
        with st.expander(f"⚠️ Warnings ({len(warnings)})"):
            for issue in warnings:
                st.warning(f"**{issue['field']}**: {issue['message']}")


def generate_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of key configuration parameters.

    Args:
        config: Full configuration

    Returns:
        Configuration summary
    """
    summary = {
        "System": {
            "LM Studio Host": config.get('lm_host', 'N/A'),
            "LM Studio Port": config.get('lm_port', 'N/A'),
            "Model": config.get('model', 'default')
        },
        "Helix Geometry": {
            "Top Radius": config.get('helix', {}).get('top_radius', 'N/A'),
            "Bottom Radius": config.get('helix', {}).get('bottom_radius', 'N/A'),
            "Height": config.get('helix', {}).get('height', 'N/A'),
            "Turns": config.get('helix', {}).get('turns', 'N/A')
        },
        "Agent Settings": {
            "Max Agents": config.get('max_agents', 'N/A'),
            "Token Budget": config.get('base_token_budget', 'N/A'),
            "Spawn Threshold": config.get('spawning', {}).get('confidence_threshold', 'N/A')
        },
        "Performance": {
            "Timeout": config.get('timeout', 30),
            "Max Retries": config.get('max_retries', 3),
            "Batch Size": config.get('batch_size', 10)
        }
    }

    return summary


def display_config_summary(config: Dict[str, Any]):
    """
    Display configuration summary in a grid layout.

    Args:
        config: Configuration dictionary
    """
    summary = generate_config_summary(config)

    # Display in columns
    cols = st.columns(len(summary))

    for idx, (section, values) in enumerate(summary.items()):
        with cols[idx]:
            st.markdown(f"### {section}")
            for key, value in values.items():
                st.text(f"{key}: {value}")


def create_config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> str:
    """
    Create a visual diff between two configurations.

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        HTML diff string
    """
    # Convert to YAML for better readability
    yaml1 = yaml.dump(config1, default_flow_style=False, sort_keys=True)
    yaml2 = yaml.dump(config2, default_flow_style=False, sort_keys=True)

    # Generate diff
    diff = difflib.unified_diff(
        yaml1.splitlines(keepends=True),
        yaml2.splitlines(keepends=True),
        fromfile='Configuration 1',
        tofile='Configuration 2',
        n=3
    )

    # Format diff with colors
    diff_html = []
    for line in diff:
        if line.startswith('+'):
            diff_html.append(f'<span style="color: green;">{line}</span>')
        elif line.startswith('-'):
            diff_html.append(f'<span style="color: red;">{line}</span>')
        elif line.startswith('@'):
            diff_html.append(f'<span style="color: blue; font-weight: bold;">{line}</span>')
        else:
            diff_html.append(line)

    return '<pre>' + ''.join(diff_html) + '</pre>'


def merge_configurations(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations with override taking precedence.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configurations(result[key], value)
        else:
            # Override value
            result[key] = value

    return result


def export_config_template() -> str:
    """
    Generate a configuration template with all available options.

    Returns:
        YAML template string
    """
    template = {
        "helix": {
            "top_radius": 3.0,
            "bottom_radius": 0.5,
            "height": 8.0,
            "turns": 2,
            "_comment": "Helix geometry defines the agent progression path"
        },
        "agents": {
            "max_agents": 25,
            "base_token_budget": 2500,
            "roles": ["research", "analysis", "synthesis", "critic"],
            "_comment": "Agent configuration and resource limits"
        },
        "spawning": {
            "confidence_threshold": 0.75,
            "max_depth": 5,
            "spawn_delay": 0.5,
            "adaptive": True,
            "_comment": "Dynamic agent spawning parameters"
        },
        "llm": {
            "lm_host": "127.0.0.1",
            "lm_port": 1234,
            "model": "default",
            "timeout": 30,
            "max_retries": 3,
            "_comment": "LLM connection and behavior settings"
        },
        "temperature": {
            "top": 1.0,
            "bottom": 0.2,
            "gradient": "linear",
            "_comment": "Temperature gradient along helix depth"
        },
        "memory": {
            "knowledge_db": "felix_knowledge.db",
            "task_db": "felix_memory.db",
            "compression_ratio": 0.3,
            "max_entries": 10000,
            "_comment": "Memory and persistence configuration"
        },
        "performance": {
            "batch_size": 10,
            "parallel_agents": 5,
            "cache_size": 100,
            "log_level": "INFO",
            "_comment": "Performance tuning parameters"
        }
    }

    return yaml.dump(template, default_flow_style=False, sort_keys=False)


def display_config_editor(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Display an interactive configuration editor (read-only simulation).

    Args:
        config: Current configuration

    Returns:
        Modified configuration (for preview only)
    """
    st.info("ℹ️ This is a preview editor. Changes are not saved to the actual configuration.")

    edited_config = config.copy()

    # Helix parameters
    with st.expander("🌀 Helix Geometry"):
        col1, col2 = st.columns(2)

        with col1:
            edited_config.setdefault('helix', {})
            edited_config['helix']['top_radius'] = st.number_input(
                "Top Radius",
                value=float(config.get('helix', {}).get('top_radius', 3.0)),
                min_value=0.5,
                max_value=10.0,
                step=0.1
            )
            edited_config['helix']['height'] = st.number_input(
                "Height",
                value=float(config.get('helix', {}).get('height', 8.0)),
                min_value=1.0,
                max_value=20.0,
                step=0.5
            )

        with col2:
            edited_config['helix']['bottom_radius'] = st.number_input(
                "Bottom Radius",
                value=float(config.get('helix', {}).get('bottom_radius', 0.5)),
                min_value=0.1,
                max_value=5.0,
                step=0.1
            )
            edited_config['helix']['turns'] = st.number_input(
                "Turns",
                value=float(config.get('helix', {}).get('turns', 2.0)),
                min_value=0.5,
                max_value=5.0,
                step=0.5
            )

    # Agent parameters
    with st.expander("🤖 Agent Settings"):
        edited_config['max_agents'] = st.slider(
            "Maximum Agents",
            min_value=1,
            max_value=100,
            value=config.get('max_agents', 25)
        )
        edited_config['base_token_budget'] = st.slider(
            "Token Budget",
            min_value=100,
            max_value=10000,
            value=config.get('base_token_budget', 2500),
            step=100
        )

    return edited_config