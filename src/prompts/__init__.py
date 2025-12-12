"""
Prompt management system for Felix Framework.

This module provides a hybrid YAML + Database system for managing agent prompts:
- Default prompts stored in config/prompts.yaml (version controlled)
- Custom overrides stored in SQLite database with versioning
- Runtime caching for performance
- Performance tracking for A/B testing

The prompt system supports:
- View and edit prompts via GUI
- Version history and rollback
- Performance metrics per prompt version
- Template variable substitution
"""

from src.prompts.prompt_manager import PromptManager
from src.prompts.prompt_pipeline import PromptPipeline, PromptBuildResult, PromptStageResult

__all__ = ['PromptManager', 'PromptPipeline', 'PromptBuildResult', 'PromptStageResult']
