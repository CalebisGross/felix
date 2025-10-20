"""
Felix Workflows Module

This module provides workflow implementations that properly integrate with
the Felix framework architecture, using CentralPost, AgentFactory, and memory systems.

Components:
- felix_workflow: Core Felix-based workflow using helix agents and central communication
- context_builder: Collaborative context builder for enriched agent context
"""

from .felix_workflow import run_felix_workflow
from .context_builder import CollaborativeContextBuilder, EnrichedContext

__all__ = ['run_felix_workflow', 'CollaborativeContextBuilder', 'EnrichedContext']
