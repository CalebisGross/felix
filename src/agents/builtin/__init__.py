"""
Built-in agent plugins for the Felix Framework.

This package contains the standard agent types that ship with Felix:
- ResearchAgentPlugin: Information gathering and exploration
- AnalysisAgentPlugin: Processing and organizing information
- CriticAgentPlugin: Quality assurance and review

These plugins implement the SpecializedAgentPlugin interface and are
automatically discovered by the AgentPluginRegistry on startup.
"""

__all__ = [
    'ResearchAgentPlugin',
    'AnalysisAgentPlugin',
    'CriticAgentPlugin'
]
