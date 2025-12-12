"""
Agent system for the Felix Framework.

This module provides the agent architecture including:
- FelixAgent: The core identity layer (unified persona)
- Base Agent class with helix progression
- LLMAgent with position-aware prompting and LLM integration
- Specialized agents (Research, Analysis, Critic)
- Dynamic spawning with confidence monitoring and content analysis
- Prompt optimization for improved LLM responses

The agent system forms the core of Felix's adaptive multi-agent architecture,
enabling intelligent task decomposition and collaborative problem-solving.

Note: Synthesis is performed by CentralPost, not by a specialized agent.
"""

from .agent import Agent, AgentState
from .llm_agent import LLMAgent, LLMTask, LLMResult
from .felix_agent import FelixAgent, FelixResponse
from .specialized_agents import (
    ResearchAgent,
    AnalysisAgent,
    CriticAgent
)
from .dynamic_spawning import (
    DynamicSpawning,
    ConfidenceMonitor,
    ContentAnalyzer,
    TeamSizeOptimizer
)
from .prompt_optimization import PromptOptimizer, PromptMetrics

__all__ = [
    # Felix identity layer
    'FelixAgent',
    'FelixResponse',

    # Base agent
    'Agent',
    'AgentState',

    # LLM agent
    'LLMAgent',
    'LLMTask',
    'LLMResult',

    # Specialized agents (Note: Synthesis done by CentralPost)
    'ResearchAgent',
    'AnalysisAgent',
    'CriticAgent',

    # Dynamic spawning
    'DynamicSpawning',
    'ConfidenceMonitor',
    'ContentAnalyzer',
    'TeamSizeOptimizer',

    # Prompt optimization
    'PromptOptimizer',
    'PromptMetrics'
]
