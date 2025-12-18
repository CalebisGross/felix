"""
Agent management endpoints.

Provides REST API for managing agents and agent plugins.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse

from src.api.dependencies import get_authenticated_felix, verify_api_key
from src.api.models import (
    AgentCreateRequest,
    AgentResponse,
    AgentListResponse,
    AgentPluginMetadata,
    AgentPluginListResponse
)
from src.core.felix_system import FelixSystem
from src.agents.agent_plugin_registry import get_global_registry

logger = logging.getLogger(__name__)

# Router
router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Agents"],
    responses={404: {"description": "Agent not found"}}
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_agent_info(agent) -> Dict[str, Any]:
    """Extract agent information as dictionary."""
    from datetime import datetime

    return {
        "agent_id": agent.agent_id,
        "agent_type": agent.agent_type,
        "spawn_time": agent.spawn_time,
        "status": "active",  # All registered agents are active
        "created_at": datetime.now(),  # TODO: Track actual creation time
        "current_task": None,  # TODO: Track current task
        "confidence": getattr(agent, 'confidence', None),
        "messages_processed": getattr(agent, 'messages_processed', 0),
        "tokens_used": getattr(agent, 'tokens_used', 0)
    }


# ============================================================================
# Agent Management Endpoints
# ============================================================================

@router.get("", response_model=AgentListResponse)
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> AgentListResponse:
    """
    List all active agents.

    Returns a list of currently active agents in the Felix system,
    including their type, status, and performance metrics.

    Args:
        agent_type: Optional filter by agent type (research, analysis, critic, etc.)
        felix: Felix system instance (injected)

    Returns:
        AgentListResponse: List of active agents

    Example:
        ```
        GET /api/v1/agents?agent_type=research

        Response:
        {
          "agents": [
            {
              "agent_id": "research_001",
              "agent_type": "research",
              "spawn_time": 0.1,
              "status": "active",
              "messages_processed": 5,
              "tokens_used": 1200
            }
          ],
          "total": 1
        }
        ```
    """
    try:
        # Get all agents from agent manager
        if not hasattr(felix, 'agent_manager') or felix.agent_manager is None:
            # No agents yet
            return AgentListResponse(agents=[], total=0)

        all_agents = felix.agent_manager.get_all_agents()

        # Filter by type if specified
        if agent_type:
            all_agents = [a for a in all_agents if a.agent_type == agent_type]

        # Convert to response models
        agent_responses = []
        for agent in all_agents:
            agent_info = get_agent_info(agent)
            agent_responses.append(AgentResponse(**agent_info))

        return AgentListResponse(
            agents=agent_responses,
            total=len(agent_responses)
        )

    except Exception as e:
        logger.exception("Error listing agents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing agents: {str(e)}"
        )


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: AgentCreateRequest,
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> AgentResponse:
    """
    Spawn a new agent.

    Creates and registers a new agent of the specified type.
    The agent will be available for task execution immediately.

    Args:
        request: Agent creation request with type and parameters
        felix: Felix system instance (injected)

    Returns:
        AgentResponse: Information about the created agent

    Raises:
        HTTPException: If agent creation fails or agent type is invalid

    Example:
        ```json
        POST /api/v1/agents
        {
          "agent_type": "research",
          "parameters": {
            "research_domain": "technical"
          }
        }

        Response (201 Created):
        {
          "agent_id": "dynamic_research_042",
          "agent_type": "research",
          "spawn_time": 0.15,
          "status": "active",
          "created_at": "2025-10-30T10:00:00Z"
        }
        ```
    """
    try:
        # Get agent factory
        if not hasattr(felix, 'agent_factory') or felix.agent_factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent factory not initialized"
            )

        agent_factory = felix.agent_factory

        # Check if agent type exists
        available_types = agent_factory.list_available_agent_types()
        if request.agent_type not in available_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown agent type: {request.agent_type}. "
                       f"Available types: {', '.join(available_types)}"
            )

        # Create agent
        agent = agent_factory.create_agent_by_type(
            agent_type=request.agent_type,
            spawn_time_range=None,  # Auto-calculate
            complexity="medium",  # Default complexity
            **request.parameters
        )

        # Register with agent manager if available
        if hasattr(felix, 'agent_manager') and felix.agent_manager:
            felix.agent_manager.register_agent(agent)

        # Convert to response
        agent_info = get_agent_info(agent)
        return AgentResponse(**agent_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating agent")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating agent: {str(e)}"
        )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> AgentResponse:
    """
    Get agent details.

    Retrieves detailed information about a specific agent including
    its status, performance metrics, and current task.

    Args:
        agent_id: Unique agent identifier
        felix: Felix system instance (injected)

    Returns:
        AgentResponse: Agent information

    Raises:
        HTTPException: If agent not found

    Example:
        ```
        GET /api/v1/agents/research_001

        Response:
        {
          "agent_id": "research_001",
          "agent_type": "research",
          "spawn_time": 0.1,
          "status": "active",
          "confidence": 0.85,
          "messages_processed": 12,
          "tokens_used": 2400
        }
        ```
    """
    try:
        # Get agent from agent manager
        if not hasattr(felix, 'agent_manager') or felix.agent_manager is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )

        agent = felix.agent_manager.get_agent(agent_id)

        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )

        # Convert to response
        agent_info = get_agent_info(agent)
        return AgentResponse(**agent_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting agent {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting agent: {str(e)}"
        )


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def terminate_agent(
    agent_id: str,
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> None:
    """
    Terminate an agent.

    Stops and unregisters the specified agent. The agent will no longer
    process tasks or receive messages.

    Args:
        agent_id: Unique agent identifier
        felix: Felix system instance (injected)

    Returns:
        None (204 No Content)

    Raises:
        HTTPException: If agent not found

    Example:
        ```
        DELETE /api/v1/agents/research_001

        Response: 204 No Content
        ```
    """
    try:
        # Get agent manager
        if not hasattr(felix, 'agent_manager') or felix.agent_manager is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )

        # Check if agent exists
        agent = felix.agent_manager.get_agent(agent_id)
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )

        # Deregister agent
        felix.agent_manager.deregister_agent(agent_id)

        logger.info(f"Agent {agent_id} terminated")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error terminating agent {agent_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error terminating agent: {str(e)}"
        )


# ============================================================================
# Agent Plugin Endpoints
# ============================================================================

@router.get("/plugins", response_model=AgentPluginListResponse, tags=["Agent Plugins"])
async def list_plugins(
    api_key: str = Depends(verify_api_key)
) -> AgentPluginListResponse:
    """
    List all available agent plugins.

    Returns metadata for all registered agent plugins, including
    both built-in and custom external plugins.

    Args:
        api_key: API key for authentication

    Returns:
        AgentPluginListResponse: List of agent plugins with metadata

    Example:
        ```
        GET /api/v1/agents/plugins

        Response:
        {
          "plugins": [
            {
              "agent_type": "research",
              "display_name": "Research Agent",
              "description": "Broad information gathering",
              "spawn_range": [0.0, 0.3],
              "capabilities": ["web_search", "information_gathering"],
              "tags": ["exploration"],
              "default_tokens": 800,
              "version": "1.0.0",
              "priority": 10
            }
          ],
          "total": 3,
          "builtin_count": 3,
          "external_count": 0
        }
        ```
    """
    try:
        # Get global plugin registry
        registry = get_global_registry()

        # Get all plugin metadata
        all_metadata = registry.get_all_metadata()

        # Convert to response models
        plugins = []
        for agent_type, metadata in all_metadata.items():
            plugin_data = asdict(metadata)
            plugins.append(AgentPluginMetadata(**plugin_data))

        # Get statistics
        stats = registry.get_statistics()

        return AgentPluginListResponse(
            plugins=plugins,
            total=len(plugins),
            builtin_count=stats.get("builtin_count", 0),
            external_count=stats.get("external_count", 0)
        )

    except Exception as e:
        logger.exception("Error listing agent plugins")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing agent plugins: {str(e)}"
        )


@router.get("/plugins/{agent_type}", response_model=AgentPluginMetadata, tags=["Agent Plugins"])
async def get_plugin_metadata(
    agent_type: str,
    api_key: str = Depends(verify_api_key)
) -> AgentPluginMetadata:
    """
    Get metadata for a specific agent plugin.

    Returns detailed metadata for the specified agent type including
    capabilities, spawn ranges, and configuration options.

    Args:
        agent_type: Type of agent plugin (research, analysis, critic, etc.)
        api_key: API key for authentication

    Returns:
        AgentPluginMetadata: Plugin metadata

    Raises:
        HTTPException: If plugin not found

    Example:
        ```
        GET /api/v1/agents/plugins/research

        Response:
        {
          "agent_type": "research",
          "display_name": "Research Agent",
          "description": "Specialized in broad information gathering",
          "spawn_range": [0.0, 0.3],
          "capabilities": ["web_search", "information_gathering"],
          "tags": ["exploration", "research"],
          "default_tokens": 800,
          "version": "1.0.0",
          "author": "Felix Framework",
          "priority": 10
        }
        ```
    """
    try:
        # Get global plugin registry
        registry = get_global_registry()

        # Get plugin metadata
        metadata = registry.get_metadata(agent_type)

        if metadata is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent plugin not found: {agent_type}"
            )

        # Convert to response model
        plugin_data = asdict(metadata)
        return AgentPluginMetadata(**plugin_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting plugin metadata for {agent_type}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting plugin metadata: {str(e)}"
        )


@router.post("/plugins/reload", tags=["Agent Plugins"])
async def reload_plugins(
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> Dict[str, Any]:
    """
    Hot-reload external agent plugins.

    Reloads all plugins from external directories without restarting
    the Felix system. Built-in plugins are not reloaded.

    This enables updating custom agent plugins without downtime.

    Args:
        felix: Felix system instance (injected)

    Returns:
        Dictionary with reload statistics

    Example:
        ```
        POST /api/v1/agents/plugins/reload

        Response:
        {
          "status": "success",
          "plugins_reloaded": 2,
          "total_plugins": 5,
          "builtin_plugins": 3,
          "external_plugins": 2
        }
        ```
    """
    try:
        # Get global plugin registry
        registry = get_global_registry()

        # Reload external plugins
        reloaded_count = registry.reload_external_plugins()

        # Get updated statistics
        stats = registry.get_statistics()

        return {
            "status": "success",
            "plugins_reloaded": reloaded_count,
            "total_plugins": stats.get("total_registered", 0),
            "builtin_plugins": stats.get("builtin_count", 0),
            "external_plugins": stats.get("external_count", 0)
        }

    except Exception as e:
        logger.exception("Error reloading plugins")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading plugins: {str(e)}"
        )


@router.get("/plugins/suitable", tags=["Agent Plugins"])
async def get_suitable_plugins(
    task: str = Query(..., description="Task description", min_length=1),
    complexity: str = Query("medium", description="Task complexity: simple, medium, complex"),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, List[str]]:
    """
    Get suitable agent types for a task.

    Analyzes the task description and returns a list of agent types
    that are suitable for handling the task, sorted by priority.

    Args:
        task: Task description
        complexity: Task complexity (simple, medium, complex)
        api_key: API key for authentication

    Returns:
        Dictionary with list of suitable agent types

    Example:
        ```
        GET /api/v1/agents/plugins/suitable?task=Review+code+for+bugs&complexity=medium

        Response:
        {
          "task": "Review code for bugs",
          "complexity": "medium",
          "suitable_agents": ["code_review", "critic", "analysis"]
        }
        ```
    """
    try:
        # Get global plugin registry
        registry = get_global_registry()

        # Get suitable agents
        suitable_agents = registry.get_agents_for_task(
            task_description=task,
            task_complexity=complexity
        )

        return {
            "task": task,
            "complexity": complexity,
            "suitable_agents": suitable_agents
        }

    except Exception as e:
        logger.exception("Error getting suitable plugins")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting suitable plugins: {str(e)}"
        )


# ============================================================================
# Agent Performance Metrics (Issue #56.10)
# ============================================================================

@router.get("/metrics", tags=["Agent Metrics"])
async def get_agent_metrics(
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> Dict[str, Any]:
    """
    Get agent performance metrics and statistics.

    Returns aggregated performance data for all agent types including
    success rates, average confidence, processing times, and phase transitions.

    This endpoint exposes data collected by AgentPerformanceTracker for
    adaptive feedback in agent spawning decisions (Issue #56.10).

    Returns:
        Dictionary containing:
        - agent_type_stats: Per-type statistics (success rate, avg confidence, etc.)
        - phase_transitions: Analysis of phase transition patterns
        - summary: High-level metrics summary

    Example:
        ```
        GET /api/v1/agents/metrics

        Response:
        {
          "agent_type_stats": {
            "research": {
              "total_runs": 42,
              "avg_confidence": 0.78,
              "success_rate": 0.95,
              "avg_processing_time": 2.3
            },
            "analysis": {...},
            "critic": {...}
          },
          "phase_transitions": {
            "exploration_to_analysis_avg": 0.35,
            "analysis_to_synthesis_avg": 0.72
          },
          "summary": {
            "total_agents_tracked": 126,
            "overall_success_rate": 0.89,
            "avg_confidence_all": 0.75
          }
        }
        ```
    """
    try:
        # Get performance tracker from Felix system
        if not felix.performance_tracker:
            return {
                "agent_type_stats": {},
                "phase_transitions": {},
                "summary": {
                    "total_agents_tracked": 0,
                    "message": "Performance tracking disabled (enable_memory=False)"
                }
            }

        # Fetch statistics from the tracker
        agent_type_stats = felix.performance_tracker.get_agent_type_statistics()
        phase_analysis = felix.performance_tracker.get_phase_transition_analysis()

        # Build summary
        total_agents = sum(
            stats.get("total_runs", 0)
            for stats in agent_type_stats.values()
        ) if agent_type_stats else 0

        avg_confidence_all = 0.0
        if agent_type_stats:
            confidences = [
                stats.get("avg_confidence", 0)
                for stats in agent_type_stats.values()
                if stats.get("total_runs", 0) > 0
            ]
            if confidences:
                avg_confidence_all = sum(confidences) / len(confidences)

        return {
            "agent_type_stats": agent_type_stats,
            "phase_transitions": phase_analysis,
            "summary": {
                "total_agents_tracked": total_agents,
                "overall_success_rate": 0.0,  # Computed if needed
                "avg_confidence_all": round(avg_confidence_all, 3)
            }
        }

    except Exception as e:
        logger.exception("Error getting agent metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting agent metrics: {str(e)}"
        )
