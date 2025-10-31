"""
Task Memory API router.

Endpoints for querying task patterns, recording executions, and getting strategy recommendations.
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import verify_api_key, get_task_memory
from src.api.models import (
    # Request models
    TaskPatternQueryRequest,
    TaskExecutionQueryRequest,
    TaskExecutionRequest,
    StrategyRecommendationRequest,
    # Response models
    TaskPatternModel,
    TaskPatternListResponse,
    TaskExecutionModel,
    TaskExecutionListResponse,
    TaskExecutionResponse,
    StrategyRecommendationResponse,
    TaskMemorySummaryResponse,
    # Enums from API models
    TaskComplexity as APITaskComplexity,
    TaskOutcome as APITaskOutcome,
)
# Import actual memory enums for conversion
from src.memory.task_memory import TaskComplexity, TaskOutcome

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/memory/tasks",
    tags=["Task Memory"],
    responses={
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    }
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_enum_value(value):
    """
    Safely get string value from enum or string.

    If value is an enum, returns its .value attribute.
    If value is already a string, returns it as-is.
    """
    if isinstance(value, str):
        return value
    return getattr(value, 'value', str(value))


def ensure_enum(value, enum_class):
    """
    Ensure value is an enum of the specified class.

    If value is already the correct enum type, returns it as-is.
    If value is a string, converts it to the enum.
    """
    if isinstance(value, enum_class):
        return value
    if isinstance(value, str):
        return enum_class(value)
    # If it's some other enum, get its value and convert
    return enum_class(get_enum_value(value))

def map_task_pattern_to_model(pattern_data: dict) -> TaskPatternModel:
    """Convert task pattern database record to Pydantic model."""
    import json

    return TaskPatternModel(
        pattern_id=pattern_data.get("pattern_id", ""),
        task_type=pattern_data.get("task_type", ""),
        complexity=APITaskComplexity(pattern_data.get("complexity", "simple")),
        keywords=json.loads(pattern_data.get("keywords_json", "[]")),
        typical_duration=pattern_data.get("typical_duration"),
        success_rate=pattern_data.get("success_rate", 0.0),
        failure_modes=json.loads(pattern_data.get("failure_modes_json", "[]")),
        optimal_strategies=json.loads(pattern_data.get("optimal_strategies_json", "[]")),
        required_agents=json.loads(pattern_data.get("required_agents_json", "[]")),
        context_requirements=json.loads(pattern_data.get("context_requirements_json", "{}")),
        usage_count=pattern_data.get("usage_count", 0),
        created_at=datetime.fromtimestamp(pattern_data.get("created_at", 0.0)),
        updated_at=datetime.fromtimestamp(pattern_data.get("updated_at", 0.0))
    )


def map_task_execution_to_model(execution_data: dict) -> TaskExecutionModel:
    """Convert task execution database record to Pydantic model."""
    import json

    return TaskExecutionModel(
        execution_id=execution_data.get("execution_id", ""),
        task_description=execution_data.get("task_description", ""),
        task_type=execution_data.get("task_type", ""),
        complexity=APITaskComplexity(execution_data.get("complexity", "simple")),
        outcome=APITaskOutcome(execution_data.get("outcome", "success")),
        duration=execution_data.get("duration", 0.0),
        agents_used=json.loads(execution_data.get("agents_used_json", "[]")),
        strategies_used=json.loads(execution_data.get("strategies_used_json", "[]")),
        context_size=execution_data.get("context_size", 0),
        error_messages=json.loads(execution_data.get("error_messages_json", "[]")),
        success_metrics=json.loads(execution_data.get("success_metrics_json", "{}")),
        patterns_matched=json.loads(execution_data.get("patterns_matched_json", "[]")),
        created_at=datetime.fromtimestamp(execution_data.get("created_at", 0.0))
    )


# ============================================================================
# Endpoint Implementations
# ============================================================================

@router.get("/patterns", response_model=TaskPatternListResponse)
async def list_patterns(
    request: TaskPatternQueryRequest = Depends(),
    api_key: str = Depends(verify_api_key),
    task_memory = Depends(get_task_memory)
):
    """
    List task patterns with optional filtering.

    Patterns are learned from historical task executions and include:
    - Success rates
    - Optimal strategies
    - Required agents
    - Typical durations
    """
    try:
        from src.memory.task_memory import TaskMemoryQuery

        # Build query - convert to strings for query
        query = TaskMemoryQuery(
            task_types=request.task_types,
            complexity_levels=[get_enum_value(level) for level in request.complexity_levels] if request.complexity_levels else None,
            keywords=request.keywords,
            min_success_rate=request.min_success_rate,
            max_duration=request.max_duration,
            limit=request.limit
        )

        # Get patterns
        patterns = task_memory.get_patterns(query)

        # Convert to models
        pattern_models = [map_task_pattern_to_model(p.to_dict()) for p in patterns]

        return TaskPatternListResponse(
            patterns=pattern_models,
            total=len(pattern_models)
        )

    except Exception as e:
        logger.exception("Error listing patterns")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing patterns: {str(e)}"
        )


@router.get("/patterns/{pattern_id}", response_model=TaskPatternModel)
async def get_pattern(
    pattern_id: str,
    api_key: str = Depends(verify_api_key),
    task_memory = Depends(get_task_memory)
):
    """
    Get specific pattern by ID.
    """
    try:
        # Query for specific pattern
        from src.memory.task_memory import TaskMemoryQuery

        query = TaskMemoryQuery(limit=1)
        patterns = task_memory.get_patterns(query)

        # Find matching pattern
        for pattern in patterns:
            if pattern.pattern_id == pattern_id:
                return map_task_pattern_to_model(pattern.to_dict())

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pattern not found: {pattern_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting pattern")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting pattern: {str(e)}"
        )


@router.get("/executions", response_model=TaskExecutionListResponse)
async def list_executions(
    request: TaskExecutionQueryRequest = Depends(),
    api_key: str = Depends(verify_api_key),
    task_memory = Depends(get_task_memory)
):
    """
    List task execution records with optional filtering.
    """
    try:
        import sqlite3
        from src.memory.task_memory import TaskMemoryQuery

        # Build SQL query
        conditions = []
        params = []

        if request.task_types:
            placeholders = ",".join(["?" for _ in request.task_types])
            conditions.append(f"task_type IN ({placeholders})")
            params.extend(request.task_types)

        if request.complexity_levels:
            placeholders = ",".join(["?" for _ in request.complexity_levels])
            conditions.append(f"complexity IN ({placeholders})")
            params.extend([get_enum_value(c) for c in request.complexity_levels])

        if request.outcomes:
            placeholders = ",".join(["?" for _ in request.outcomes])
            conditions.append(f"outcome IN ({placeholders})")
            params.extend([get_enum_value(o) for o in request.outcomes])

        if request.from_date:
            conditions.append("created_at >= ?")
            params.append(request.from_date.timestamp())

        if request.to_date:
            conditions.append("created_at <= ?")
            params.append(request.to_date.timestamp())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM task_executions
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(request.limit)

        # Execute query
        with sqlite3.connect(task_memory.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        # Convert to models
        executions = [map_task_execution_to_model(dict(row)) for row in rows]

        return TaskExecutionListResponse(
            executions=executions,
            total=len(executions)
        )

    except Exception as e:
        logger.exception("Error listing executions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing executions: {str(e)}"
        )


@router.get("/executions/{execution_id}", response_model=TaskExecutionModel)
async def get_execution(
    execution_id: str,
    api_key: str = Depends(verify_api_key),
    task_memory = Depends(get_task_memory)
):
    """
    Get specific execution by ID.
    """
    try:
        import sqlite3

        query = "SELECT * FROM task_executions WHERE execution_id = ?"

        with sqlite3.connect(task_memory.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (execution_id,))
            row = cursor.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution not found: {execution_id}"
            )

        return map_task_execution_to_model(dict(row))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting execution: {str(e)}"
        )


@router.post("/executions", response_model=TaskExecutionResponse)
async def record_execution(
    request: TaskExecutionRequest,
    api_key: str = Depends(verify_api_key),
    task_memory = Depends(get_task_memory)
):
    """
    Record a new task execution.

    This creates an execution record and updates relevant patterns.
    Patterns are created when 2+ similar executions exist.
    """
    try:
        # Record execution - ensure enums are correct type
        execution_id = task_memory.record_task_execution(
            task_description=request.task_description,
            task_type=request.task_type,
            complexity=ensure_enum(request.complexity, TaskComplexity),
            outcome=ensure_enum(request.outcome, TaskOutcome),
            duration=request.duration,
            agents_used=request.agents_used,
            strategies_used=request.strategies_used,
            context_size=request.context_size,
            error_messages=request.error_messages,
            success_metrics=request.success_metrics
        )

        # Get patterns that were matched/updated
        from src.memory.task_memory import TaskMemoryQuery
        query = TaskMemoryQuery(
            task_types=[request.task_type],
            complexity_levels=[get_enum_value(request.complexity)],
            limit=10
        )
        patterns = task_memory.get_patterns(query)

        patterns_matched = []
        patterns_updated = []

        for pattern in patterns:
            # Check if this execution matches the pattern
            pattern_keywords = set(pattern.keywords)
            task_keywords = set(task_memory._extract_keywords(request.task_description))
            overlap = len(pattern_keywords & task_keywords) / max(len(pattern_keywords), 1)

            if overlap >= 0.5:  # 50% threshold
                patterns_matched.append(pattern.pattern_id)
                patterns_updated.append(pattern.pattern_id)

        message = f"Execution recorded successfully. "
        if patterns_matched:
            message += f"Matched {len(patterns_matched)} existing patterns."
        else:
            message += "No existing patterns matched. New pattern will be created after 2+ similar executions."

        return TaskExecutionResponse(
            execution_id=execution_id,
            patterns_matched=patterns_matched,
            patterns_updated=patterns_updated,
            message=message
        )

    except Exception as e:
        logger.exception("Error recording execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording execution: {str(e)}"
        )


@router.post("/recommend-strategy", response_model=StrategyRecommendationResponse)
async def recommend_strategy(
    request: StrategyRecommendationRequest,
    api_key: str = Depends(verify_api_key),
    task_memory = Depends(get_task_memory)
):
    """
    Get strategy recommendation for a task.

    Uses historical patterns to recommend:
    - Optimal strategies
    - Required agents
    - Estimated duration
    - Success probability
    """
    try:
        # Get recommendation - pass complexity as string
        recommendation = task_memory.recommend_strategy(
            task_description=request.task_description,
            task_type=request.task_type,
            complexity=get_enum_value(request.complexity)
        )

        # Get similar patterns for context
        from src.memory.task_memory import TaskMemoryQuery
        query = TaskMemoryQuery(
            task_types=[request.task_type],
            complexity_levels=[get_enum_value(request.complexity)],
            limit=5
        )
        patterns = task_memory.get_patterns(query)
        similar_patterns = [map_task_pattern_to_model(p.to_dict()) for p in patterns]

        # Calculate confidence based on number of similar patterns
        confidence = min(0.95, 0.5 + (len(similar_patterns) * 0.1))

        return StrategyRecommendationResponse(
            recommended_strategies=recommendation.get("strategies", []),
            recommended_agents=recommendation.get("agents", []),
            estimated_duration=recommendation.get("estimated_duration"),
            success_probability=recommendation.get("success_probability", 0.5),
            similar_patterns=similar_patterns,
            confidence=confidence
        )

    except Exception as e:
        logger.exception("Error recommending strategy")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recommending strategy: {str(e)}"
        )


@router.get("/summary", response_model=TaskMemorySummaryResponse)
async def get_summary(
    api_key: str = Depends(verify_api_key),
    task_memory = Depends(get_task_memory)
):
    """
    Get task memory statistics and summary.
    """
    try:
        summary = task_memory.get_memory_summary()

        # Pass distributions as-is - Pydantic will handle enum validation
        return TaskMemorySummaryResponse(
            total_patterns=summary.get("total_patterns", 0),
            total_executions=summary.get("total_executions", 0),
            average_success_rate=summary.get("average_success_rate", 0.0),
            most_common_task_types=summary.get("most_common_task_types", {}),
            complexity_distribution=summary.get("complexity_distribution", {}),
            outcome_distribution=summary.get("outcome_distribution", {})
        )

    except Exception as e:
        logger.exception("Error getting summary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting summary: {str(e)}"
        )
