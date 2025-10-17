"""
Task Memory System for the Felix Framework.

Provides pattern recognition, success/failure tracking, and adaptive strategy
selection based on historical task execution data.
"""

import json
import sqlite3
import hashlib
import time
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

class TaskOutcome(Enum):
    """Possible outcomes for task execution."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"

class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

@dataclass
class TaskPattern:
    """Pattern extracted from task execution history."""
    pattern_id: str
    task_type: str
    complexity: TaskComplexity
    keywords: List[str]
    typical_duration: float
    success_rate: float
    failure_modes: List[str]
    optimal_strategies: List[str]
    required_agents: List[str]
    context_requirements: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['complexity'] = self.complexity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskPattern':
        """Create from dictionary."""
        data['complexity'] = TaskComplexity(data['complexity'])
        return cls(**data)

@dataclass
class TaskExecution:
    """Record of a task execution."""
    execution_id: str
    task_description: str
    task_type: str
    complexity: TaskComplexity
    outcome: TaskOutcome
    duration: float
    agents_used: List[str]
    strategies_used: List[str]
    context_size: int
    error_messages: List[str]
    success_metrics: Dict[str, float]
    patterns_matched: List[str]
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['complexity'] = self.complexity.value
        data['outcome'] = self.outcome.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskExecution':
        """Create from dictionary."""
        data['complexity'] = TaskComplexity(data['complexity'])
        data['outcome'] = TaskOutcome(data['outcome'])
        return cls(**data)

@dataclass
class TaskMemoryQuery:
    """Query structure for task memory retrieval."""
    task_types: Optional[List[str]] = None
    complexity_levels: Optional[List[TaskComplexity]] = None
    outcomes: Optional[List[TaskOutcome]] = None
    keywords: Optional[List[str]] = None
    min_success_rate: Optional[float] = None
    max_duration: Optional[float] = None
    time_range: Optional[Tuple[float, float]] = None
    limit: int = 10

class TaskMemory:
    """
    Task memory system for pattern recognition and adaptive strategy selection.
    
    Tracks task execution history, identifies patterns, and recommends
    optimal strategies based on past performance.
    """
    
    def __init__(self, storage_path: str = "felix_task_memory.db"):
        """
        Initialize task memory system.
        
        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = Path(storage_path)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.storage_path) as conn:
            # Task patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    complexity TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    typical_duration REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    failure_modes_json TEXT NOT NULL,
                    optimal_strategies_json TEXT NOT NULL,
                    required_agents_json TEXT NOT NULL,
                    context_requirements_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            
            # Task executions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_executions (
                    execution_id TEXT PRIMARY KEY,
                    task_description TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    complexity TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    duration REAL NOT NULL,
                    agents_used_json TEXT NOT NULL,
                    strategies_used_json TEXT NOT NULL,
                    context_size INTEGER NOT NULL,
                    error_messages_json TEXT NOT NULL,
                    success_metrics_json TEXT NOT NULL,
                    patterns_matched_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            
            # Create indices for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_type ON task_patterns(task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_complexity ON task_patterns(complexity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_success_rate ON task_patterns(success_rate)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_type ON task_executions(task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_outcome ON task_executions(outcome)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_execution_created ON task_executions(created_at)")
    
    def _generate_execution_id(self, task_description: str) -> str:
        """Generate unique ID for task execution."""
        hash_input = f"{task_description}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _generate_pattern_id(self, task_type: str, complexity: TaskComplexity, 
                           keywords: List[str]) -> str:
        """Generate unique ID for task pattern."""
        keywords_str = ":".join(sorted(keywords))
        hash_input = f"{task_type}:{complexity.value}:{keywords_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def record_task_execution(self, task_description: str, task_type: str,
                            complexity: TaskComplexity, outcome: TaskOutcome,
                            duration: float, agents_used: List[str],
                            strategies_used: List[str], context_size: int,
                            error_messages: Optional[List[str]] = None,
                            success_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Record a task execution for future pattern analysis.
        
        Args:
            task_description: Description of the task
            task_type: Type/category of the task
            complexity: Assessed complexity level
            outcome: Execution outcome
            duration: Execution duration in seconds
            agents_used: List of agent types used
            strategies_used: List of strategies employed
            context_size: Size of context used
            error_messages: List of error messages if any
            success_metrics: Success metrics if available
            
        Returns:
            Execution ID
        """
        if error_messages is None:
            error_messages = []
        if success_metrics is None:
            success_metrics = {}
        
        execution_id = self._generate_execution_id(task_description)
        
        execution = TaskExecution(
            execution_id=execution_id,
            task_description=task_description,
            task_type=task_type,
            complexity=complexity,
            outcome=outcome,
            duration=duration,
            agents_used=agents_used,
            strategies_used=strategies_used,
            context_size=context_size,
            error_messages=error_messages,
            success_metrics=success_metrics,
            patterns_matched=[]  # Will be filled by pattern matching
        )
        
        # Find matching patterns and update them
        matched_patterns = self._find_matching_patterns(execution)
        execution.patterns_matched = [p.pattern_id for p in matched_patterns]
        
        # Store execution
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT INTO task_executions 
                (execution_id, task_description, task_type, complexity, outcome,
                 duration, agents_used_json, strategies_used_json, context_size,
                 error_messages_json, success_metrics_json, patterns_matched_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                task_description,
                task_type,
                complexity.value,
                outcome.value,
                duration,
                json.dumps(agents_used),
                json.dumps(strategies_used),
                context_size,
                json.dumps(error_messages),
                json.dumps(success_metrics),
                json.dumps(execution.patterns_matched),
                execution.created_at
            ))
        
        # Update or create patterns based on this execution
        self._update_patterns_from_execution(execution)
        
        return execution_id
    
    def _find_matching_patterns(self, execution: TaskExecution) -> List[TaskPattern]:
        """Find patterns that match the given execution."""
        patterns = self.get_patterns(TaskMemoryQuery(
            task_types=[execution.task_type],
            complexity_levels=[execution.complexity]
        ))
        
        matched = []
        task_keywords = self._extract_keywords(execution.task_description)
        
        for pattern in patterns:
            # Check keyword overlap
            keyword_overlap = len(set(task_keywords) & set(pattern.keywords))
            if keyword_overlap >= len(pattern.keywords) * 0.5:  # 50% overlap threshold
                matched.append(pattern)
        
        return matched
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from task description."""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Filter out common words
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'man', 'she', 'use', 'way', 'who', 'oil', 'sit', 'set', 'run'
        }
        
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return list(set(keywords))  # Remove duplicates
    
    def _update_patterns_from_execution(self, execution: TaskExecution) -> None:
        """Update or create patterns based on task execution."""
        task_keywords = self._extract_keywords(execution.task_description)
        
        if not task_keywords:
            return
        
        pattern_id = self._generate_pattern_id(
            execution.task_type, execution.complexity, task_keywords
        )
        
        # Check if pattern exists
        existing_pattern = self._get_pattern_by_id(pattern_id)
        
        if existing_pattern:
            # Update existing pattern
            self._update_existing_pattern(existing_pattern, execution)
        else:
            # Create new pattern
            self._create_new_pattern(pattern_id, execution, task_keywords)
    
    def _get_pattern_by_id(self, pattern_id: str) -> Optional[TaskPattern]:
        """Get pattern by ID."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM task_patterns WHERE pattern_id = ?",
                (pattern_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_pattern(row)
            return None
    
    def _update_existing_pattern(self, pattern: TaskPattern, 
                               execution: TaskExecution) -> None:
        """Update existing pattern with new execution data."""
        # Get all executions for this pattern to recalculate metrics
        executions = self._get_executions_for_pattern(pattern.pattern_id)
        executions.append(execution)
        
        # Recalculate success rate
        successes = sum(1 for e in executions 
                       if e.outcome in [TaskOutcome.SUCCESS, TaskOutcome.PARTIAL_SUCCESS])
        pattern.success_rate = successes / len(executions)
        
        # Recalculate typical duration
        durations = [e.duration for e in executions]
        pattern.typical_duration = sum(durations) / len(durations)
        
        # Update failure modes
        failures = [e for e in executions if e.outcome in [TaskOutcome.FAILURE, TaskOutcome.ERROR]]
        failure_modes = []
        for f in failures:
            failure_modes.extend(f.error_messages)
        pattern.failure_modes = list(set(failure_modes))
        
        # Update optimal strategies (from successful executions)
        successes = [e for e in executions if e.outcome == TaskOutcome.SUCCESS]
        strategy_counts = {}
        for s in successes:
            for strategy in s.strategies_used:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Sort strategies by usage in successful executions
        pattern.optimal_strategies = sorted(strategy_counts.keys(), 
                                          key=lambda x: strategy_counts[x], 
                                          reverse=True)[:5]
        
        # Update required agents (from successful executions)
        agent_counts = {}
        for s in successes:
            for agent in s.agents_used:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        pattern.required_agents = sorted(agent_counts.keys(),
                                       key=lambda x: agent_counts[x],
                                       reverse=True)[:3]
        
        pattern.updated_at = time.time()
        pattern.usage_count += 1
        
        # Save updated pattern
        self._save_pattern(pattern)
    
    def _create_new_pattern(self, pattern_id: str, execution: TaskExecution,
                          keywords: List[str]) -> None:
        """Create new pattern from execution."""
        pattern = TaskPattern(
            pattern_id=pattern_id,
            task_type=execution.task_type,
            complexity=execution.complexity,
            keywords=keywords,
            typical_duration=execution.duration,
            success_rate=1.0 if execution.outcome in [TaskOutcome.SUCCESS, TaskOutcome.PARTIAL_SUCCESS] else 0.0,
            failure_modes=execution.error_messages if execution.outcome in [TaskOutcome.FAILURE, TaskOutcome.ERROR] else [],
            optimal_strategies=execution.strategies_used if execution.outcome == TaskOutcome.SUCCESS else [],
            required_agents=execution.agents_used if execution.outcome == TaskOutcome.SUCCESS else [],
            context_requirements={
                "min_context_size": execution.context_size,
                "success_metrics": execution.success_metrics
            },
            usage_count=1
        )
        
        self._save_pattern(pattern)
    
    def _save_pattern(self, pattern: TaskPattern) -> None:
        """Save pattern to database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO task_patterns 
                (pattern_id, task_type, complexity, keywords_json, typical_duration,
                 success_rate, failure_modes_json, optimal_strategies_json,
                 required_agents_json, context_requirements_json, created_at,
                 updated_at, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.task_type,
                pattern.complexity.value,
                json.dumps(pattern.keywords),
                pattern.typical_duration,
                pattern.success_rate,
                json.dumps(pattern.failure_modes),
                json.dumps(pattern.optimal_strategies),
                json.dumps(pattern.required_agents),
                json.dumps(pattern.context_requirements),
                pattern.created_at,
                pattern.updated_at,
                pattern.usage_count
            ))
    
    def _get_executions_for_pattern(self, pattern_id: str) -> List[TaskExecution]:
        """Get all executions that match a pattern."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM task_executions 
                WHERE patterns_matched_json LIKE ?
            """, (f'%"{pattern_id}"%',))
            
            return [self._row_to_execution(row) for row in cursor.fetchall()]
    
    def _row_to_pattern(self, row) -> TaskPattern:
        """Convert database row to TaskPattern."""
        (pattern_id, task_type, complexity, keywords_json, typical_duration,
         success_rate, failure_modes_json, optimal_strategies_json,
         required_agents_json, context_requirements_json, created_at,
         updated_at, usage_count) = row
        
        return TaskPattern(
            pattern_id=pattern_id,
            task_type=task_type,
            complexity=TaskComplexity(complexity),
            keywords=json.loads(keywords_json),
            typical_duration=typical_duration,
            success_rate=success_rate,
            failure_modes=json.loads(failure_modes_json),
            optimal_strategies=json.loads(optimal_strategies_json),
            required_agents=json.loads(required_agents_json),
            context_requirements=json.loads(context_requirements_json),
            created_at=created_at,
            updated_at=updated_at,
            usage_count=usage_count
        )
    
    def _row_to_execution(self, row) -> TaskExecution:
        """Convert database row to TaskExecution."""
        (execution_id, task_description, task_type, complexity, outcome,
         duration, agents_used_json, strategies_used_json, context_size,
         error_messages_json, success_metrics_json, patterns_matched_json, created_at) = row
        
        return TaskExecution(
            execution_id=execution_id,
            task_description=task_description,
            task_type=task_type,
            complexity=TaskComplexity(complexity),
            outcome=TaskOutcome(outcome),
            duration=duration,
            agents_used=json.loads(agents_used_json),
            strategies_used=json.loads(strategies_used_json),
            context_size=context_size,
            error_messages=json.loads(error_messages_json),
            success_metrics=json.loads(success_metrics_json),
            patterns_matched=json.loads(patterns_matched_json),
            created_at=created_at
        )
    
    def get_patterns(self, query: TaskMemoryQuery) -> List[TaskPattern]:
        """
        Retrieve task patterns matching query criteria.
        
        Args:
            query: Query parameters
            
        Returns:
            List of matching task patterns
        """
        sql_parts = ["SELECT * FROM task_patterns WHERE 1=1"]
        params = []
        
        if query.task_types:
            type_placeholders = ",".join("?" * len(query.task_types))
            sql_parts.append(f"AND task_type IN ({type_placeholders})")
            params.extend(query.task_types)
        
        if query.complexity_levels:
            complexity_placeholders = ",".join("?" * len(query.complexity_levels))
            sql_parts.append(f"AND complexity IN ({complexity_placeholders})")
            params.extend([c.value for c in query.complexity_levels])
        
        if query.min_success_rate:
            sql_parts.append("AND success_rate >= ?")
            params.append(query.min_success_rate)
        
        if query.max_duration:
            sql_parts.append("AND typical_duration <= ?")
            params.append(query.max_duration)
        
        if query.time_range:
            sql_parts.append("AND created_at BETWEEN ? AND ?")
            params.extend(query.time_range)
        
        # Order by success rate and usage count
        sql_parts.append("ORDER BY success_rate DESC, usage_count DESC")
        sql_parts.append("LIMIT ?")
        params.append(query.limit)
        
        sql = " ".join(sql_parts)
        
        patterns = []
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                pattern = self._row_to_pattern(row)
                
                # Apply keyword filtering if specified
                if query.keywords:
                    pattern_keywords_lower = [k.lower() for k in pattern.keywords]
                    if not any(keyword.lower() in pattern_keywords_lower 
                             for keyword in query.keywords):
                        continue
                
                patterns.append(pattern)
                
                # Update usage count
                self._increment_pattern_usage(pattern.pattern_id)
        
        return patterns
    
    def _increment_pattern_usage(self, pattern_id: str) -> None:
        """Increment usage count for pattern."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                UPDATE task_patterns 
                SET usage_count = usage_count + 1 
                WHERE pattern_id = ?
            """, (pattern_id,))
    
    def recommend_strategy(self, task_description: str, task_type: str,
                          complexity: TaskComplexity) -> Dict[str, Any]:
        """
        Recommend optimal strategy for a task based on historical patterns.
        
        Args:
            task_description: Description of the task
            task_type: Type/category of the task
            complexity: Assessed complexity level
            
        Returns:
            Dictionary with strategy recommendations
        """
        # Find similar patterns
        keywords = self._extract_keywords(task_description)
        
        query = TaskMemoryQuery(
            task_types=[task_type],
            complexity_levels=[complexity],
            keywords=keywords,
            min_success_rate=0.5,
            limit=5
        )
        
        patterns = self.get_patterns(query)
        
        if not patterns:
            return {
                "strategies": [],
                "agents": [],
                "estimated_duration": None,
                "success_probability": 0.0,
                "recommendations": "No similar patterns found. Proceeding with default strategy.",
                "potential_issues": []
            }
        
        # Aggregate recommendations from top patterns
        all_strategies = []
        all_agents = []
        durations = []
        success_rates = []
        potential_issues = []
        
        for pattern in patterns:
            all_strategies.extend(pattern.optimal_strategies)
            all_agents.extend(pattern.required_agents)
            durations.append(pattern.typical_duration)
            success_rates.append(pattern.success_rate)
            potential_issues.extend(pattern.failure_modes)
        
        # Get most common strategies and agents
        strategy_counts = {}
        for strategy in all_strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        agent_counts = {}
        for agent in all_agents:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        recommended_strategies = sorted(strategy_counts.keys(),
                                      key=lambda x: strategy_counts[x],
                                      reverse=True)[:3]
        
        recommended_agents = sorted(agent_counts.keys(),
                                  key=lambda x: agent_counts[x],
                                  reverse=True)[:3]
        
        # Calculate metrics
        avg_duration = sum(durations) / len(durations) if durations else None
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        # Generate recommendations text
        recommendations = []
        if recommended_strategies:
            recommendations.append(f"Use proven strategies: {', '.join(recommended_strategies[:2])}")
        if recommended_agents:
            recommendations.append(f"Deploy agents: {', '.join(recommended_agents[:2])}")
        if avg_duration:
            recommendations.append(f"Expected duration: {avg_duration:.1f} seconds")
        
        return {
            "strategies": recommended_strategies,
            "agents": recommended_agents,
            "estimated_duration": avg_duration,
            "success_probability": avg_success_rate,
            "recommendations": ". ".join(recommendations),
            "potential_issues": list(set(potential_issues))[:3],
            "patterns_used": len(patterns)
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics of task memory."""
        with sqlite3.connect(self.storage_path) as conn:
            # Total patterns and executions
            cursor = conn.execute("SELECT COUNT(*) FROM task_patterns")
            total_patterns = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM task_executions")
            total_executions = cursor.fetchone()[0]
            
            # Success rate distribution
            cursor = conn.execute("""
                SELECT outcome, COUNT(*) 
                FROM task_executions 
                GROUP BY outcome
            """)
            outcome_distribution = dict(cursor.fetchall())
            
            # Most common task types
            cursor = conn.execute("""
                SELECT task_type, COUNT(*) 
                FROM task_patterns 
                GROUP BY task_type 
                ORDER BY COUNT(*) DESC 
                LIMIT 5
            """)
            top_task_types = dict(cursor.fetchall())
            
            # Average success rate by complexity
            cursor = conn.execute("""
                SELECT complexity, AVG(success_rate) 
                FROM task_patterns 
                GROUP BY complexity
            """)
            success_by_complexity = dict(cursor.fetchall())
            
            return {
                "total_patterns": total_patterns,
                "total_executions": total_executions,
                "outcome_distribution": outcome_distribution,
                "top_task_types": top_task_types,
                "success_by_complexity": success_by_complexity,
                "storage_path": str(self.storage_path)
            }
    
    def cleanup_old_patterns(self, max_age_days: int = 60,
                           min_usage_count: int = 2) -> int:
        """
        Clean up old or unused task patterns.
        
        Args:
            max_age_days: Maximum age in days
            min_usage_count: Minimum usage count to keep
            
        Returns:
            Number of patterns deleted
        """
        max_age_seconds = max_age_days * 24 * 3600
        cutoff_time = time.time() - max_age_seconds
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                DELETE FROM task_patterns 
                WHERE (created_at < ? AND usage_count < ?)
                   OR (success_rate = 0.0 AND usage_count = 1)
            """, (cutoff_time, min_usage_count))
            
            return cursor.rowcount
