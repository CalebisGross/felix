"""
Linear pipeline architecture implementation for Felix Framework comparison.

This module implements a traditional sequential processing pipeline to serve
as a baseline comparison against the helix-based architecture. The linear
pipeline provides controlled workload distribution characteristics for
statistical validation of research hypotheses.

Mathematical Foundation:
- Sequential stage processing: agents progress through stages 0 → 1 → ... → N-1
- Linear workload distribution: W_i = Total_Work / N + ε_i 
- Stage-based communication: O(1) connections per agent to adjacent stages
- Uniform capacity constraints: each stage has fixed capacity limits

Key Features:
- Fixed sequential progression (no branching or looping)
- Stage-based capacity management and load balancing
- Performance metrics collection for comparison analysis
- Workload distribution statistics (coefficient of variation)
- Completion time tracking for throughput analysis

This implementation supports Hypothesis H1 validation by providing measurable
workload distribution characteristics that can be statistically compared
against the helix architecture.

Mathematical references:
- docs/hypothesis_mathematics.md, Section H1.2: Linear workload distribution theory
- docs/mathematical_model.md: Comparison baseline methodology
"""

import time
import statistics
from typing import List, Dict, Any, Optional
from enum import Enum


class PipelineStage:
    """
    Individual stage in the linear pipeline with capacity management.
    
    Each stage processes agents sequentially and tracks workload metrics
    for comparison analysis against the helix architecture.
    """
    
    def __init__(self, stage_id: int, capacity: int):
        """
        Initialize pipeline stage.
        
        Args:
            stage_id: Unique identifier for this stage (0 to N-1)
            capacity: Maximum number of agents this stage can process concurrently
            
        Raises:
            ValueError: If parameters are invalid
        """
        if capacity <= 0:
            raise ValueError("Stage capacity must be positive")
            
        self.stage_id = stage_id
        self.capacity = capacity
        self.agents: List['PipelineAgent'] = []
        self.total_processed = 0
        self.total_processing_time = 0.0
    
    @property
    def current_load(self) -> int:
        """Get current number of agents in this stage."""
        return len(self.agents)
    
    def is_full(self) -> bool:
        """Check if stage is at full capacity."""
        return self.current_load >= self.capacity
    
    def add_agent(self, agent: 'PipelineAgent') -> None:
        """
        Add agent to this stage.
        
        Args:
            agent: Agent to add to the stage
            
        Raises:
            ValueError: If stage is at full capacity
        """
        if self.is_full():
            raise ValueError("Stage is at full capacity")
        
        self.agents.append(agent)
        agent.current_stage = self.stage_id
    
    def remove_agent(self, agent: 'PipelineAgent') -> None:
        """
        Remove agent from this stage.
        
        Args:
            agent: Agent to remove
        """
        if agent in self.agents:
            self.agents.remove(agent)
            self.total_processed += 1
            self.total_processing_time += agent.processing_time
    
    def get_workload(self) -> float:
        """
        Calculate total workload (sum of processing times) in this stage.
        
        Returns:
            Total workload as sum of agent processing times
        """
        return sum(agent.processing_time for agent in self.agents)
    
    def get_utilization(self) -> float:
        """
        Calculate stage utilization as percentage of capacity used.
        
        Returns:
            Utilization ratio between 0.0 and 1.0
        """
        return self.current_load / self.capacity if self.capacity > 0 else 0.0
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"PipelineStage(id={self.stage_id}, capacity={self.capacity}, "
                f"load={self.current_load}, workload={self.get_workload():.2f})")


class PipelineAgent:
    """
    Agent implementation for linear pipeline processing.
    
    Pipeline agents progress sequentially through fixed stages, providing
    a controlled baseline for comparison against helix-based agent behavior.
    """
    
    def __init__(self, agent_id: str, spawn_time: float):
        """
        Initialize pipeline agent.
        
        Args:
            agent_id: Unique identifier for the agent
            spawn_time: Time when agent becomes active (0.0 to 1.0)
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_initialization(agent_id, spawn_time)
        
        self.agent_id = agent_id
        self.spawn_time = spawn_time
        self.current_stage = 0
        self.state = "waiting"  # waiting, active, completed
        self.current_task: Optional[Any] = None
        self.processing_time: float = 0.0
        self.spawn_timestamp: Optional[float] = None
        self.completion_timestamp: Optional[float] = None
    
    def _validate_initialization(self, agent_id: str, spawn_time: float) -> None:
        """Validate agent initialization parameters."""
        if not agent_id or agent_id.strip() == "":
            raise ValueError("agent_id cannot be empty")
        
        if not (0.0 <= spawn_time <= 1.0):
            raise ValueError("spawn_time must be between 0 and 1")
    
    def can_spawn(self, current_time: float) -> bool:
        """
        Check if agent can spawn at the current time.
        
        Args:
            current_time: Current simulation time (0.0 to 1.0)
            
        Returns:
            True if agent can spawn, False otherwise
        """
        return current_time >= self.spawn_time
    
    def spawn(self, current_time: float, task: Any, stage: int = 0) -> None:
        """
        Spawn the agent and begin processing.
        
        Args:
            current_time: Current simulation time
            task: Task to assign to the agent
            stage: Initial stage (default 0)
            
        Raises:
            ValueError: If spawn conditions are not met
        """
        if not self.can_spawn(current_time):
            raise ValueError("Cannot spawn agent before spawn_time")
        
        if self.state != "waiting":
            raise ValueError("Agent already spawned")
        
        self.state = "active"
        self.current_stage = stage
        self.current_task = task
        self.processing_time = 0.0
        self.spawn_timestamp = current_time
    
    def advance_stage(self, max_stages: Optional[int] = None) -> None:
        """
        Advance agent to next pipeline stage or complete processing.
        
        This implements the sequential progression characteristic of
        linear pipeline architecture.
        
        Args:
            max_stages: Maximum number of stages (for completion detection)
        """
        if self.state != "active":
            return
        
        self.current_stage += 1
        
        # If max_stages provided, check for completion
        if max_stages is not None and self.current_stage >= max_stages:
            self.state = "completed"
    
    def complete(self, current_time: float) -> None:
        """
        Mark agent as completed.
        
        Args:
            current_time: Time when agent completed processing
        """
        self.state = "completed"
        self.completion_timestamp = current_time
    
    def get_task_id(self) -> Optional[str]:
        """Get ID of current task, if any."""
        if self.current_task and hasattr(self.current_task, 'id'):
            return self.current_task.id
        return None
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return (f"PipelineAgent(id={self.agent_id}, spawn_time={self.spawn_time}, "
                f"stage={self.current_stage}, state={self.state})")
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return str(self)


class LinearPipeline:
    """
    Linear sequential processing pipeline for architecture comparison.
    
    Implements traditional stage-based processing where agents progress
    sequentially through fixed stages. Provides baseline metrics for
    statistical comparison against helix architecture.
    
    Mathematical Model:
    - Sequential processing: agents flow through stages 0 → 1 → ... → N-1
    - Workload distribution: approximately uniform across stages
    - Communication topology: adjacent stage connections only
    - Capacity constraints: each stage has fixed processing capacity
    
    This implementation supports Hypothesis H1 validation by providing
    measurable workload distribution characteristics.
    """
    
    def __init__(self, num_stages: int, stage_capacity: int):
        """
        Initialize linear pipeline.
        
        Args:
            num_stages: Number of sequential processing stages
            stage_capacity: Maximum agents per stage
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_parameters(num_stages, stage_capacity)
        
        self.num_stages = num_stages
        self.stage_capacity = stage_capacity
        self.stages = [PipelineStage(i, stage_capacity) for i in range(num_stages)]
        self.agents: List[PipelineAgent] = []
        self.current_time: float = 0.0
        self.completed_agents: List[PipelineAgent] = []
    
    def _validate_parameters(self, num_stages: int, stage_capacity: int) -> None:
        """Validate pipeline parameters."""
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        
        if stage_capacity <= 0:
            raise ValueError("stage_capacity must be positive")
    
    @property
    def total_agents(self) -> int:
        """Get total number of agents in the pipeline system."""
        return len(self.agents)
    
    @property
    def active_agents(self) -> List[PipelineAgent]:
        """Get list of currently active agents."""
        return [agent for agent in self.agents if agent.state == "active"]
    
    def add_agent(self, agent: PipelineAgent, current_time: float = 0.0) -> None:
        """
        Add agent to the pipeline system.
        
        Args:
            agent: Agent to add to the system
            current_time: Current simulation time
        """
        self.agents.append(agent)
        # Add agent to first stage if it has capacity
        if not self.stages[0].is_full():
            self.stages[0].add_agent(agent)
    
    def get_active_agents(self) -> List[PipelineAgent]:
        """Get list of currently active agents."""
        return [agent for agent in self.agents if agent.state == "active"]
    
    def update(self, current_time: float) -> None:
        """
        Update pipeline state and spawn agents based on current time.
        
        Args:
            current_time: Current simulation time (0.0 to 1.0)
        """
        self.current_time = current_time
        
        # Spawn agents whose spawn time has arrived
        for agent in self.agents:
            if agent.state == "waiting" and agent.can_spawn(current_time):
                # Create mock task for consistency with tests
                mock_task = type('Task', (), {'id': f'task_{agent.agent_id}'})()
                agent.spawn(current_time, mock_task)
                
                # Add spawned agent to stage 0 if capacity allows and not already assigned
                if agent.current_stage == -1 and not self.stages[0].is_full():
                    self.stages[0].add_agent(agent)
    
    def advance_agents(self) -> None:
        """
        Advance agents through pipeline stages.
        
        This implements the sequential progression characteristic of
        linear pipeline architecture, moving agents from stage to stage.
        """
        for agent in self.get_active_agents():
            if agent.current_stage < self.num_stages - 1:
                # Remove from current stage if assigned
                current_stage = self.stages[agent.current_stage]
                if agent in current_stage.agents:
                    current_stage.remove_agent(agent)
                
                # Advance to next stage
                agent.advance_stage(max_stages=self.num_stages)
                
                # Add to new stage if not at capacity
                new_stage = self.stages[agent.current_stage]
                if not new_stage.is_full():
                    new_stage.add_agent(agent)
            else:
                # Agent has completed all stages
                agent.complete(self.current_time)
                self.completed_agents.append(agent)
                
                # Remove from final stage
                final_stage = self.stages[-1]
                if agent in final_stage.agents:
                    final_stage.remove_agent(agent)
    
    def distribute_agents_across_stages(self) -> None:
        """
        Distribute active agents across pipeline stages.
        
        This is used for testing and metrics collection to simulate
        steady-state operation where agents are distributed across stages.
        """
        active_agents = self.get_active_agents()
        
        for i, agent in enumerate(active_agents):
            target_stage = i % self.num_stages
            
            # Remove from current stage
            if agent.current_stage < len(self.stages):
                current_stage = self.stages[agent.current_stage]
                if agent in current_stage.agents:
                    current_stage.remove_agent(agent)
            
            # Add to target stage if capacity allows
            target_stage_obj = self.stages[target_stage]
            if not target_stage_obj.is_full():
                agent.current_stage = target_stage
                target_stage_obj.add_agent(agent)
    
    def calculate_workload_cv(self) -> float:
        """
        Calculate coefficient of variation for workload distribution.
        
        This is a key metric for Hypothesis H1 validation, measuring
        how evenly workload is distributed across pipeline stages.
        
        Returns:
            Coefficient of variation (std_dev / mean) of stage workloads
        """
        workloads = [stage.get_workload() for stage in self.stages]
        
        if not workloads or all(w == 0 for w in workloads):
            return 0.0
        
        mean_workload = statistics.mean(workloads)
        if mean_workload == 0:
            return 0.0
        
        std_workload = statistics.stdev(workloads) if len(workloads) > 1 else 0.0
        return std_workload / mean_workload
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive performance metrics.
        
        Returns:
            Dictionary containing performance metrics for analysis
        """
        active_agents = self.get_active_agents()
        
        return {
            "total_agents": self.total_agents,
            "active_agents": len(active_agents),
            "completed_agents": len(self.completed_agents),
            "stage_utilizations": [stage.get_utilization() for stage in self.stages],
            "total_workload": sum(stage.get_workload() for stage in self.stages),
            "workload_distribution": [stage.get_workload() for stage in self.stages],
            "current_time": self.current_time
        }
    
    def get_comparison_metrics(self) -> Dict[str, Any]:
        """
        Get metrics specifically needed for architecture comparison.
        
        Returns:
            Dictionary with metrics for helix vs linear comparison
        """
        workloads = [stage.get_workload() for stage in self.stages]
        utilizations = [stage.get_utilization() for stage in self.stages]
        
        # Calculate completion time (for completed agents)
        completion_times = [
            agent.completion_timestamp - agent.spawn_timestamp
            for agent in self.completed_agents
            if agent.completion_timestamp and agent.spawn_timestamp
        ]
        avg_completion_time = statistics.mean(completion_times) if completion_times else 0.0
        
        return {
            "workload_cv": self.calculate_workload_cv(),
            "completion_time": avg_completion_time,
            "utilization_variance": statistics.variance(utilizations) if len(utilizations) > 1 else 0.0,
            "agent_distribution": [stage.current_load for stage in self.stages],
            "total_throughput": len(self.completed_agents),
            "average_utilization": statistics.mean(utilizations) if utilizations else 0.0
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"LinearPipeline(stages={self.num_stages}, capacity={self.stage_capacity}, "
                f"agents={self.total_agents}, active={len(self.get_active_agents())})")