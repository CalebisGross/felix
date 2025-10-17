"""
Token Budget Management for Felix Framework.

This module implements adaptive token allocation based on agent position
in the helix, enabling efficient resource usage without truncating content.

Key Features:
- Position-based token budgets (more at top, less at bottom)
- Stage-aware allocation to prevent overconsumption
- Content compression guidance for iterative refinement
- Adaptive prompting based on helix depth

Mathematical Foundation:
- Token allocation follows inverse relationship with depth
- Budget decreases as agents focus toward synthesis
- Maintains quality while improving efficiency
"""

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TokenAllocation:
    """Token allocation details for an agent at a specific position."""
    stage_budget: int           # Tokens for current processing stage
    remaining_budget: int       # Total remaining tokens for agent
    total_budget: int          # Original total budget for agent
    compression_ratio: float   # Suggested compression from previous stages
    style_guidance: str        # Recommended output style
    depth_ratio: float         # Agent's current depth in helix


class TokenBudgetManager:
    """
    Manages token budgets for agents based on helix position and processing stage.
    
    Implements adaptive allocation strategy where agents receive more tokens
    for exploration at the top of the helix and fewer tokens for focused
    synthesis at the bottom.
    """
    
    def __init__(self, base_budget: int = 1000, min_budget: int = 200, 
                 max_budget: int = 800, strict_mode: bool = False):
        """
        Initialize token budget manager.
        
        Args:
            base_budget: Base token allocation per agent
            min_budget: Minimum tokens for any single stage
            max_budget: Maximum tokens for any single stage
            strict_mode: Enable strict token budgets for lightweight models
        """
        self.base_budget = base_budget
        self.min_budget = min_budget  
        self.max_budget = max_budget
        self.strict_mode = strict_mode
        
        # Track agent budgets
        self._agent_budgets: Dict[str, int] = {}  # remaining budget per agent
        self._agent_total_used: Dict[str, int] = {}  # total used per agent
        self._agent_types: Dict[str, str] = {}  # agent type mapping
    
    def initialize_agent_budget(self, agent_id: str, agent_type: str, max_tokens_per_stage: int = None) -> int:
        """
        Initialize total budget for an agent based on type.
        
        Args:
            agent_id: Agent identifier
            agent_type: Type of agent (research, analysis, synthesis, etc.)
            max_tokens_per_stage: Maximum tokens per stage (if provided, adjusts max_budget)
            
        Returns:
            Total allocated budget for the agent
        """
        # Different agent types get different base budgets
        if self.strict_mode:
            # Strict budgets for lightweight models (much lower)
            type_budgets = {
                "research": 400,    # Research agents: 400 base budget
                "analysis": 350,    # Analysis agents: 350 base budget
                "synthesis": 300,   # Synthesis agents: 300 base budget
                "critic": 250       # Critic agents: 250 base budget
            }
            total_budget = type_budgets.get(agent_type, 350)
        else:
            # Original multiplier-based system
            type_multipliers = {
                "research": 1.2,      # More tokens for exploration
                "analysis": 1.0,      # Standard allocation  
                "synthesis": 0.8,     # Fewer tokens for focused synthesis
                "critic": 0.9         # Slightly fewer for critique
            }
            multiplier = type_multipliers.get(agent_type, 1.0)
            total_budget = int(self.base_budget * multiplier)
        
        # This is now handled above in the if/else block
        
        # Adjust max_budget to respect agent's max_tokens_per_stage and strict mode
        if self.strict_mode and agent_type in ["research", "analysis", "synthesis", "critic"]:
            # Strict per-stage limits for lightweight models
            strict_stage_limits = {
                "research": 150,    # Research agents: max 150 per stage
                "analysis": 120,    # Analysis agents: max 120 per stage
                "synthesis": 100,   # Synthesis agents: max 100 per stage
                "critic": 80        # Critic agents: max 80 per stage
            }
            stage_limit = strict_stage_limits.get(agent_type, 100)
            if max_tokens_per_stage is not None:
                stage_limit = min(stage_limit, max_tokens_per_stage)
            self.max_budget = min(self.max_budget, stage_limit)
        elif max_tokens_per_stage is not None:
            self.max_budget = min(self.max_budget, max_tokens_per_stage)
        
        self._agent_budgets[agent_id] = total_budget
        self._agent_total_used[agent_id] = 0
        self._agent_types[agent_id] = agent_type
        
        return total_budget
    
    def calculate_stage_allocation(self, agent_id: str, depth_ratio: float, 
                                 stage: int) -> TokenAllocation:
        """
        Calculate token allocation for a specific processing stage.
        
        Args:
            agent_id: Agent identifier
            depth_ratio: Current depth in helix (0.0 = top, 1.0 = bottom)
            stage: Current processing stage number (1-based)
            
        Returns:
            TokenAllocation with budget and guidance information
        """
        if agent_id not in self._agent_budgets:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        remaining_budget = self._agent_budgets[agent_id]
        total_used = self._agent_total_used[agent_id]
        original_budget = remaining_budget + total_used
        agent_type = self._agent_types[agent_id]
        
        # Calculate stage budget based on agent type, position and remaining allocation
        stage_budget = self._calculate_position_budget(
            agent_type, depth_ratio, remaining_budget, stage
        )
        
        # Ensure budget constraints
        stage_budget = max(self.min_budget, min(stage_budget, self.max_budget))
        stage_budget = min(stage_budget, remaining_budget)  # Don't exceed remaining
        
        # Calculate compression and style guidance based on agent type
        compression_ratio = self._calculate_compression_ratio(agent_type, depth_ratio, stage)
        style_guidance = self._generate_style_guidance(agent_type, depth_ratio, stage_budget)
        
        return TokenAllocation(
            stage_budget=stage_budget,
            remaining_budget=remaining_budget,
            total_budget=original_budget,
            compression_ratio=compression_ratio,
            style_guidance=style_guidance,
            depth_ratio=depth_ratio
        )
    
    def record_usage(self, agent_id: str, tokens_used: int) -> None:
        """
        Record token usage for an agent.
        
        Args:
            agent_id: Agent identifier
            tokens_used: Number of tokens consumed in last operation
        """
        if agent_id not in self._agent_budgets:
            raise ValueError(f"Agent {agent_id} not initialized")
        
        self._agent_budgets[agent_id] -= tokens_used
        self._agent_total_used[agent_id] += tokens_used
        
        # Ensure non-negative budget
        self._agent_budgets[agent_id] = max(0, self._agent_budgets[agent_id])
    
    def _calculate_position_budget(self, agent_type: str, depth_ratio: float, 
                                 remaining_budget: int, stage: int) -> int:
        """Calculate token budget based on agent type, helix position and stage."""
        # FIXED: Token allocation should INCREASE for synthesis agents, not decrease
        # Different agent types need different token allocations based on their role
        
        # Agent-type-specific base budgets (corrects the inverted allocation)
        if agent_type == "research":
            # Research agents: Small fixed budget for bullet points
            base_budget = 150
            position_factor = 1.0  # Constant allocation
        elif agent_type == "analysis":
            # Analysis agents: Medium budget that grows slightly with depth  
            base_budget = 250
            position_factor = 0.8 + (depth_ratio * 0.4)  # 0.8 to 1.2
        elif agent_type == "synthesis":
            # Synthesis agents: Large budget that grows significantly with depth
            base_budget = 400
            position_factor = 0.6 + (depth_ratio * 0.8)  # 0.6 to 1.4
        elif agent_type == "critic":
            # Critic agents: Small fixed budget for focused feedback
            base_budget = 100
            position_factor = 1.0  # Constant allocation
        else:
            # Default fallback
            base_budget = 300
            position_factor = 1.0
        
        # Calculate budget with position factor
        stage_budget = int(base_budget * position_factor)
        
        # Respect remaining budget constraints
        estimated_remaining_stages = max(1, int((1.0 - depth_ratio) * 3) + 1)
        max_from_remaining = remaining_budget // estimated_remaining_stages
        stage_budget = min(stage_budget, max_from_remaining)
        
        # Apply progressive reduction for strict mode
        if self.strict_mode:
            stage_budget = self._apply_progressive_reduction(stage_budget, stage)
        
        return max(self.min_budget, stage_budget)
    
    def _apply_progressive_reduction(self, stage_budget: int, stage: int) -> int:
        """Apply progressive token reduction based on stage number."""
        if stage <= 2:
            # Stages 1-2: 100% of budget
            reduction_factor = 1.0
        elif stage <= 4:
            # Stages 3-4: 75% of budget
            reduction_factor = 0.75
        else:
            # Stages 5+: 50% of budget
            reduction_factor = 0.50
        
        return int(stage_budget * reduction_factor)
    
    def _calculate_compression_ratio(self, agent_type: str, depth_ratio: float, stage: int) -> float:
        """Calculate suggested compression ratio for content refinement based on agent type."""
        # FIXED: Synthesis agents should have LOWER compression (need more space)
        if agent_type == "research":
            # Research agents always highly compressed (bullet points)
            base_compression = 0.7
        elif agent_type == "analysis":
            # Analysis agents moderately compressed (structured lists)
            base_compression = 0.5 + (depth_ratio * 0.2)  # 0.5 to 0.7
        elif agent_type == "synthesis":
            # Synthesis agents LESS compressed (need space for integration)
            base_compression = 0.1 + (depth_ratio * 0.2)  # 0.1 to 0.3 (MUCH lower)
        elif agent_type == "critic":
            # Critic agents highly compressed (focused feedback)
            base_compression = 0.8
        else:
            # Default fallback
            base_compression = 0.5
        
        # Slight increase with processing stages (but much less for synthesis)
        if agent_type == "synthesis":
            stage_factor = min(stage * 0.02, 0.1)  # Very small increase for synthesis
        else:
            stage_factor = min(stage * 0.05, 0.2)  # Normal increase for others
        
        return min(base_compression + stage_factor, 0.9)
    
    def _generate_style_guidance(self, agent_type: str, depth_ratio: float, token_budget: int) -> str:
        """Generate style guidance based on agent type, position and budget."""
        # Convert tokens to approximate word count (1 token ≈ 0.75 words)
        word_limit = int(token_budget * 0.75)
        
        # FIXED: Agent-type-specific guidance that matches their role
        if agent_type == "research":
            style = "bullet points"
            detail_level = "5-10 factual bullet points with sources"
            format_example = "• Fact 1 (Source)\n• Fact 2 (Source)"
        elif agent_type == "analysis":
            style = "structured analysis"
            detail_level = "numbered insights with connections"
            format_example = "1. Key insight\n2. Connection to other data\n3. Implications"
        elif agent_type == "synthesis":
            style = "comprehensive narrative"
            detail_level = "complete output with introduction, body, and conclusion"
            format_example = "Introduction paragraph, main content with sections, conclusion"
        elif agent_type == "critic":
            style = "focused critique"
            detail_level = "specific feedback points with suggestions"
            format_example = "Strengths: ...\nWeaknesses: ...\nSuggestions: ..."
        else:
            style = "structured response"
            detail_level = "clear organization with main points"
            format_example = "Main points with supporting details"
        
        if self.strict_mode:
            return f"⚠️ HARD LIMIT: {token_budget} tokens ({word_limit} words) MAX - OUTPUT WILL BE CUT OFF IF EXCEEDED! Use {style} format: {detail_level}. Format: {format_example}. COUNT YOUR WORDS AND STAY UNDER {word_limit}!"
        else:
            return f"TARGET: ~{token_budget} tokens (~{word_limit} words). Use {style} format: {detail_level}. Example structure: {format_example}. Aim for approximately {word_limit} words."
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current budget status for an agent."""
        if agent_id not in self._agent_budgets:
            return None
        
        remaining = self._agent_budgets[agent_id]
        used = self._agent_total_used[agent_id]
        total = remaining + used
        
        return {
            "agent_id": agent_id,
            "total_budget": total,
            "tokens_used": used,
            "tokens_remaining": remaining,
            "usage_ratio": used / total if total > 0 else 0.0
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system token usage status."""
        agent_statuses = [
            self.get_agent_status(agent_id) 
            for agent_id in self._agent_budgets.keys()
        ]
        
        total_allocated = sum(status["total_budget"] for status in agent_statuses)
        total_used = sum(status["tokens_used"] for status in agent_statuses)
        total_remaining = sum(status["tokens_remaining"] for status in agent_statuses)
        
        return {
            "total_agents": len(self._agent_budgets),
            "total_allocated": total_allocated,
            "total_used": total_used,
            "total_remaining": total_remaining,
            "system_efficiency": total_used / total_allocated if total_allocated > 0 else 0.0,
            "agent_statuses": agent_statuses,
            "strict_mode": self.strict_mode,
            "performance_target_met": total_used < 2000 if self.strict_mode else True
        }