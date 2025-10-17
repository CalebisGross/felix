"""
Specialized LLM agent types for the Felix Framework.

This module provides specialized agent implementations that extend LLMAgent
for specific roles in multi-agent coordination tasks. Each agent type has
custom prompting, behavior patterns, and processing approaches optimized
for their role in the helix-based coordination system.

Agent Types:
- ResearchAgent: Broad information gathering and exploration
- AnalysisAgent: Processing and organizing information from research
- SynthesisAgent: Integration and final output generation  
- CriticAgent: Quality assurance and review
"""

import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.agents.llm_agent import LLMAgent, LLMTask, LLMResult
from src.agents.agent import generate_spawn_times
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient
from src.llm.token_budget import TokenBudgetManager


class ResearchAgent(LLMAgent):
    """
    Research agent specializing in broad information gathering.
    
    Characteristics:
    - High creativity/temperature when at top of helix
    - Focuses on breadth over depth initially
    - Provides diverse perspectives and information sources
    - Spawns early in the process
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, research_domain: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None):
        """
        Initialize research agent.
        
        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active
            helix: Helix geometry
            llm_client: LM Studio client
            research_domain: Specific domain focus (general, technical, creative, etc.)
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per processing stage
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="research",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager
        )
        
        self.research_domain = research_domain
        self.search_queries = []
        self.information_sources = []
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create research-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Use agent's max_tokens
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget
        
        base_prompt = f"""You are a specialized RESEARCH AGENT in the Felix multi-agent system.

Research Domain: {self.research_domain}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix (0.0=start, 1.0=end)

Your Research Approach Based on Position:
"""
        
        if depth_ratio < 0.3:
            if self.token_budget_manager and self.token_budget_manager.strict_mode:
                base_prompt += """
- BULLET POINTS ONLY: 3-5 facts
- NO explanations or background
- Sources: names/dates only
- BREVITY REQUIRED
"""
            else:
                base_prompt += """
- BROAD EXPLORATION PHASE: Cast a wide net
- Generate diverse research angles and questions
- Don't worry about precision - focus on coverage
- Explore unconventional perspectives and sources
- Think creatively and associatively
"""
        elif depth_ratio < 0.7:
            if self.token_budget_manager and self.token_budget_manager.strict_mode:
                base_prompt += """
- 2-3 SPECIFIC FACTS only
- Numbers, quotes, key data
- NO context or explanation
"""
            else:
                base_prompt += """
- FOCUSED RESEARCH PHASE: Narrow down promising leads
- Build on earlier findings from other agents
- Dive deeper into specific aspects that seem relevant
- Start connecting dots and identifying patterns
- Balance breadth with increasing depth
"""
        else:
            if self.token_budget_manager and self.token_budget_manager.strict_mode:
                base_prompt += """
- FINAL FACTS: 1-2 verified points
- Citation format: Author (Year)
- NO elaboration
"""
            else:
                base_prompt += """
- DEEP RESEARCH PHASE: Precise investigation
- Focus on specific details and verification
- Provide authoritative sources and evidence
- Prepare findings for analysis agents
- Ensure accuracy and completeness
"""
        
        if self.shared_context:
            base_prompt += "\n\nContext from Other Agents:\n"
            for key, value in self.shared_context.items():
                base_prompt += f"- {key}: {value}\n"
        
        base_prompt += f"""
Task Context: {task.context}

Remember: As a research agent, your job is to gather information, not to synthesize or conclude. 
Focus on providing raw material and insights for other agents to build upon.
"""
        
        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nCompress previous research insights by ~{token_allocation.compression_ratio:.0%} while preserving key findings."
            
            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt, stage_token_budget
    
    def process_research_task(self, task: LLMTask, current_time: float) -> LLMResult:
        """Process research task with domain-specific handling."""
        # Add research-specific metadata
        enhanced_task = LLMTask(
            task_id=task.task_id,
            description=task.description,
            context=f"{task.context}\nResearch Domain: {self.research_domain}",
            metadata={**task.metadata, "research_domain": self.research_domain}
        )
        
        result = self.process_task_with_llm(enhanced_task, current_time)
        
        # Extract potential search queries and sources from the result
        self._extract_research_metadata(result)
        
        return result
    
    def _extract_research_metadata(self, result: LLMResult) -> None:
        """Extract research queries and sources from result content."""
        content = result.content.lower()
        
        # Simple heuristics to extract useful metadata
        if "search for" in content or "look up" in content:
            # Could extract specific search terms
            pass
        
        if "source:" in content or "reference:" in content:
            # Could extract cited sources
            pass


class AnalysisAgent(LLMAgent):
    """
    Analysis agent specializing in processing and organizing information.
    
    Characteristics:
    - Balanced creativity/logic for pattern recognition
    - Synthesizes information from multiple research agents
    - Identifies key insights and relationships
    - Spawns in middle of process
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, analysis_type: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None):
        """
        Initialize analysis agent.
        
        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active
            helix: Helix geometry
            llm_client: LM Studio client
            analysis_type: Analysis specialization (general, technical, critical, etc.)
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per processing stage
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="analysis",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager
        )
        
        self.analysis_type = analysis_type
        self.identified_patterns = []
        self.key_insights = []
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create analysis-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Use agent's max_tokens
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget
        
        base_prompt = f"""You are a specialized ANALYSIS AGENT in the Felix multi-agent system.

Analysis Type: {self.analysis_type}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

Your Analysis Approach:
- Process information gathered by research agents
- Identify patterns, themes, and relationships
- Organize findings into structured insights
- Look for contradictions and gaps
- Prepare organized information for synthesis agents

Analysis Focus Based on Position:
"""
        
        if depth_ratio < 0.5:
            if self.token_budget_manager and self.token_budget_manager.strict_mode:
                base_prompt += """
- 2 PATTERNS maximum
- Numbered list format
- NO explanations
"""
            else:
                base_prompt += """
- PATTERN IDENTIFICATION: Look for themes and connections
- Organize information into categories
- Identify what's missing or contradictory
"""
        else:
            if self.token_budget_manager and self.token_budget_manager.strict_mode:
                base_prompt += """
- PRIORITY RANKING: Top 3 insights
- 1. 2. 3. format
- NO background
"""
            else:
                base_prompt += """
- DEEP ANALYSIS: Provide detailed evaluation
- Prioritize insights by importance
- Structure findings for final synthesis
"""
        
        if self.shared_context:
            base_prompt += "\n\nInformation from Research Agents:\n"
            research_items = {k: v for k, v in self.shared_context.items() if "research" in k.lower()}
            for key, value in research_items.items():
                base_prompt += f"- {key}: {value}\n"
        
        base_prompt += f"""
Task Context: {task.context}

Remember: Your job is to process and organize information, not to make final decisions.
Focus on creating clear, structured insights for synthesis agents to use.
"""
        
        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nCompress analysis by ~{token_allocation.compression_ratio:.0%} while preserving key patterns and insights."
            
            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt, stage_token_budget


class SynthesisAgent(LLMAgent):
    """
    Synthesis agent specializing in integration and final output generation.
    
    Characteristics:
    - Lower temperature for precise, coherent output
    - Integrates work from research and analysis agents
    - Makes final decisions and creates deliverables
    - Spawns late in process, near helix bottom
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, output_format: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None):
        """
        Initialize synthesis agent.
        
        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active
            helix: Helix geometry
            llm_client: LM Studio client
            output_format: Desired output format (report, summary, decision, etc.)
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per processing stage
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="synthesis",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager
        )
        
        self.output_format = output_format
        self.final_output = None
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create synthesis-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Use agent's max_tokens
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget
        
        base_prompt = f"""You are a specialized SYNTHESIS AGENT in the Felix multi-agent system.

Output Format: {self.output_format}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix (near the focused end)

Your Synthesis Approach:
- Integrate ALL previous work from research and analysis agents
- Create coherent, comprehensive final output
- Make final decisions and conclusions
- Ensure completeness and quality
- Focus on clarity and actionability

STRICT MODE OVERRIDE: If token budget < 150, provide ONLY final output in 2-3 paragraphs, NO explanations.

Previous Agent Work:
"""
        
        if self.shared_context:
            for key, value in self.shared_context.items():
                base_prompt += f"- {key}: {value}\n"
        
        base_prompt += f"""
Task Context: {task.context}

Your task is to create the final deliverable that integrates all previous work.
Be decisive, comprehensive, and ensure the output serves the original task goal.
"""
        
        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nFocus on synthesizing key points with ~{token_allocation.compression_ratio:.0%} compression from all sources."
            
            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt, stage_token_budget
    
    def finalize_output(self, result: LLMResult) -> Dict[str, Any]:
        """Create final output package with all metadata."""
        self.final_output = {
            "content": result.content,
            "metadata": {
                "agent_id": self.agent_id,
                "output_format": self.output_format,
                "position_info": result.position_info,
                "tokens_used": result.llm_response.tokens_used,
                "processing_time": result.processing_time,
                "timestamp": result.timestamp,
                "synthesis_quality_score": self._calculate_quality_score(result)
            },
            "source_agents": list(self.shared_context.keys()),
            "task_completion": True
        }
        
        return self.final_output
    
    def _calculate_quality_score(self, result: LLMResult) -> float:
        """Calculate a quality score for the synthesis."""
        # Simple heuristic based on content length, context integration, etc.
        base_score = min(len(result.content) / 1000, 1.0)  # Length factor
        context_score = min(len(self.shared_context) / 5, 1.0)  # Context integration
        
        return (base_score + context_score) / 2


class CriticAgent(LLMAgent):
    """
    Critic agent specializing in quality assurance and review.
    
    Characteristics:
    - Critical evaluation of other agents' work
    - Identifies gaps, errors, and improvements
    - Provides quality feedback and suggestions
    - Can spawn at various points for ongoing QA
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, review_focus: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None):
        """
        Initialize critic agent.
        
        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active
            helix: Helix geometry
            llm_client: LM Studio client
            review_focus: Review focus (accuracy, completeness, style, logic, etc.)
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per processing stage
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="critic",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager
        )
        
        self.review_focus = review_focus
        self.identified_issues = []
        self.suggestions = []
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create critic-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Use agent's max_tokens
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget
        
        base_prompt = f"""You are a specialized CRITIC AGENT in the Felix multi-agent system.

Review Focus: {self.review_focus}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

Your Critical Review Approach:
- Evaluate work from other agents with a critical eye
- Identify gaps, errors, inconsistencies, and weak points
- Suggest specific improvements and corrections
- Ensure quality standards are maintained
- Be constructive but thorough in your criticism

STRICT MODE OVERRIDE: If token budget < 100, list ONLY 3 specific issues in numbered format, NO background.

Work to Review:
"""
        
        if self.shared_context:
            for key, value in self.shared_context.items():
                base_prompt += f"- {key}: {value}\n"
        
        base_prompt += f"""
Task Context: {task.context}

Focus your review on {self.review_focus}. Provide specific, actionable feedback.
Be thorough but constructive - the goal is to improve the final output quality.
"""
        
        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nProvide focused critique with ~{token_allocation.compression_ratio:.0%} compression while covering key quality issues."
            
            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt, stage_token_budget


def create_specialized_team(helix: HelixGeometry, llm_client: LMStudioClient,
                          task_complexity: str = "medium", 
                          token_budget_manager: Optional[TokenBudgetManager] = None,
                          random_seed: Optional[int] = None) -> List[LLMAgent]:
    """
    Create a balanced team of specialized agents for a task.
    
    Args:
        helix: Helix geometry
        llm_client: LM Studio client
        task_complexity: Complexity level (simple, medium, complex)
        token_budget_manager: Optional token budget manager for all agents
        random_seed: Optional seed for spawn time randomization
        
    Returns:
        List of specialized agents with randomized spawn times
    """
    if task_complexity == "simple":
        return _create_simple_team(helix, llm_client, token_budget_manager, random_seed)
    elif task_complexity == "medium":
        return _create_medium_team(helix, llm_client, token_budget_manager, random_seed)
    else:  # complex
        return _create_complex_team(helix, llm_client, token_budget_manager, random_seed)


def _create_simple_team(helix: HelixGeometry, llm_client: LMStudioClient, 
                       token_budget_manager: Optional[TokenBudgetManager] = None,
                       random_seed: Optional[int] = None) -> List[LLMAgent]:
    """Create team for simple tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)
    
    # Generate random spawn times within appropriate ranges for each agent type
    research_spawn = random.uniform(0.05, 0.25)  # Research agents spawn early
    analysis_spawn = random.uniform(0.3, 0.7)    # Analysis agents in middle
    synthesis_spawn = random.uniform(0.7, 0.95)  # Synthesis agents late
    
    return [
        ResearchAgent("research_001", research_spawn, helix, llm_client, 
                     token_budget_manager=token_budget_manager, max_tokens=800),
        AnalysisAgent("analysis_001", analysis_spawn, helix, llm_client, 
                     token_budget_manager=token_budget_manager, max_tokens=800),
        SynthesisAgent("synthesis_001", synthesis_spawn, helix, llm_client, 
                      token_budget_manager=token_budget_manager, max_tokens=800)
    ]


def _create_medium_team(helix: HelixGeometry, llm_client: LMStudioClient, 
                       token_budget_manager: Optional[TokenBudgetManager] = None,
                       random_seed: Optional[int] = None) -> List[LLMAgent]:
    """Create team for medium complexity tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)
    
    # Generate random spawn times within appropriate ranges
    research_spawns = [random.uniform(0.02, 0.2) for _ in range(2)]
    analysis_spawns = [random.uniform(0.25, 0.65) for _ in range(2)]
    critic_spawn = random.uniform(0.6, 0.8)
    synthesis_spawn = random.uniform(0.8, 0.95)
    
    # Sort to maintain some ordering within types
    research_spawns.sort()
    analysis_spawns.sort()
    
    return [
        ResearchAgent("research_001", research_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        ResearchAgent("research_002", research_spawns[1], helix, llm_client, "technical", token_budget_manager, 800),
        AnalysisAgent("analysis_001", analysis_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        AnalysisAgent("analysis_002", analysis_spawns[1], helix, llm_client, "critical", token_budget_manager, 800),
        CriticAgent("critic_001", critic_spawn, helix, llm_client, "accuracy", token_budget_manager, 800),
        SynthesisAgent("synthesis_001", synthesis_spawn, helix, llm_client, "general", token_budget_manager, 800)
    ]


def _create_complex_team(helix: HelixGeometry, llm_client: LMStudioClient,
                        token_budget_manager: Optional[TokenBudgetManager] = None,
                        random_seed: Optional[int] = None) -> List[LLMAgent]:
    """Create team for complex tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)
    
    # Generate random spawn times within appropriate ranges
    research_spawns = [random.uniform(0.01, 0.25) for _ in range(3)]
    analysis_spawns = [random.uniform(0.2, 0.7) for _ in range(3)]
    critic_spawns = [random.uniform(0.6, 0.8) for _ in range(2)]
    synthesis_spawns = [random.uniform(0.8, 0.98) for _ in range(2)]
    
    # Sort to maintain some ordering within types
    research_spawns.sort()
    analysis_spawns.sort()
    critic_spawns.sort()
    synthesis_spawns.sort()
    
    return [
        ResearchAgent("research_001", research_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        ResearchAgent("research_002", research_spawns[1], helix, llm_client, "technical", token_budget_manager, 800),
        ResearchAgent("research_003", research_spawns[2], helix, llm_client, "creative", token_budget_manager, 800),
        AnalysisAgent("analysis_001", analysis_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        AnalysisAgent("analysis_002", analysis_spawns[1], helix, llm_client, "technical", token_budget_manager, 800),
        AnalysisAgent("analysis_003", analysis_spawns[2], helix, llm_client, "critical", token_budget_manager, 800),
        CriticAgent("critic_001", critic_spawns[0], helix, llm_client, "accuracy", token_budget_manager, 800),
        CriticAgent("critic_002", critic_spawns[1], helix, llm_client, "completeness", token_budget_manager, 800),
        SynthesisAgent("synthesis_001", synthesis_spawns[0], helix, llm_client, "report", token_budget_manager, 800),
        SynthesisAgent("synthesis_002", synthesis_spawns[1], helix, llm_client, "executive_summary", token_budget_manager, 800)
    ]
