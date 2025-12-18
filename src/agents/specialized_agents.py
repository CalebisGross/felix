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
- SystemAgent: System operations and command execution (see system_agent.py)
"""

import time
import random
import logging
import traceback
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.agents.llm_agent import LLMAgent, LLMTask
from src.agents.agent import generate_spawn_times
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)
from src.llm.token_budget import TokenBudgetManager
from src.llm.web_search_client import WebSearchClient  # For type hints only


# Shared tool instructions header for all specialized agents
# Imperative execution directive - used when memory system unavailable
# Primary tool instructions should come from KnowledgeStore via conditional retrieval
EXECUTION_DIRECTIVE = """⚡ TOOL EXECUTION PROTOCOL:

🔍 WEB SEARCH - Execute for current/real-time information:
Write on its own line: WEB_SEARCH_NEEDED: [your query]

🖥️ SYSTEM COMMANDS - Execute for file operations and system checks:
Write on its own line: SYSTEM_ACTION_NEEDED: [command]

Examples:
SYSTEM_ACTION_NEEDED: date
SYSTEM_ACTION_NEEDED: head -n [N] filename.py
SYSTEM_ACTION_NEEDED: mkdir -p results && echo "content" > results/file.txt

Commands requiring approval (file writes, installs) will prompt user.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

# Backwards compatibility alias (deprecated - use EXECUTION_DIRECTIVE)
MINIMAL_TOOLS_FALLBACK = EXECUTION_DIRECTIVE


class ResearchAgent(LLMAgent):
    """
    Research agent specializing in broad information gathering.

    Characteristics:
    - High creativity/temperature when at top of helix
    - Focuses on breadth over depth initially
    - Provides several perspectives and information sources
    - Spawns early in the process

    Note: Web search is handled by CentralPost's WebSearchCoordinator.
    Search results are available via task.knowledge_entries (domain="web_search").
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, research_domain: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None,
                 web_search_client: Optional[WebSearchClient] = None,  # Legacy parameter (ignored)
                 max_web_queries: int = 3,  # Legacy parameter (ignored)
                 prompt_manager: Optional['PromptManager'] = None,
                 prompt_optimizer: Optional['PromptOptimizer'] = None):
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
            web_search_client: Legacy parameter (ignored). Web search handled by CentralPost WebSearchCoordinator.
            max_web_queries: Legacy parameter (ignored).
            prompt_manager: Optional prompt manager for custom prompts
            prompt_optimizer: Optional prompt optimizer for learning and optimization
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="research",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager,
            prompt_optimizer=prompt_optimizer
        )

        self.research_domain = research_domain
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """
        Create research-specific system prompt with token budget.

        Uses base class _try_direct_answer_mode() for simple factual queries,
        otherwise delegates to PromptPipeline for standard prompt construction.
        """
        # Try direct answer mode first (inherited from LLMAgent base class)
        direct_result = self._try_direct_answer_mode(task)
        if direct_result is not None:
            return direct_result

        # NORMAL MODE: Delegate to PromptPipeline for standard research agent prompt
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        stage_token_budget = self.max_tokens  # Default
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        # Delegate to PromptPipeline
        try:
            result = self.prompt_pipeline.build_agent_prompt(
                task=task,
                agent=self,
                position_info=position_info,
                current_time=current_time
            )
        except Exception as e:
            logger.error(f"Exception in build_agent_prompt(): {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise

        return result.system_prompt, stage_token_budget

    def _determine_prompt_key(self, depth_ratio: float, strict_mode: bool) -> str:
        """Determine prompt key based on depth and mode."""
        mode_suffix = "strict" if strict_mode else "normal"

        if depth_ratio < 0.3:
            return f"research_exploration_{mode_suffix}"
        elif depth_ratio < 0.7:
            return f"research_focused_{mode_suffix}"
        else:
            return f"research_deep_{mode_suffix}"

    def _get_agent_traits(self) -> Dict[str, Any]:
        """Return research-specific traits for synthesis context."""
        return {"research_domain": self.research_domain}



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
                 max_tokens: Optional[int] = None,
                 prompt_manager: Optional['PromptManager'] = None,
                 prompt_optimizer: Optional['PromptOptimizer'] = None):
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
            prompt_manager: Optional prompt manager for custom prompts
            prompt_optimizer: Optional prompt optimizer for learning and optimization
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="analysis",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager,
            prompt_optimizer=prompt_optimizer
        )

        self.analysis_type = analysis_type
        self.identified_patterns = []
        self.key_insights = []
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create analysis-specific system prompt with token budget using PromptPipeline."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Calculate token budget for this stage
        stage_token_budget = self.max_tokens
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        # Delegate to PromptPipeline for unified prompt construction
        result = self.prompt_pipeline.build_agent_prompt(
            task=task,
            agent=self,
            position_info=position_info,
            current_time=current_time
        )

        return result.system_prompt, stage_token_budget

    def _get_agent_traits(self) -> Dict[str, Any]:
        """Return analysis-specific traits for synthesis context."""
        return {"analysis_type": self.analysis_type}


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
                 max_tokens: Optional[int] = None,
                 prompt_manager: Optional['PromptManager'] = None,
                 prompt_optimizer: Optional['PromptOptimizer'] = None):
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
            prompt_manager: Optional prompt manager for custom prompts
            prompt_optimizer: Optional prompt optimizer for learning and optimization
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="critic",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager,
            prompt_optimizer=prompt_optimizer
        )

        self.review_focus = review_focus
        self.identified_issues = []
        self.suggestions = []
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create critic-specific system prompt with token budget using PromptPipeline."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Calculate token budget for this stage
        stage_token_budget = self.max_tokens
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        # Delegate to PromptPipeline for unified prompt construction
        result = self.prompt_pipeline.build_agent_prompt(
            task=task,
            agent=self,
            position_info=position_info,
            current_time=current_time
        )

        return result.system_prompt, stage_token_budget

    def _get_agent_traits(self) -> Dict[str, Any]:
        """Return critic-specific traits for synthesis context."""
        return {"review_focus": self.review_focus}

    def evaluate_reasoning_process(self, agent_output: Dict[str, Any],
                                   agent_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the reasoning process quality of another agent's output.

        This extends CriticAgent beyond content evaluation to assess HOW agents reasoned,
        not just WHAT they produced. Implements meta-cognitive evaluation for self-improvement.

        Args:
            agent_output: Agent's output including result and metadata
            agent_metadata: Optional metadata about agent's reasoning process

        Returns:
            Dictionary with reasoning evaluation:
                - reasoning_quality_score: 0.0-1.0
                - logical_coherence: 0.0-1.0
                - evidence_quality: 0.0-1.0
                - methodology_appropriateness: 0.0-1.0
                - identified_issues: List[str]
                - improvement_recommendations: List[str]
                - re_evaluation_needed: bool
        """
        result = agent_output.get('result', '')
        confidence = agent_output.get('confidence', 0.5)
        agent_id = agent_output.get('agent_id', 'unknown')

        # Initialize evaluation
        issues = []
        recommendations = []
        scores = {'logical_coherence': 0.5, 'evidence_quality': 0.5, 'methodology': 0.5}

        # 1. Evaluate logical coherence
        if self._has_logical_fallacies(result):
            issues.append("Contains potential logical fallacies")
            scores['logical_coherence'] = 0.4
            recommendations.append("Review reasoning chain for logical consistency")
        else:
            scores['logical_coherence'] = 0.8

        # 2. Evaluate evidence quality
        if self._has_weak_evidence(result):
            issues.append("Evidence appears weak or unsupported")
            scores['evidence_quality'] = 0.4
            recommendations.append("Strengthen claims with more reliable evidence")
        else:
            scores['evidence_quality'] = 0.8

        # 3. Evaluate methodology appropriateness
        if agent_metadata:
            agent_type = agent_metadata.get('agent_type', 'unknown')
            if not self._methodology_appropriate(result, agent_type):
                issues.append(f"Methodology not well-suited for {agent_type} agent")
                scores['methodology'] = 0.4
                recommendations.append(f"Consider approaches more aligned with {agent_type} role")
            else:
                scores['methodology'] = 0.8
        else:
            # No metadata, default moderate score
            scores['methodology'] = 0.6

        # 4. Check reasoning depth
        if len(result.split()) < 50:
            issues.append("Reasoning appears shallow - insufficient depth")
            recommendations.append("Provide more detailed reasoning and analysis")
            # Penalize all scores slightly
            for key in scores:
                scores[key] *= 0.9

        # 5. Check for over/under confidence
        avg_score = sum(scores.values()) / len(scores)
        confidence_gap = abs(confidence - avg_score)
        if confidence_gap > 0.3:
            if confidence > avg_score:
                issues.append(f"Agent appears overconfident (confidence={confidence:.2f} vs quality={avg_score:.2f})")
                recommendations.append("Calibrate confidence based on reasoning quality")
            else:
                issues.append(f"Agent appears underconfident (confidence={confidence:.2f} vs quality={avg_score:.2f})")
                recommendations.append("Increase confidence when reasoning is solid")

        # Calculate overall reasoning quality score
        reasoning_quality = sum(scores.values()) / len(scores)

        # Determine if re-evaluation is needed
        re_evaluation_needed = reasoning_quality < 0.5 or len(issues) >= 3

        logger.info(f"🧠 Reasoning evaluation for {agent_id}:")
        logger.info(f"   Quality: {reasoning_quality:.2f}, Coherence: {scores['logical_coherence']:.2f}, "
                   f"Evidence: {scores['evidence_quality']:.2f}, Methodology: {scores['methodology']:.2f}")
        if issues:
            logger.info(f"   Issues: {', '.join(issues)}")
        if re_evaluation_needed:
            logger.warning(f"   ⚠️  Re-evaluation recommended for {agent_id}")

        return {
            'reasoning_quality_score': reasoning_quality,
            'logical_coherence': scores['logical_coherence'],
            'evidence_quality': scores['evidence_quality'],
            'methodology_appropriateness': scores['methodology'],
            'identified_issues': issues,
            'improvement_recommendations': recommendations,
            're_evaluation_needed': re_evaluation_needed,
            'agent_id': agent_id
        }

    def _has_logical_fallacies(self, text: str) -> bool:
        """Check for common logical fallacies in reasoning."""
        fallacy_indicators = [
            'everyone knows', 'obviously', 'clearly', 'it goes without saying',
            'all experts agree', 'no one would disagree', 'always', 'never'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in fallacy_indicators)

    def _has_weak_evidence(self, text: str) -> bool:
        """Check if evidence appears weak or unsupported."""
        weak_indicators = [
            'i think', 'i believe', 'probably', 'maybe', 'might be',
            'could be', 'seems like', 'appears to'
        ]
        text_lower = text.lower()
        # Count weak indicators
        weak_count = sum(1 for indicator in weak_indicators if indicator in text_lower)
        # High proportion of weak language suggests weak evidence
        return weak_count > 3

    def _methodology_appropriate(self, text: str, agent_type: str) -> bool:
        """Check if reasoning methodology is appropriate for agent type."""
        if agent_type == 'research':
            # Research should explore multiple perspectives
            return 'perspective' in text.lower() or 'source' in text.lower()
        elif agent_type == 'analysis':
            # Analysis should break things down
            return 'because' in text.lower() or 'therefore' in text.lower()
        elif agent_type == 'critic':
            # Critics should identify issues
            return 'issue' in text.lower() or 'problem' in text.lower() or 'improve' in text.lower()
        else:
            # Default: accept methodology
            return True


def create_specialized_team(helix: HelixGeometry, llm_client: LMStudioClient,
                          task_complexity: str = "medium",
                          token_budget_manager: Optional[TokenBudgetManager] = None,
                          random_seed: Optional[int] = None,
                          web_search_client: Optional[WebSearchClient] = None,
                          max_web_queries: int = 3) -> List[LLMAgent]:
    """
    Create a balanced team of specialized agents for a task.

    Args:
        helix: Helix geometry
        llm_client: LM Studio client
        task_complexity: Complexity level (simple, medium, complex)
        token_budget_manager: Optional token budget manager for all agents
        random_seed: Optional seed for spawn time randomization
        web_search_client: Optional web search client for Research agents
        max_web_queries: Maximum web queries per research agent (default: 3)

    Returns:
        List of specialized agents with randomized spawn times
    """
    if task_complexity == "simple":
        return _create_simple_team(helix, llm_client, token_budget_manager, random_seed,
                                  web_search_client, max_web_queries)
    elif task_complexity == "medium":
        return _create_medium_team(helix, llm_client, token_budget_manager, random_seed,
                                   web_search_client, max_web_queries)
    else:  # complex
        return _create_complex_team(helix, llm_client, token_budget_manager, random_seed,
                                    web_search_client, max_web_queries)


def _create_simple_team(helix: HelixGeometry, llm_client: LMStudioClient,
                       token_budget_manager: Optional[TokenBudgetManager] = None,
                       random_seed: Optional[int] = None,
                       web_search_client: Optional[WebSearchClient] = None,
                       max_web_queries: int = 3) -> List[LLMAgent]:
    """Create team for simple tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random spawn times within appropriate ranges for each agent type
    research_spawn = random.uniform(0.05, 0.25)  # Research agents spawn early
    analysis_spawn = random.uniform(0.3, 0.7)    # Analysis agents in middle
    # Note: Synthesis is now handled by CentralPost, not a specialized agent

    return [
        ResearchAgent("research_001", research_spawn, helix, llm_client,
                     token_budget_manager=token_budget_manager, max_tokens=16000,
                     web_search_client=web_search_client, max_web_queries=max_web_queries),
        AnalysisAgent("analysis_001", analysis_spawn, helix, llm_client,
                     token_budget_manager=token_budget_manager, max_tokens=16000)
    ]


def _create_medium_team(helix: HelixGeometry, llm_client: LMStudioClient,
                       token_budget_manager: Optional[TokenBudgetManager] = None,
                       random_seed: Optional[int] = None,
                       web_search_client: Optional[WebSearchClient] = None,
                       max_web_queries: int = 3) -> List[LLMAgent]:
    """Create team for medium complexity tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random spawn times within appropriate ranges
    research_spawns = [random.uniform(0.02, 0.2) for _ in range(2)]
    analysis_spawns = [random.uniform(0.25, 0.65) for _ in range(2)]
    critic_spawn = random.uniform(0.6, 0.8)
    # Note: Synthesis is now handled by CentralPost, not a specialized agent

    # Sort to maintain some ordering within types
    research_spawns.sort()
    analysis_spawns.sort()

    return [
        ResearchAgent("research_001", research_spawns[0], helix, llm_client, "general",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        ResearchAgent("research_002", research_spawns[1], helix, llm_client, "technical",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        AnalysisAgent("analysis_001", analysis_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        AnalysisAgent("analysis_002", analysis_spawns[1], helix, llm_client, "critical", token_budget_manager, 800),
        CriticAgent("critic_001", critic_spawn, helix, llm_client, "accuracy", token_budget_manager, 800)
    ]


def _create_complex_team(helix: HelixGeometry, llm_client: LMStudioClient,
                        token_budget_manager: Optional[TokenBudgetManager] = None,
                        random_seed: Optional[int] = None,
                        web_search_client: Optional[WebSearchClient] = None,
                        max_web_queries: int = 3) -> List[LLMAgent]:
    """Create team for complex tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random spawn times within appropriate ranges
    research_spawns = [random.uniform(0.01, 0.25) for _ in range(3)]
    analysis_spawns = [random.uniform(0.2, 0.7) for _ in range(3)]
    critic_spawns = [random.uniform(0.6, 0.8) for _ in range(2)]
    # Note: Synthesis is now handled by CentralPost, not a specialized agent

    # Sort to maintain some ordering within types
    research_spawns.sort()
    analysis_spawns.sort()
    critic_spawns.sort()

    return [
        ResearchAgent("research_001", research_spawns[0], helix, llm_client, "general",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        ResearchAgent("research_002", research_spawns[1], helix, llm_client, "technical",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        ResearchAgent("research_003", research_spawns[2], helix, llm_client, "creative",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        AnalysisAgent("analysis_001", analysis_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        AnalysisAgent("analysis_002", analysis_spawns[1], helix, llm_client, "technical", token_budget_manager, 800),
        AnalysisAgent("analysis_003", analysis_spawns[2], helix, llm_client, "critical", token_budget_manager, 800),
        CriticAgent("critic_001", critic_spawns[0], helix, llm_client, "accuracy", token_budget_manager, 800),
        CriticAgent("critic_002", critic_spawns[1], helix, llm_client, "completeness", token_budget_manager, 800)
    ]
