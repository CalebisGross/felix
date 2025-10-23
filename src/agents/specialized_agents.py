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
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.agents.llm_agent import LLMAgent, LLMTask, LLMResult
from src.agents.agent import generate_spawn_times
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)
from src.llm.token_budget import TokenBudgetManager
from src.llm.web_search_client import WebSearchClient, SearchResult


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
                 max_tokens: Optional[int] = None,
                 web_search_client: Optional[WebSearchClient] = None,  # DEPRECATED: Web search now handled by CentralPost
                 max_web_queries: int = 3):  # DEPRECATED
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
            web_search_client: DEPRECATED - Web search now handled by CentralPost
            max_web_queries: DEPRECATED
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
        # Note: web_search_client parameters kept for backward compatibility but ignored
        # Web search is now performed by CentralPost when confidence is low
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create research-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Check for "direct answer mode" - when trustable knowledge exists for simple query
        logger.info(f"🔍 DIRECT ANSWER MODE CHECK for {self.agent_id}")
        logger.info(f"   task.knowledge_entries exists: {hasattr(task, 'knowledge_entries')}")
        if hasattr(task, 'knowledge_entries'):
            logger.info(f"   task.knowledge_entries length: {len(task.knowledge_entries) if task.knowledge_entries else 0}")

        use_direct_mode = False
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            logger.info(f"   ✓ Knowledge entries available: {len(task.knowledge_entries)}")
            # Check if this is a simple factual query with trustable knowledge
            try:
                from src.workflows.truth_assessment import assess_answer_confidence

                logger.info(f"   🤖 Calling assess_answer_confidence()...")
                # Quick assessment
                trustable, trust_score, trust_reason = assess_answer_confidence(
                    task.knowledge_entries,
                    task.description
                )

                logger.info(f"   📊 Trust assessment results:")
                logger.info(f"      Trustable: {trustable}")
                logger.info(f"      Trust score: {trust_score:.2f}")
                logger.info(f"      Reason: {trust_reason}")

                if trustable and trust_score >= 0.85:
                    use_direct_mode = True
                    logger.info(f"🎯 DIRECT ANSWER MODE ACTIVATED for {self.agent_id}")
                    logger.info(f"   Reason: {trust_reason}")
                else:
                    logger.info(f"   ⚠️ Direct answer mode NOT activated:")
                    logger.info(f"      trustable={trustable}, trust_score={trust_score:.2f} (need ≥0.85)")
            except Exception as e:
                logger.error(f"   ❌ Could not assess for direct mode: {e}", exc_info=True)
        else:
            logger.warning(f"   ⚠️ No knowledge entries available - direct answer mode cannot activate")

        # DIRECT ANSWER MODE: Override for simple queries with trustable knowledge
        if use_direct_mode:
            # Force low temperature for precision
            # This will be applied in process_task_with_llm by checking task metadata
            if not hasattr(task, 'metadata'):
                task.metadata = {}
            task.metadata['direct_answer_mode'] = True
            task.metadata['override_temperature'] = 0.2
            task.metadata['override_tokens'] = 200

            # Build direct answer prompt
            knowledge_summary = "\n\nAVAILABLE KNOWLEDGE (HIGH CONFIDENCE):\n"
            for entry in task.knowledge_entries:
                if hasattr(entry, 'content'):
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                # Show full content for direct answer mode (no truncation)
                knowledge_summary += f"• {content_str}\n"

            base_prompt = f"""You are answering a SIMPLE FACTUAL QUESTION with HIGH confidence knowledge available.

{knowledge_summary}

🎯 DIRECT ANSWER INSTRUCTIONS:
- State ONLY the direct factual answer from the knowledge above
- 1-2 sentences maximum (15-30 words)
- NO exploration, NO analysis, NO elaboration, NO uncertainty
- NO "according to" or "based on" qualifiers
- Format: "The [answer] is [fact]." or "Current [X] is [Y]."

Example good responses:
- "The current date is October 23, 2025."
- "The time is 1:24 PM EDT."
- "The answer is 42."

Example bad responses (TOO LONG):
- "Based on the available sources, the current date and time vary by location..."
- "Multiple authoritative sources confirm that the date is..."

Your response (15-30 words, direct answer only):"""

            return base_prompt, 200  # Force low token budget

        # NORMAL MODE: Standard research agent behavior
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

        # Add knowledge entries if available
        knowledge_summary = ""
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            knowledge_summary = "\n\nRelevant Knowledge from Memory:\n"
            for entry in task.knowledge_entries:
                # Extract key information from knowledge entry
                if hasattr(entry, 'content'):
                    # Extract 'result' key from dictionary if present (web search results)
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                confidence = entry.confidence_level.value if hasattr(entry, 'confidence_level') else "unknown"
                source = entry.source_agent if hasattr(entry, 'source_agent') else "system"
                domain = entry.domain if hasattr(entry, 'domain') else "unknown"

                # Use longer truncation for web_search domain (detailed factual data)
                max_chars = 400 if domain == "web_search" else 200
                if len(content_str) > max_chars:
                    content_str = content_str[:max_chars-3] + "..."

                # Add emoji prefix for web search entries
                prefix = "🌐" if domain == "web_search" else "📝"
                knowledge_summary += f"{prefix} [{source}, conf: {confidence}]: {content_str}\n"

            # Add important instructions for using available knowledge
            knowledge_summary += "\nIMPORTANT: Use the knowledge provided above to answer the task if possible. "
            knowledge_summary += "Only request additional web search if the available knowledge is insufficient or outdated.\n"

        base_prompt += knowledge_summary
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

    def process_research_task(self, task: LLMTask, current_time: float,
                              central_post: Optional['CentralPost'] = None) -> LLMResult:
        """Process research task with domain-specific handling and optional web search."""
        import logging
        logger = logging.getLogger(__name__)

        # Get position info to determine if we should search
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Perform web search if enabled and in early exploration phase (0.0-0.3)
        web_search_context = ""
        if self.web_search_client and depth_ratio <= 0.3:
            logger.info(f"[{self.agent_id}] Performing web search at depth {depth_ratio:.2f}")

            # Formulate search queries based on task and research domain
            search_queries = self._formulate_search_queries(task)

            # Perform searches (up to max_web_queries)
            all_search_results = []
            for i, query in enumerate(search_queries[:self.max_web_queries]):
                logger.info(f"[{self.agent_id}] Searching: '{query}'")
                try:
                    results = self.web_search_client.search(
                        query=query,
                        task_id=task.task_id
                    )
                    all_search_results.extend(results)
                    self.search_queries.append(query)

                    # Extract sources
                    for result in results:
                        if result.url not in self.information_sources:
                            self.information_sources.append(result.url)

                except Exception as e:
                    logger.error(f"[{self.agent_id}] Web search failed for '{query}': {e}")

            # Store results
            self.web_search_results = all_search_results

            # Format results for LLM
            if all_search_results:
                web_search_context = "\n\n" + self.web_search_client.format_results_for_llm(all_search_results)
                logger.info(f"[{self.agent_id}] Found {len(all_search_results)} web search results")

        # Add research-specific metadata and web search results
        enhanced_context = f"{task.context}\nResearch Domain: {self.research_domain}{web_search_context}"

        enhanced_task = LLMTask(
            task_id=task.task_id,
            description=task.description,
            context=enhanced_context,
            metadata={
                **task.metadata,
                "research_domain": self.research_domain,
                "web_search_enabled": self.web_search_client is not None,
                "web_search_results_count": len(self.web_search_results)
            }
        )

        result = super().process_task_with_llm(enhanced_task, current_time, central_post)

        # Add metadata to result for tracking
        result.metadata = enhanced_task.metadata

        # Extract potential search queries and sources from the result
        self._extract_research_metadata(result)

        return result

    def _formulate_search_queries(self, task: LLMTask) -> List[str]:
        """
        Formulate search queries based on task and research domain.

        Args:
            task: The research task

        Returns:
            List of search query strings
        """
        queries = []

        # Base query from task description
        base_query = task.description.strip()

        # Add domain-specific queries
        if self.research_domain != "general":
            queries.append(f"{base_query} {self.research_domain}")
        else:
            queries.append(base_query)

        # Add variations based on research domain
        if self.research_domain == "technical":
            queries.append(f"{base_query} documentation tutorial")
        elif self.research_domain == "creative":
            queries.append(f"{base_query} examples ideas inspiration")
        elif self.research_domain == "general":
            queries.append(f"{base_query} overview guide")

        # Add a focused "latest" query for current information
        if len(queries) < 3:
            queries.append(f"{base_query} latest 2024 2025")

        return queries
    
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

        # Add knowledge entries if available
        knowledge_summary = ""
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            knowledge_summary = "\n\nRelevant Knowledge from Memory:\n"
            for entry in task.knowledge_entries:
                # Extract key information from knowledge entry
                if hasattr(entry, 'content'):
                    # Extract 'result' key from dictionary if present (web search results)
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                confidence = entry.confidence_level.value if hasattr(entry, 'confidence_level') else "unknown"
                source = entry.source_agent if hasattr(entry, 'source_agent') else "system"
                domain = entry.domain if hasattr(entry, 'domain') else "unknown"

                # Use longer truncation for web_search domain (detailed factual data)
                max_chars = 400 if domain == "web_search" else 200
                if len(content_str) > max_chars:
                    content_str = content_str[:max_chars-3] + "..."

                # Add emoji prefix for web search entries
                prefix = "🌐" if domain == "web_search" else "📝"
                knowledge_summary += f"{prefix} [{source}, conf: {confidence}]: {content_str}\n"

            # Add important instructions for using available knowledge
            knowledge_summary += "\nIMPORTANT: Use the knowledge provided above to answer the task if possible. "
            knowledge_summary += "Only request additional web search if the available knowledge is insufficient or outdated.\n"

        base_prompt += knowledge_summary
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

STRICT MODE OVERRIDE: If token budget < 300, list key issues in numbered format with brief explanations. Otherwise, provide comprehensive detailed critique.

Work to Review:
"""
        
        if self.shared_context:
            for key, value in self.shared_context.items():
                base_prompt += f"- {key}: {value}\n"

        # Add knowledge entries if available
        knowledge_summary = ""
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            knowledge_summary = "\n\nRelevant Knowledge from Memory:\n"
            for entry in task.knowledge_entries:
                # Extract key information from knowledge entry
                if hasattr(entry, 'content'):
                    # Extract 'result' key from dictionary if present (web search results)
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                confidence = entry.confidence_level.value if hasattr(entry, 'confidence_level') else "unknown"
                source = entry.source_agent if hasattr(entry, 'source_agent') else "system"
                domain = entry.domain if hasattr(entry, 'domain') else "unknown"

                # Use longer truncation for web_search domain (detailed factual data)
                max_chars = 400 if domain == "web_search" else 200
                if len(content_str) > max_chars:
                    content_str = content_str[:max_chars-3] + "..."

                # Add emoji prefix for web search entries
                prefix = "🌐" if domain == "web_search" else "📝"
                knowledge_summary += f"{prefix} [{source}, conf: {confidence}]: {content_str}\n"

            # Add important instructions for using available knowledge
            knowledge_summary += "\nIMPORTANT: Use the knowledge provided above to answer the task if possible. "
            knowledge_summary += "Only request additional web search if the available knowledge is insufficient or outdated.\n"

        base_prompt += knowledge_summary
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
    synthesis_spawn = random.uniform(0.7, 0.95)  # Synthesis agents late

    return [
        ResearchAgent("research_001", research_spawn, helix, llm_client,
                     token_budget_manager=token_budget_manager, max_tokens=800,
                     web_search_client=web_search_client, max_web_queries=max_web_queries),
        AnalysisAgent("analysis_001", analysis_spawn, helix, llm_client,
                     token_budget_manager=token_budget_manager, max_tokens=800),
        SynthesisAgent("synthesis_001", synthesis_spawn, helix, llm_client,
                      token_budget_manager=token_budget_manager, max_tokens=800)
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
    synthesis_spawn = random.uniform(0.8, 0.95)

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
        CriticAgent("critic_001", critic_spawn, helix, llm_client, "accuracy", token_budget_manager, 800),
        SynthesisAgent("synthesis_001", synthesis_spawn, helix, llm_client, "general", token_budget_manager, 800)
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
    synthesis_spawns = [random.uniform(0.8, 0.98) for _ in range(2)]

    # Sort to maintain some ordering within types
    research_spawns.sort()
    analysis_spawns.sort()
    critic_spawns.sort()
    synthesis_spawns.sort()

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
        CriticAgent("critic_002", critic_spawns[1], helix, llm_client, "completeness", token_budget_manager, 800),
        SynthesisAgent("synthesis_001", synthesis_spawns[0], helix, llm_client, "report", token_budget_manager, 800),
        SynthesisAgent("synthesis_002", synthesis_spawns[1], helix, llm_client, "executive_summary", token_budget_manager, 800)
    ]
