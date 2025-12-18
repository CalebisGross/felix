"""
Unified Prompt Pipeline for Felix Framework

Single owner for all prompt construction with explicit stages and comprehensive logging.
Eliminates scattered prompt logic across multiple files.

Architecture Note:
- For SIMPLE_FACTUAL tasks, uses override prompts and stage skipping instead of full pipeline
- This prevents signal-to-noise degradation where agents receive contradictory instructions
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Reasoning Protocol for Multi-Step Agent Discovery
# This teaches agents to signal their reasoning state for intelligent iteration
REASONING_PROTOCOL = """
🧠 REASONING PROTOCOL (Multi-Step Discovery):

When performing tasks that require multiple steps (like file discovery → file reading → analysis),
signal your reasoning state so the system knows whether to continue:

REASONING_STATE: CONTINUE  → I need more information or need to execute another command
REASONING_STATE: COMPLETE  → I have everything needed, here is my final answer
REASONING_STATE: BLOCKED   → Task appears impossible (explain why)

Example workflow for file discovery:
1. "I need to find the file first.
   SYSTEM_ACTION_NEEDED: find . -name 'central_post.py' -type f
   REASONING_STATE: CONTINUE"

2. After receiving find result (./src/communication/central_post.py):
   "Found the file, now reading it.
   SYSTEM_ACTION_NEEDED: cat ./src/communication/central_post.py
   REASONING_STATE: CONTINUE"

3. After receiving file content:
   "Here is the analysis of central_post.py: [summary]
   REASONING_STATE: COMPLETE"

IMPORTANT:
- Always include REASONING_STATE at the end of your response
- If you need to execute a command, use SYSTEM_ACTION_NEEDED: followed by REASONING_STATE: CONTINUE
- Only use REASONING_STATE: COMPLETE when you have the final answer
- Use REASONING_STATE: BLOCKED if you encounter an error you can't resolve
"""


@dataclass
class PromptStageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    content: str
    tokens_estimate: int
    duration_ms: float
    metadata: Dict[str, Any]


@dataclass
class PromptBuildResult:
    """Complete result from pipeline execution."""
    system_prompt: str
    user_prompt: str
    total_tokens_estimate: int
    total_duration_ms: float
    stages: List[PromptStageResult]
    metadata: Dict[str, Any]


class PromptPipeline:
    """
    Unified prompt construction pipeline with explicit stages.

    Replaces scattered prompt logic with single, traceable flow:
    1. Load base prompt (PromptManager or fallback)
    2. Inject tool instructions (conditional based on task)
    3. Inject knowledge context (formatted from memory)
    4. Inject existing concepts (from concept registry)
    5. Apply Context Awareness Protocol (inventory + rules)
    6. Add collaborative context (previous agent outputs)
    7. Add metadata (tokens, temperature, position)
    8. Build user prompt (if needed)

    Each stage is logged with [PROMPT_STAGE] tags for debugging.
    """

    def __init__(self, prompt_manager: Optional['PromptManager'] = None,
                 enable_verbose_logging: bool = False):
        """
        Initialize prompt pipeline.

        Args:
            prompt_manager: Optional PromptManager for base prompt retrieval
            enable_verbose_logging: Enable detailed stage-by-stage logging
        """
        self.prompt_manager = prompt_manager
        self.enable_verbose_logging = enable_verbose_logging
        self.stages: List[PromptStageResult] = []

    def build_agent_prompt(self,
                          task: 'LLMTask',
                          agent: 'LLMAgent',
                          position_info: Dict[str, Any],
                          current_time: float) -> PromptBuildResult:
        """
        Build complete prompt for agent through explicit pipeline stages.

        Args:
            task: LLMTask with task description and enriched context
            agent: Agent instance requesting prompt
            position_info: Helix position information (depth_ratio, etc.)
            current_time: Current simulation time

        Returns:
            PromptBuildResult with system_prompt, user_prompt, and metadata
        """
        logger.info(f"🐛 DEBUG build_agent_prompt called: task type = {type(task)}, hasattr metadata = {hasattr(task, 'metadata')}")
        pipeline_start = time.time()
        self.stages = []

        if self.enable_verbose_logging:
            logger.info(f"[PROMPT_PIPELINE] Building prompt for {agent.agent_id}")
            logger.info(f"[PROMPT_PIPELINE] Agent type: {agent.agent_type}")
            logger.info(f"[PROMPT_PIPELINE] Position: depth={position_info.get('depth_ratio', 0):.2f}")

        logger.info(f"🐛 DEBUG about to call Stage 1 _stage_load_base_prompt")
        # Stage 1: Load base prompt
        base_prompt = self._stage_load_base_prompt(task, agent, position_info)
        logger.info(f"🐛 DEBUG Stage 1 complete")

        # Stage 2: Inject tool instructions (conditional)
        tools_section = self._stage_inject_tool_instructions(task, agent)

        # Stage 3: Inject knowledge context
        knowledge_section = self._stage_inject_knowledge_context(task, agent)

        # Stage 4: Inject existing concepts
        concepts_section = self._stage_inject_concepts(task, agent)

        # Stage 5: Apply Context Awareness Protocol
        protocol_section = self._stage_apply_context_protocol(task, agent)

        # Stage 6: Add collaborative context
        collaborative_section = self._stage_add_collaborative_context(task, agent)

        # Stage 6.5: Add verbosity constraints (Phase 3.3 & 3.4)
        verbosity_section = self._stage_add_verbosity_constraints(task, agent)

        # Stage 6.6: Add reasoning protocol for multi-step tasks
        reasoning_section = self._stage_inject_reasoning_protocol(task, agent)

        # Stage 7: Add metadata
        metadata_section = self._stage_add_metadata(task, agent, position_info, current_time)

        # Assemble final system prompt
        system_prompt = self._assemble_system_prompt(
            base_prompt=base_prompt,
            tools=tools_section,
            knowledge=knowledge_section,
            concepts=concepts_section,
            protocol=protocol_section,
            verbosity=verbosity_section,
            reasoning=reasoning_section,
            metadata=metadata_section
        )

        # Stage 8: Build user prompt (if collaborative context exists)
        user_prompt = self._build_user_prompt(task, collaborative_section)

        # Calculate final metrics
        pipeline_duration = (time.time() - pipeline_start) * 1000  # ms
        total_tokens = self._estimate_tokens(system_prompt + user_prompt)

        result = PromptBuildResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            total_tokens_estimate=total_tokens,
            total_duration_ms=pipeline_duration,
            stages=self.stages,
            metadata={
                'agent_id': agent.agent_id,
                'agent_type': agent.agent_type,
                'depth_ratio': position_info.get('depth_ratio', 0),
                'pipeline_version': '1.0'
            }
        )

        if self.enable_verbose_logging:
            logger.info(f"[PROMPT_PIPELINE] Complete! Total tokens: {total_tokens}, Duration: {pipeline_duration:.1f}ms")
            self._log_pipeline_summary(result)

        # Log prompt effectiveness metrics
        self._log_prompt_effectiveness(result, task, agent)

        return result

    def _should_skip_stage(self, stage_name: str, task: 'LLMTask') -> bool:
        """
        Determine if a pipeline stage should be skipped based on task complexity.

        For SIMPLE_FACTUAL tasks, skip irrelevant stages to reduce noise:
        - inject_concepts: No need for concept registry on simple file reads
        - apply_context_protocol: No need for protocol on simple operations
        - add_collaborative_context: No need for collaboration on simple queries

        Args:
            stage_name: Name of the stage to check
            task: LLMTask instance with complexity metadata

        Returns:
            True if stage should be skipped, False otherwise
        """
        # Check if task has complexity classification
        # Safe metadata access for both LLMTask objects and dict representations
        logger.debug(f"🐛 DEBUG _should_skip_stage: task type = {type(task)}, hasattr = {hasattr(task, 'metadata')}, isinstance dict = {isinstance(task, dict)}")
        if hasattr(task, 'metadata') and task.metadata:
            task_complexity = task.metadata.get('complexity', 'COMPLEX')
        elif isinstance(task, dict) and 'metadata' in task:
            task_complexity = task['metadata'].get('complexity', 'COMPLEX')
        else:
            task_complexity = 'COMPLEX'

        # For SIMPLE_FACTUAL tasks, skip collaborative/conceptual overhead
        if task_complexity == "SIMPLE_FACTUAL":
            skippable_stages = {
                'inject_concepts',           # No need for concept registry
                'apply_context_protocol',    # No need for awareness protocol
                'add_collaborative_context'  # No need for collaboration on simple reads
            }
            if stage_name in skippable_stages:
                logger.debug(f"[STAGE_SKIP] Skipping '{stage_name}' for SIMPLE_FACTUAL task")
                return True

        return False

    def _apply_priority_ordering(self, prompts_available: Dict[str, Any],
                                task_complexity: Optional[str],
                                agent_type: str) -> tuple[str, str]:
        """
        Apply priority ordering to resolve prompt conflicts.

        Priority order (highest to lowest):
        1. Minimal mode prompt (if task is SIMPLE_FACTUAL and minimal prompt available)
        2. Task complexity override (e.g., simple_file_reading for SIMPLE_FACTUAL)
        3. Position-based prompt (depth on helix)
        4. Agent fallback (hardcoded minimal prompt)

        Args:
            prompts_available: Dict with available prompt sources and their content
            task_complexity: Task complexity classification (SIMPLE_FACTUAL, MEDIUM, COMPLEX)
            agent_type: Type of agent (research, analysis, critic)

        Returns:
            Tuple of (selected_prompt, source_name)
        """
        # Priority 1: Minimal mode (handled earlier in pipeline, this is for conflict resolution)
        if prompts_available.get('minimal_mode'):
            logger.debug("[PRIORITY] Selected minimal mode prompt (highest priority)")
            return prompts_available['minimal_mode'], 'minimal_mode'

        # Priority 2: Task complexity override
        if task_complexity == "SIMPLE_FACTUAL" and prompts_available.get('complexity_override'):
            logger.debug("[PRIORITY] Selected complexity override (priority 2)")
            return prompts_available['complexity_override'], 'complexity_override'

        # Priority 3: Position-based prompt
        if prompts_available.get('position_based'):
            logger.debug("[PRIORITY] Selected position-based prompt (priority 3)")
            return prompts_available['position_based'], 'position_based'

        # Priority 4: Agent fallback
        if prompts_available.get('agent_fallback'):
            logger.debug("[PRIORITY] Selected agent fallback (lowest priority)")
            return prompts_available['agent_fallback'], 'agent_fallback'

        # Should never reach here, but provide safety fallback
        logger.warning("[PRIORITY] No prompts available - using emergency fallback")
        return f"You are a {agent_type} agent. Provide concise analysis.", 'emergency_fallback'

    def _extract_agent_traits(self, agent: 'LLMAgent', position_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract agent personality traits for template rendering.

        These traits customize agent prompts with their specialized focus areas.
        Falls back to 'general' if trait not set on agent.

        Args:
            agent: Agent instance with potential trait attributes
            position_info: Helix position information

        Returns:
            Dictionary of trait values for template substitution
        """
        traits = {
            'agent_type': agent.agent_type,
            'depth_ratio': position_info.get('depth_ratio', 0.0),
        }
        # Agent-specific personality traits (set during spawning)
        traits['research_domain'] = getattr(agent, 'research_domain', None) or 'general'
        traits['analysis_type'] = getattr(agent, 'analysis_type', None) or 'general'
        traits['review_focus'] = getattr(agent, 'review_focus', None) or 'general'
        return traits

    def _stage_load_base_prompt(self, task: 'LLMTask', agent: 'LLMAgent',
                                position_info: Dict[str, Any]) -> str:
        """
        Stage 1: Load base prompt from PromptManager or use fallback.

        Checks task complexity metadata and uses override prompts for simple tasks.

        Returns:
            Base prompt string
        """
        logger.info(f"🐛 DEBUG Stage 1 _stage_load_base_prompt: task type = {type(task)}")
        stage_start = time.time()
        stage_name = "load_base_prompt"

        depth_ratio = position_info.get('depth_ratio', 0)

        # Check if task has complexity classification
        # Safe metadata access for both LLMTask objects and dict representations
        logger.info(f"🐛 DEBUG Stage 1 about to access task.metadata: hasattr = {hasattr(task, 'metadata')}")
        if hasattr(task, 'metadata') and task.metadata:
            task_complexity = task.metadata.get('complexity', None)
        elif isinstance(task, dict) and 'metadata' in task:
            task_complexity = task['metadata'].get('complexity', None)
        else:
            task_complexity = None

        # PRIORITY ORDER: Task complexity > Depth-based selection
        # This ensures simple tasks get simple prompts regardless of helix position
        base_prompt = ""
        prompt_key = None
        if self.prompt_manager and agent.prompt_manager:
            try:
                # FIRST: Check for complexity-based override prompts (highest priority)
                if task_complexity == "SIMPLE_FACTUAL" and agent.agent_type in ['research', 'analysis', 'critic']:
                    # Try to load simple_file_reading override
                    override_key = f"{agent.agent_type}_simple_file_reading"
                    try:
                        override_template = self.prompt_manager.get_prompt(override_key)
                        if override_template and override_template.template:
                            base_prompt = override_template.template
                            prompt_key = override_key
                            logger.info(f"[PROMPT_STAGE:{stage_name}] Using SIMPLE_FACTUAL override: {override_key}")
                    except Exception:
                        pass  # Override not found, fall through to depth-based selection

                # SECOND: If no complexity override, use depth-based prompt (fallback)
                if not base_prompt:
                    prompt_key = self._determine_prompt_key(agent.agent_type, depth_ratio)
                    prompt_template = self.prompt_manager.get_prompt(prompt_key)
                    base_prompt = prompt_template.template if prompt_template else ""

                if base_prompt:
                    logger.debug(f"[PROMPT_STAGE:{stage_name}] Loaded from PromptManager: {prompt_key}")
                else:
                    logger.warning(f"[PROMPT_STAGE:{stage_name}] No prompt found for key: {prompt_key}")
            except Exception as e:
                logger.warning(f"[PROMPT_STAGE:{stage_name}] PromptManager failed: {e}")

        # Use agent's built-in fallback if PromptManager failed
        if not base_prompt:
            base_prompt = self._get_agent_fallback_prompt(agent, depth_ratio, task_complexity)
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Using agent fallback prompt (complexity: {task_complexity})")

        # RENDER TEMPLATE: Substitute agent personality traits
        # This replaces {research_domain}, {analysis_type}, {review_focus} with actual values
        if base_prompt and self.prompt_manager:
            traits = self._extract_agent_traits(agent, position_info)
            base_prompt = self.prompt_manager.render_template(base_prompt, **traits)
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Rendered traits: {list(traits.keys())}")

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=base_prompt[:100] + "..." if len(base_prompt) > 100 else base_prompt,
            tokens_estimate=self._estimate_tokens(base_prompt),
            duration_ms=stage_duration,
            metadata={'source': 'prompt_manager' if self.prompt_manager else 'fallback'}
        ))

        return base_prompt

    def _stage_inject_tool_instructions(self, task: 'LLMTask', agent: 'LLMAgent') -> str:
        """
        Stage 2: Inject tool instructions (conditional based on task requirements).

        Returns:
            Tool instructions section (empty if not needed)
        """
        stage_start = time.time()
        stage_name = "inject_tool_instructions"

        tools_section = ""
        source = "none"

        if hasattr(task, 'tool_instructions') and task.tool_instructions:
            tools_section = task.tool_instructions
            source = "task_context"
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Using task.tool_instructions ({len(tools_section)} chars)")
        else:
            # Use imperative execution directive if no instructions provided
            from src.agents.specialized_agents import EXECUTION_DIRECTIVE
            tools_section = EXECUTION_DIRECTIVE
            source = "execution_directive"
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Using imperative execution directive")

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=tools_section[:100] + "..." if len(tools_section) > 100 else tools_section,
            tokens_estimate=self._estimate_tokens(tools_section),
            duration_ms=stage_duration,
            metadata={'source': source, 'has_instructions': bool(tools_section)}
        ))

        return tools_section

    def _stage_inject_knowledge_context(self, task: 'LLMTask', agent: 'LLMAgent') -> str:
        """
        Stage 3: Inject knowledge context from memory.

        Returns:
            Knowledge context section (empty if no knowledge)
        """
        stage_start = time.time()
        stage_name = "inject_knowledge_context"

        # Use shared knowledge formatter from LLMAgent
        from src.agents.llm_agent import LLMAgent
        knowledge_section = LLMAgent.format_knowledge_summary(
            task,
            include_relevance_guidance=(agent.agent_type == "research")
        )

        entry_count = len(task.knowledge_entries) if hasattr(task, 'knowledge_entries') and task.knowledge_entries else 0

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=knowledge_section[:100] + "..." if len(knowledge_section) > 100 else knowledge_section,
            tokens_estimate=self._estimate_tokens(knowledge_section),
            duration_ms=stage_duration,
            metadata={'entry_count': entry_count, 'has_knowledge': bool(knowledge_section)}
        ))

        logger.debug(f"[PROMPT_STAGE:{stage_name}] Injected {entry_count} knowledge entries")

        return knowledge_section

    def _stage_inject_concepts(self, task: 'LLMTask', agent: 'LLMAgent') -> str:
        """
        Stage 4: Inject existing concept definitions for terminology consistency.

        Returns:
            Concepts section (empty if no concepts)
        """
        stage_start = time.time()
        stage_name = "inject_concepts"

        # Skip this stage for simple tasks
        if self._should_skip_stage(stage_name, task):
            self.stages.append(PromptStageResult(
                stage_name=stage_name,
                content="[SKIPPED for SIMPLE_FACTUAL]",
                tokens_estimate=0,
                duration_ms=(time.time() - stage_start) * 1000,
                metadata={'skipped': True}
            ))
            return ""

        concepts_section = ""
        has_concepts = False

        if hasattr(task, 'existing_concepts') and task.existing_concepts:
            if task.existing_concepts != "No concepts defined yet in this workflow.":
                concepts_section = "\n\n=== Existing Concept Definitions ===\n"
                concepts_section += "The following concepts have already been defined by other agents in this workflow.\n"
                concepts_section += "Please use these definitions consistently instead of redefining them:\n\n"
                concepts_section += task.existing_concepts
                concepts_section += "\n\nIMPORTANT: Reference these existing concepts when relevant. "
                concepts_section += "Only define new concepts if they are not already covered above.\n"
                has_concepts = True
                logger.debug(f"[PROMPT_STAGE:{stage_name}] Injected concept definitions")

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=concepts_section[:100] + "..." if len(concepts_section) > 100 else concepts_section,
            tokens_estimate=self._estimate_tokens(concepts_section),
            duration_ms=stage_duration,
            metadata={'has_concepts': has_concepts}
        ))

        return concepts_section

    def _stage_apply_context_protocol(self, task: 'LLMTask', agent: 'LLMAgent') -> str:
        """
        Stage 5: Apply Context Awareness Protocol (inventory + strict rules + response format).

        Returns:
            Protocol section
        """
        stage_start = time.time()
        stage_name = "apply_context_protocol"

        # Skip this stage for simple tasks
        if self._should_skip_stage(stage_name, task):
            self.stages.append(PromptStageResult(
                stage_name=stage_name,
                content="[SKIPPED for SIMPLE_FACTUAL]",
                tokens_estimate=0,
                duration_ms=(time.time() - stage_start) * 1000,
                metadata={'skipped': True}
            ))
            return ""

        protocol_section = ""

        # Add context inventory if available
        if hasattr(task, 'context_inventory') and task.context_inventory:
            protocol_section += task.context_inventory
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Added context inventory")

        # Add strict rules (use agent's method)
        if hasattr(agent, '_build_strict_rules'):
            strict_rules = agent._build_strict_rules(task)
            if strict_rules:
                protocol_section += "\n\n🎯 STRICT RULES:\n"
                for i, rule in enumerate(strict_rules, 1):
                    protocol_section += f"{i}. {rule}\n"
                logger.debug(f"[PROMPT_STAGE:{stage_name}] Added {len(strict_rules)} strict rules")

        # Add response format (use agent's method)
        if hasattr(agent, '_build_response_format'):
            response_format = agent._build_response_format()
            protocol_section += "\n" + response_format
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Added response format")

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=protocol_section[:100] + "..." if len(protocol_section) > 100 else protocol_section,
            tokens_estimate=self._estimate_tokens(protocol_section),
            duration_ms=stage_duration,
            metadata={'has_protocol': bool(protocol_section)}
        ))

        return protocol_section

    def _stage_add_collaborative_context(self, task: 'LLMTask', agent: 'LLMAgent') -> str:
        """
        Stage 6: Add collaborative context from previous agents.

        Returns:
            Collaborative context section (for user prompt, not system prompt)
        """
        stage_start = time.time()
        stage_name = "add_collaborative_context"

        # Skip this stage for simple tasks
        if self._should_skip_stage(stage_name, task):
            self.stages.append(PromptStageResult(
                stage_name=stage_name,
                content="[SKIPPED for SIMPLE_FACTUAL]",
                tokens_estimate=0,
                duration_ms=(time.time() - stage_start) * 1000,
                metadata={'skipped': True}
            ))
            return ""

        collaborative_section = ""
        message_count = 0

        if hasattr(task, 'context_history') and task.context_history:
            message_count = len(task.context_history)
            # This will be used in user prompt, not system prompt
            collaborative_section = self._format_collaborative_context(task.context_history, agent.agent_type)
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Formatted {message_count} previous outputs")

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=collaborative_section[:100] + "..." if len(collaborative_section) > 100 else collaborative_section,
            tokens_estimate=self._estimate_tokens(collaborative_section),
            duration_ms=stage_duration,
            metadata={'message_count': message_count, 'has_collaboration': bool(collaborative_section)}
        ))

        return collaborative_section

    def _stage_add_metadata(self, task: 'LLMTask', agent: 'LLMAgent',
                           position_info: Dict[str, Any], current_time: float) -> str:
        """
        Stage 7: Add processing metadata (tokens, temperature, position).

        Returns:
            Metadata section
        """
        stage_start = time.time()
        stage_name = "add_metadata"

        depth_ratio = position_info.get('depth_ratio', 0)
        temperature = agent.get_adaptive_temperature(current_time)

        metadata_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Processing Parameters:
- Agent: {agent.agent_id} ({agent.agent_type})
- Helix Position: depth {depth_ratio:.2f}/1.0
- Temperature: {temperature:.2f}
- Current Time: {current_time:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=metadata_section,
            tokens_estimate=self._estimate_tokens(metadata_section),
            duration_ms=stage_duration,
            metadata={
                'depth_ratio': depth_ratio,
                'temperature': temperature
            }
        ))

        logger.debug(f"[PROMPT_STAGE:{stage_name}] Added metadata")

        return metadata_section

    def _stage_add_verbosity_constraints(self, task: 'LLMTask', agent: 'LLMAgent') -> str:
        """
        Stage 6.5: Add verbosity constraints and output format requirements.

        Injects constraints based on task complexity:
        - SIMPLE_FACTUAL: Maximum 3 sentences
        - MEDIUM: 2-3 paragraphs (~150-200 words)
        - COMPLEX: Step-by-step format, 5 steps max, 2-3 sentences per step

        Phase 3.3: Force task decomposition for complex tasks
        Phase 3.4: Add output verbosity constraints by task complexity

        Args:
            task: LLMTask with complexity metadata
            agent: Agent instance

        Returns:
            Formatted verbosity constraints section
        """
        stage_start = time.time()
        stage_name = "add_verbosity_constraints"

        # Skip if stage should be skipped
        if self._should_skip_stage(stage_name, task):
            return ""

        # Get task complexity
        # Safe metadata access for both LLMTask objects and dict representations
        if hasattr(task, 'metadata') and task.metadata:
            task_complexity = task.metadata.get('complexity', 'COMPLEX')
        elif isinstance(task, dict) and 'metadata' in task:
            task_complexity = task['metadata'].get('complexity', 'COMPLEX')
        else:
            task_complexity = 'COMPLEX'

        logger.debug(f"[PROMPT_STAGE:{stage_name}] Building verbosity constraints for {task_complexity} task")

        # Build constraints based on complexity
        verbosity_section = "\n\n"

        if task_complexity == "SIMPLE_FACTUAL":
            verbosity_section += "🎯 OUTPUT CONSTRAINT:\n"
            verbosity_section += "- Maximum 3 sentences\n"
            verbosity_section += "- Be direct and concise\n"
            verbosity_section += "- No philosophical tangents or excessive explanation\n"
            verbosity_section += "- Answer the question, nothing more\n"

        elif task_complexity == "MEDIUM":
            verbosity_section += "🎯 OUTPUT CONSTRAINT:\n"
            verbosity_section += "- 2-3 paragraphs maximum (~150-200 words)\n"
            verbosity_section += "- Be clear and focused\n"
            verbosity_section += "- Avoid redundant explanations\n"
            verbosity_section += "- Provide sufficient detail without rambling\n"

        else:  # COMPLEX
            verbosity_section += "🎯 STRUCTURED OUTPUT REQUIRED:\n"
            verbosity_section += "Complex task detected. Structure your response as:\n\n"
            verbosity_section += "**Step 1: [Action]**\n"
            verbosity_section += "[Brief execution of step 1 in 2-3 sentences]\n\n"
            verbosity_section += "**Step 2: [Action]**\n"
            verbosity_section += "[Brief execution of step 2 in 2-3 sentences]\n\n"
            verbosity_section += "**Step 3: [Action]**\n"
            verbosity_section += "[Brief execution of step 3 in 2-3 sentences]\n\n"
            verbosity_section += "CONSTRAINTS:\n"
            verbosity_section += "- Maximum 5 steps total\n"
            verbosity_section += "- Each step: 2-3 sentences only\n"
            verbosity_section += "- Total response: ~200-300 words maximum\n"
            verbosity_section += "- NO philosophical tangents or excessive background\n"
            verbosity_section += "- Focus on ACTIONABLE steps and CONCRETE results\n"

        # Record stage metrics
        stage_duration = (time.time() - stage_start) * 1000  # ms
        tokens = self._estimate_tokens(verbosity_section)

        self.stages.append({
            'name': stage_name,
            'duration_ms': stage_duration,
            'tokens': tokens,
            'skipped': False,
            'complexity': task_complexity
        })

        logger.debug(f"[PROMPT_STAGE:{stage_name}] Added {tokens} tokens of verbosity constraints ({task_complexity})")

        return verbosity_section

    def _stage_inject_reasoning_protocol(self, task: 'LLMTask', agent: 'LLMAgent') -> str:
        """
        Stage 6.6: Inject reasoning protocol for multi-step tasks.

        This teaches agents to signal their reasoning state (CONTINUE/COMPLETE/BLOCKED)
        for intelligent multi-step iteration. Only injected for tasks that need file ops
        or system commands.

        Args:
            task: LLMTask with tool requirements metadata
            agent: Agent instance

        Returns:
            Reasoning protocol section (empty if not needed)
        """
        stage_start = time.time()
        stage_name = "inject_reasoning_protocol"

        reasoning_section = ""

        # Check if task has file operations or system commands requirements
        needs_reasoning = False
        if hasattr(task, 'metadata') and task.metadata:
            # Check if task involves file operations
            needs_file_ops = task.metadata.get('needs_file_ops', False)
            needs_system = task.metadata.get('needs_system_commands', False)

            # Also inject for COMPLEX tasks that might need multi-step reasoning
            task_complexity = task.metadata.get('complexity', 'MEDIUM')
            needs_reasoning = needs_file_ops or needs_system or task_complexity == 'COMPLEX'

        # Also check tool_instructions presence - if tools are provided, likely needs reasoning
        if hasattr(task, 'tool_instructions') and task.tool_instructions:
            # If file ops instructions are present, enable reasoning
            if 'SYSTEM_ACTION_NEEDED' in task.tool_instructions:
                needs_reasoning = True

        if needs_reasoning:
            reasoning_section = REASONING_PROTOCOL
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Injected reasoning protocol for multi-step task")
        else:
            logger.debug(f"[PROMPT_STAGE:{stage_name}] Skipped reasoning protocol (simple task)")

        stage_duration = (time.time() - stage_start) * 1000
        self.stages.append(PromptStageResult(
            stage_name=stage_name,
            content=reasoning_section[:100] + "..." if len(reasoning_section) > 100 else reasoning_section,
            tokens_estimate=self._estimate_tokens(reasoning_section),
            duration_ms=stage_duration,
            metadata={'needs_reasoning': needs_reasoning}
        ))

        return reasoning_section

    def _assemble_system_prompt(self, base_prompt: str, tools: str, knowledge: str,
                                concepts: str, protocol: str, verbosity: str,
                                reasoning: str, metadata: str) -> str:
        """
        Assemble final system prompt from all sections.

        Args:
            base_prompt: Base prompt template
            tools: Tool instructions section
            knowledge: Knowledge context section
            concepts: Concept definitions section
            protocol: Context awareness protocol section
            verbosity: Verbosity constraints and format requirements
            reasoning: Reasoning protocol for multi-step tasks
            metadata: Processing metadata section

        Returns:
            Complete system prompt
        """
        # Assemble in logical order
        system_prompt = tools  # Tools first (critical instructions)
        system_prompt += base_prompt  # Base prompt (agent role and instructions)
        system_prompt += knowledge  # Knowledge context
        system_prompt += concepts  # Concept definitions
        system_prompt += protocol  # Context awareness protocol
        system_prompt += verbosity  # Verbosity constraints (Phase 3.3 & 3.4)
        system_prompt += reasoning  # Reasoning protocol for multi-step discovery
        system_prompt += metadata  # Processing metadata

        return system_prompt

    def _build_user_prompt(self, task: 'LLMTask', collaborative_section: str) -> str:
        """
        Build user prompt with task description and collaborative context.

        Args:
            task: LLMTask with task description
            collaborative_section: Formatted collaborative context

        Returns:
            User prompt string
        """
        user_prompt = f"Task: {task.description}"

        if collaborative_section:
            user_prompt += "\n\n" + collaborative_section

        return user_prompt

    def _format_collaborative_context(self, context_history: List[Dict[str, Any]],
                                     agent_type: str) -> str:
        """
        Format collaborative context from previous agent outputs.

        Args:
            context_history: List of previous agent messages
            agent_type: Current agent type

        Returns:
            Formatted collaborative context
        """
        if not context_history:
            return ""

        formatted = "Previous Agent Outputs:\n"
        formatted += f"({len(context_history)} agent(s) have contributed - build upon their work)\n\n"

        for i, msg in enumerate(context_history, 1):
            sender = msg.get('agent_id', 'unknown')
            content = msg.get('content', '')
            formatted += f"--- Output {i} (from {sender}) ---\n{content}\n\n"

        return formatted

    def _determine_prompt_key(self, agent_type: str, depth_ratio: float, strict_mode: bool = False) -> str:
        """
        Determine PromptManager key based on agent type and depth.

        Maps to YAML config keys:
        - research: exploration/focused/deep + normal/strict
        - analysis: mid_phase/late_phase + normal/strict
        - critic: base_header (no depth variants)

        Args:
            agent_type: Agent type (research, analysis, critic, etc.)
            depth_ratio: Helix depth ratio (0.0-1.0)
            strict_mode: Whether strict token budget is active

        Returns:
            Prompt key string matching config/prompts.yaml structure
        """
        mode_suffix = "strict" if strict_mode else "normal"

        if agent_type == "research":
            # Research: exploration (0.0-0.3), focused (0.3-0.7), deep (0.7-1.0)
            if depth_ratio < 0.3:
                return f"research_exploration_{mode_suffix}"
            elif depth_ratio < 0.7:
                return f"research_focused_{mode_suffix}"
            else:
                return f"research_deep_{mode_suffix}"

        elif agent_type == "analysis":
            # Analysis: mid_phase (0.0-0.5), late_phase (0.5-1.0)
            if depth_ratio < 0.5:
                return f"analysis_mid_phase_{mode_suffix}"
            else:
                return f"analysis_late_phase_{mode_suffix}"

        elif agent_type == "critic":
            # Critic has no depth variants in YAML, use base_header
            return "critic_base_header"

        else:
            # Fallback for other agent types
            return f"{agent_type}_base"

    def _get_agent_fallback_prompt(self, agent: 'LLMAgent', depth_ratio: float,
                                   task_complexity: Optional[str] = None) -> str:
        """
        Get minimal fallback prompt when PromptManager fails.

        This should rarely be used - prompts should come from config/prompts.yaml.
        Respects task complexity to ensure simple tasks get simple instructions.

        Args:
            agent: Agent instance
            depth_ratio: Helix depth ratio
            task_complexity: Optional task complexity (SIMPLE_FACTUAL, MEDIUM, COMPLEX)

        Returns:
            Minimal fallback prompt string
        """
        # Minimal fallback - log warning since this indicates config issue
        logger.warning(f"Using minimal fallback prompt for {agent.agent_type} agent - check config/prompts.yaml")

        # PRIORITY: Task complexity overrides depth-based instructions
        if task_complexity == "SIMPLE_FACTUAL":
            return f"""You are a {agent.agent_type.upper()} agent.

Task: Provide a DIRECT, CONCISE answer to the user's question.

Output Requirements:
- Answer in 2-3 sentences (50-100 words maximum)
- NO philosophical analysis or elaborate discussion
- Stick to factual information only

Current depth: {depth_ratio:.2f}/1.0"""
        else:
            # Default fallback for MEDIUM/COMPLEX or unclassified tasks
            return f"""You are a {agent.agent_type.upper()} agent in the Felix multi-agent system.

Your role: Process the task with your specialized perspective.

Current depth: {depth_ratio:.2f}/1.0 on the helix.

Provide your analysis clearly and concisely."""

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count using 4 chars ≈ 1 token heuristic.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _log_pipeline_summary(self, result: PromptBuildResult) -> None:
        """
        Log summary of pipeline execution.

        Args:
            result: Complete pipeline result
        """
        logger.info(f"[PROMPT_PIPELINE] ━━━━━━ Pipeline Summary ━━━━━━")
        logger.info(f"[PROMPT_PIPELINE] Agent: {result.metadata['agent_id']}")
        logger.info(f"[PROMPT_PIPELINE] Total tokens: {result.total_tokens_estimate}")
        logger.info(f"[PROMPT_PIPELINE] Total duration: {result.total_duration_ms:.1f}ms")
        logger.info(f"[PROMPT_PIPELINE] Stages executed:")

        for stage in result.stages:
            logger.info(f"[PROMPT_PIPELINE]   - {stage.stage_name}: "
                       f"{stage.tokens_estimate} tokens, {stage.duration_ms:.1f}ms")

        logger.info(f"[PROMPT_PIPELINE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _log_prompt_effectiveness(self, result: PromptBuildResult,
                                  task: 'LLMTask', agent: 'LLMAgent') -> None:
        """
        Log prompt effectiveness metrics for analysis and optimization.

        Tracks:
        - Prompt mode (minimal vs full pipeline)
        - Token efficiency (tokens per stage)
        - Task complexity alignment
        - Stage skipping effectiveness

        Args:
            result: Complete pipeline result
            task: Original task
            agent: Agent that received the prompt
        """
        # Extract key metrics
        is_minimal = result.metadata.get('minimal_mode', False)
        # Safe metadata access for both LLMTask objects and dict representations
        if hasattr(task, 'metadata') and task.metadata:
            task_complexity = task.metadata.get('complexity', 'UNKNOWN')
        elif isinstance(task, dict) and 'metadata' in task:
            task_complexity = task['metadata'].get('complexity', 'UNKNOWN')
        else:
            task_complexity = 'UNKNOWN'
        total_tokens = result.total_tokens_estimate
        stages_executed = len(result.stages)
        # Handle both PromptStageResult objects and dicts in stages list
        skipped_stages = sum(1 for s in result.stages if (
            s.metadata.get('skipped', False) if hasattr(s, 'metadata') else s.get('skipped', False)
        ))

        # Calculate efficiency metrics
        avg_tokens_per_stage = total_tokens / stages_executed if stages_executed > 0 else 0
        token_efficiency_ratio = 1.0 - (skipped_stages / max(stages_executed, 1))

        # Log effectiveness metrics
        logger.info(f"[PROMPT_EFFECTIVENESS] ━━━━━━ Effectiveness Metrics ━━━━━━")
        logger.info(f"[PROMPT_EFFECTIVENESS] Agent: {agent.agent_id} ({agent.agent_type})")
        logger.info(f"[PROMPT_EFFECTIVENESS] Task complexity: {task_complexity}")
        logger.info(f"[PROMPT_EFFECTIVENESS] Prompt mode: {'MINIMAL' if is_minimal else 'FULL_PIPELINE'}")
        logger.info(f"[PROMPT_EFFECTIVENESS] Total tokens: {total_tokens}")
        logger.info(f"[PROMPT_EFFECTIVENESS] Stages: {stages_executed} executed, {skipped_stages} skipped")
        logger.info(f"[PROMPT_EFFECTIVENESS] Avg tokens/stage: {avg_tokens_per_stage:.0f}")
        logger.info(f"[PROMPT_EFFECTIVENESS] Efficiency ratio: {token_efficiency_ratio:.2f}")

        # Validation checks
        if task_complexity == "SIMPLE_FACTUAL":
            if is_minimal:
                logger.info(f"[PROMPT_EFFECTIVENESS] ✅ OPTIMAL: Minimal prompt for simple task")
            elif total_tokens < 500:
                logger.info(f"[PROMPT_EFFECTIVENESS] ✅ GOOD: Compact prompt for simple task")
            else:
                logger.warning(f"[PROMPT_EFFECTIVENESS] ⚠️  INEFFICIENT: {total_tokens} tokens for simple task")

        elif task_complexity in ["MEDIUM", "COMPLEX"]:
            if total_tokens < 200:
                logger.warning(f"[PROMPT_EFFECTIVENESS] ⚠️  POSSIBLY TOO SHORT: {total_tokens} tokens for {task_complexity} task")
            else:
                logger.info(f"[PROMPT_EFFECTIVENESS] ✅ APPROPRIATE: {total_tokens} tokens for {task_complexity} task")

        # Log signal-to-noise ratio estimate
        if not is_minimal and stages_executed > 0:
            # Estimate: base prompt is ~30% signal, rest is context
            estimated_signal_tokens = result.stages[0].tokens_estimate if result.stages else 0
            signal_ratio = estimated_signal_tokens / max(total_tokens, 1) * 100
            logger.info(f"[PROMPT_EFFECTIVENESS] Signal-to-noise: {signal_ratio:.1f}% (estimated)")

        logger.info(f"[PROMPT_EFFECTIVENESS] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
