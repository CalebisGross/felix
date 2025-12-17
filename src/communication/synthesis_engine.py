"""
Synthesis Engine for the Felix Framework.

Handles final output synthesis from agent communications, implementing
the convergence point where all helical agent trajectories meet.

Key Features:
- Adaptive synthesis parameters based on agent confidence
- Task complexity-aware synthesis (SIMPLE/MEDIUM/COMPLEX)
- Integration of agent outputs AND system command results
- Temperature and token budget optimization
- Helical-aware synthesis prompting
- CentralPost represents the central axis convergence point

This module was extracted from CentralPost to improve separation of concerns
and maintainability while preserving all functionality.
"""

import time
import re
import os
import json
import sqlite3
import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from collections import namedtuple

# Set up logging first
logger = logging.getLogger(__name__)

# Import message types
from src.communication.message_types import Message, MessageType

# Import validation functions
from src.workflows.truth_assessment import calculate_validation_score, get_validation_flags


class SynthesisEngine:
    """
    Manages synthesis of final outputs from agent communications.

    Responsibilities:
    - Calculate adaptive synthesis parameters
    - Build synthesis prompts with helical context
    - Integrate agent outputs and system results
    - Generate final synthesized output via LLM
    """

    def __init__(self,
                 llm_client: Any,
                 get_recent_messages_callback: Any,
                 prompt_manager: Optional['PromptManager'] = None):
        """
        Initialize Synthesis Engine.

        Args:
            llm_client: LLM client for synthesis generation
            get_recent_messages_callback: Callback to get recent messages from CentralPost
            prompt_manager: Optional prompt manager for synthesis prompts
        """
        self.llm_client = llm_client
        self._get_recent_messages = get_recent_messages_callback
        self.prompt_manager = prompt_manager

        # Load classification patterns from YAML
        self._load_patterns()

        logger.info("✓ SynthesisEngine initialized with pattern-based classification and PromptManager integration")

    def _load_patterns(self):
        """
        Load classification patterns from YAML configuration files.

        Falls back to hardcoded patterns if YAML files are missing or invalid.
        Patterns are loaded once at initialization and cached for performance.
        """
        # Get project root (3 levels up from this file: src/communication/ -> src/ -> root/)
        project_root = Path(__file__).parent.parent.parent
        complexity_config = project_root / "config" / "task_complexity_patterns.yaml"
        tool_config = project_root / "config" / "tool_requirements_patterns.yaml"

        # Load task complexity patterns
        try:
            with open(complexity_config, 'r') as f:
                complexity_data = yaml.safe_load(f)
                self.simple_factual_patterns = [p['pattern'] for p in complexity_data.get('simple_factual', [])]
                self.medium_patterns = [p['pattern'] for p in complexity_data.get('medium', [])]
                logger.info(f"✓ Loaded {len(self.simple_factual_patterns)} simple_factual + {len(self.medium_patterns)} medium patterns from YAML")
        except Exception as e:
            logger.warning(f"Failed to load task complexity patterns from YAML: {e}. Using hardcoded fallback.")
            # Hardcoded fallback patterns
            self.simple_factual_patterns = [
                r'\b(what|when|who|where)\s+(is|are|was|were)\s+(the\s+)?current',
                r'\bwhat\s+time\b',
                r'\bwhat\s+date\b',
                r'\btoday\'?s?\s+(date|time)',
                r'\bcurrent\s+(time|date|datetime)',
                r'\bwho\s+(won|is|was)\b',
                r'\bwhen\s+(did|is|was)\b',
                r'\bhow\s+many\b.*\b(now|current|today)',
                r'\blatest\s+(news|update)\b',
                r'^\s*(hello|hi|hey|greetings?|howdy|yo)\s*[!.?]*\s*$',
                r'^\s*good\s+(morning|afternoon|evening|night|day)\s*[!.?]*\s*$',
            ]
            self.medium_patterns = [
                r'\bexplain\b',
                r'\bcompare\b',
                r'\bwhat\s+are\s+the\s+(benefits|advantages|disadvantages)',
                r'\bhow\s+does\b',
                r'\bhow\s+to\b',
                r'\blist\b',
                r'\bsummarize\b',
            ]

        # Load tool requirements patterns
        try:
            with open(tool_config, 'r') as f:
                tool_data = yaml.safe_load(f)
                self.file_operation_patterns = [p['pattern'] for p in tool_data.get('file_operations', [])]
                self.web_search_patterns = [p['pattern'] for p in tool_data.get('web_search', [])]
                self.system_command_patterns = [p['pattern'] for p in tool_data.get('system_commands', [])]
                logger.info(f"✓ Loaded {len(self.file_operation_patterns)} file_ops + {len(self.web_search_patterns)} web_search + {len(self.system_command_patterns)} system_cmd patterns from YAML")
        except Exception as e:
            logger.warning(f"Failed to load tool requirement patterns from YAML: {e}. Using hardcoded fallback.")
            # Hardcoded fallback patterns (write-only, no read patterns - old behavior)
            self.file_operation_patterns = [
                r'\b(create|write|save|generate|export|output)\s+(a\s+)?file',
                r'\bsave\s+(to|in|as)\b',
                r'\bwrite\s+(to|into)\b',
                r'\b(create|make|generate)\s+(a\s+)?(document|report|output|result)',
                r'\bfile\s+(with|containing)',
                r'\bgenerate.*\.(txt|md|json|csv|py|js)',
                r'\bexport\s+(to|as)\b',
            ]
            self.web_search_patterns = [
                r'\b(current|latest|recent|now|today|this\s+(week|month|year))',
                r'\bwhat\'?s?\s+(happening|new|the\s+news)',
                r'\b(search|find|look\s+up|google)\b',
                r'\bup\s+to\s+date\b',
                r'\blive\s+(data|information)',
                r'\breal[-\s]?time\b',
                r'\blatest\s+(update|version|news)',
            ]
            self.system_command_patterns = [
                r'\b(install|uninstall|upgrade)\s+(package|library|module|dependency)',
                r'\bmkdir\b',
                r'\bchmod\b',
                r'\b(run|execute)\s+(command|script|process)',
                r'\b(start|stop|restart)\s+(process|service)',
                r'\bcheck\s+(if|whether).*installed',
                r'\bpip\s+(install|list)',
                r'\bnpm\s+(install|run)',
                r'\bapt[-\s]get\b',
            ]

    def _log_synthesis_audit(
        self,
        workflow_id: Optional[str],
        task_description: str,
        task_complexity: str,
        messages: List[Message],
        system_prompt: str,
        user_prompt: str,
        result: Dict[str, Any],
        validation_score: Optional[float] = None,
        validation_flags: Optional[List[str]] = None,
        reasoning_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log full synthesis audit trail to database.

        Records all synthesis decisions with full context for auditability,
        including prompts, per-agent contributions, confidence calculations,
        and validation results.

        Args:
            workflow_id: ID of the workflow that triggered synthesis
            task_description: Original task
            task_complexity: SIMPLE_FACTUAL, MEDIUM, or COMPLEX
            messages: Agent messages used in synthesis
            system_prompt: Full system prompt sent to LLM
            user_prompt: Full user prompt with agent outputs
            result: Synthesis result dictionary
            validation_score: Optional validation score (0.0-1.0)
            validation_flags: Optional list of validation issues
            reasoning_weights: Optional per-agent reasoning weights
        """
        try:
            # Use project root database location
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "felix_knowledge.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check if synthesis_audit table exists (migration may not have run)
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='synthesis_audit'
            """)
            if not cursor.fetchone():
                logger.debug("synthesis_audit table not found, skipping audit logging")
                conn.close()
                return

            # Extract per-agent data for audit
            agent_outputs = []
            raw_confidences = []
            for msg in messages:
                if msg.message_type == MessageType.STATUS_UPDATE:
                    agent_id = msg.sender_id
                    agent_type = msg.content.get('agent_type', 'unknown')
                    confidence = msg.content.get('confidence', 0.0)
                    content = msg.content.get('content', '')

                    agent_outputs.append({
                        'agent_id': agent_id,
                        'type': agent_type,
                        'confidence': confidence,
                        'content_preview': content[:200] if content else ''
                    })
                    if confidence > 0:
                        raw_confidences.append(confidence)

            # Calculate confidence standard deviation
            if len(raw_confidences) >= 2:
                avg = sum(raw_confidences) / len(raw_confidences)
                variance = sum((c - avg) ** 2 for c in raw_confidences) / len(raw_confidences)
                confidence_std = variance ** 0.5
            else:
                confidence_std = 0.0

            # Prepare degradation reasons
            degraded_reasons = result.get('degraded_reasons', [])

            cursor.execute("""
                INSERT INTO synthesis_audit (
                    workflow_id, timestamp, task_description, task_complexity,
                    agent_count, agent_outputs_json, reasoning_weights_json,
                    raw_confidences_json, weighted_avg, confidence_std, synthesis_confidence,
                    validation_called, validation_score, validation_flags_json,
                    system_prompt, user_prompt, synthesis_content,
                    tokens_used, synthesis_time, used_fallback, degraded, degraded_reasons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                time.time(),
                task_description,
                task_complexity,
                len(messages),
                json.dumps(agent_outputs),
                json.dumps(reasoning_weights) if reasoning_weights else None,
                json.dumps(raw_confidences),
                result.get('avg_agent_confidence', 0.0),
                confidence_std,
                result.get('confidence', 0.0),
                1 if validation_score is not None else 0,
                validation_score,
                json.dumps(validation_flags) if validation_flags else None,
                system_prompt,
                user_prompt,
                result.get('synthesis_content', ''),
                result.get('tokens_used', 0),
                result.get('synthesis_time', 0.0),
                1 if result.get('used_fallback', False) else 0,
                1 if result.get('degraded', False) else 0,
                json.dumps(degraded_reasons) if degraded_reasons else None
            ))

            conn.commit()
            conn.close()
            logger.debug(f"Synthesis audit logged for workflow {workflow_id}")

        except Exception as e:
            logger.warning(f"Failed to log synthesis audit: {e}")
            # Don't let audit failures break synthesis

    def classify_task_complexity(self, task_description: str) -> str:
        """
        Classify task complexity to optimize synthesis strategy.

        Task complexity affects:
        - Synthesis temperature and token budgets
        - Agent spawning strategies
        - Workflow execution parameters

        Args:
            task_description: The task description from user

        Returns:
            Task complexity: "SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX"

        Examples:
            >>> engine.classify_task_complexity("What time is it?")
            'SIMPLE_FACTUAL'
            >>> engine.classify_task_complexity("Explain quantum computing")
            'MEDIUM'
            >>> engine.classify_task_complexity("Design a microservices architecture")
            'COMPLEX'
        """
        task_lower = task_description.lower()

        # Check for simple factual patterns (loaded from YAML)
        for pattern in self.simple_factual_patterns:
            if re.search(pattern, task_lower):
                return "SIMPLE_FACTUAL"

        # Check for medium complexity patterns (loaded from YAML)
        for pattern in self.medium_patterns:
            if re.search(pattern, task_lower):
                return "MEDIUM"

        # Default to complex for open-ended, analytical tasks
        return "COMPLEX"

    def classify_tool_requirements(self, task_description: str) -> Dict[str, bool]:
        """
        Classify which tool capabilities are needed for a task.

        This enables conditional tool instruction injection - agents only receive
        instructions for tools they actually need, reducing token waste.

        Tool categories:
        - file_operations: Creating, modifying, or managing files
        - web_search: Accessing current/real-time information
        - system_commands: Package management, process control, system inspection

        Args:
            task_description: The task description from user

        Returns:
            Dictionary with tool requirement flags:
            {
                'needs_file_ops': bool,
                'needs_web_search': bool,
                'needs_system_commands': bool
            }

        Examples:
            >>> engine.classify_tool_requirements("Create a report and save it to report.txt")
            {'needs_file_ops': True, 'needs_web_search': False, 'needs_system_commands': False}
            >>> engine.classify_tool_requirements("What's the latest news on AI?")
            {'needs_file_ops': False, 'needs_web_search': True, 'needs_system_commands': False}
            >>> engine.classify_tool_requirements("Explain quantum computing")
            {'needs_file_ops': False, 'needs_web_search': False, 'needs_system_commands': False}
            >>> engine.classify_tool_requirements("Read the file src/agents/prompt_optimization.py")
            {'needs_file_ops': True, 'needs_web_search': False, 'needs_system_commands': False}
        """
        task_lower = task_description.lower()

        # Classify tool requirements using patterns loaded from YAML
        needs_file_ops = any(re.search(pattern, task_lower) for pattern in self.file_operation_patterns)
        needs_web_search = any(re.search(pattern, task_lower) for pattern in self.web_search_patterns)
        needs_system_commands = any(re.search(pattern, task_lower) for pattern in self.system_command_patterns)

        return {
            'needs_file_ops': needs_file_ops,
            'needs_web_search': needs_web_search,
            'needs_system_commands': needs_system_commands
        }

    def classify_task_type(self, task_description: str) -> str:
        """
        Classify task type for meta-learning differentiation.

        Different task types benefit from different knowledge - enabling the
        meta-learning boost to rank knowledge appropriately per task type.

        Task types:
        - factual_question: Direct queries for specific facts (who, what, when, where)
        - explanatory_question: Questions seeking understanding (why, how)
        - creation_task: Tasks that produce new content (write, create, build)
        - analysis_task: Tasks examining existing content (analyze, evaluate)
        - research_task: Open-ended investigation tasks (research, explore)
        - problem_solving: Debugging, fixing, resolving issues

        Args:
            task_description: The task description from user

        Returns:
            Task type string for meta-learning categorization

        Examples:
            >>> engine.classify_task_type("What is the capital of France?")
            'factual_question'
            >>> engine.classify_task_type("Why does Python use indentation?")
            'explanatory_question'
            >>> engine.classify_task_type("Create a REST API for user management")
            'creation_task'
        """
        task_lower = task_description.lower()

        # Pattern matching for task type classification
        # Order matters - more specific patterns first
        patterns = {
            'factual_question': [
                r'\b(what|who|when|where|which)\b.*\?',
                r'\bis\s+(it|this|that|there)\b.*\?',
                r'\blist\s+(the|all)\b.*\?',
            ],
            'explanatory_question': [
                r'\b(why|how)\b.*\?',
                r'\bexplain\b',
                r'\bwhat\s+(causes?|makes?|leads?)\b',
            ],
            'creation_task': [
                r'\b(create|write|build|generate|make|design|implement|develop)\b',
                r'\b(add|new)\s+(a|an|the)?\s*(feature|function|class|file|module)\b',
            ],
            'analysis_task': [
                r'\b(analyze|examine|evaluate|assess|review|compare|audit)\b',
                r'\blook\s+(at|into)\b',
                r'\bcheck\s+(the|for)\b',
            ],
            'research_task': [
                r'\b(research|investigate|explore|study|find\s+out|discover)\b',
                r'\blearn\s+about\b',
                r'\bunderstand\b',
            ],
            'problem_solving': [
                r'\b(fix|solve|debug|resolve|troubleshoot|repair)\b',
                r'\b(error|bug|issue|problem|broken|failing)\b',
                r'\bnot\s+working\b',
            ],
        }

        # Check patterns in priority order
        for task_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                if re.search(pattern, task_lower):
                    logger.debug(f"Task classified as '{task_type}': matched pattern '{pattern}'")
                    return task_type

        # Default fallback
        logger.debug(f"Task classification defaulting to 'general_task' - no patterns matched")
        return 'general_task'

    def synthesize_agent_outputs(self, task_description: str, max_messages: int = 20,
                                 task_complexity: str = "COMPLEX",
                                 reasoning_evals: Optional[Dict[str, Dict[str, Any]]] = None,
                                 coverage_report: Optional[Any] = None,
                                 successful_agents: Optional[List[str]] = None,
                                 failed_agents: Optional[List[str]] = None,
                                 streaming_callback: Optional[Callable] = None,
                                 workflow_id: Optional[str] = None,
                                 approval_callback: Optional[Callable[[Dict], Dict]] = None,
                                 approval_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Synthesize final output from all agent communications.

        This is the core synthesis capability of CentralPost, replacing the need for
        synthesis agents. CentralPost represents the central axis of the helix where
        all agent trajectories converge.

        Args:
            task_description: Original task description
            max_messages: Maximum number of agent messages to include in synthesis
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX")
            reasoning_evals: Optional dict mapping agent_id to reasoning evaluation results
                from CriticAgent.evaluate_reasoning_process(). Used to weight agent
                contributions - agents with low reasoning_quality_score have reduced
                influence on synthesis confidence.
            coverage_report: Optional CoverageReport from KnowledgeCoverageAnalyzer.
                Used to compute meta-confidence and generate epistemic caveats.
            successful_agents: Optional list of agent IDs that produced valid output.
                Used for degradation assessment (Issue #18).
            failed_agents: Optional list of agent IDs that failed.
                Used for degradation assessment (Issue #18).
            streaming_callback: Optional callback for streaming synthesis output.
                If provided, synthesis will stream chunks via complete_streaming().
                Callback signature: callback(chunk) where chunk has .content attribute.
            workflow_id: Optional workflow ID for audit trail correlation.
            approval_callback: Optional callback for user approval of low-confidence
                synthesis. If provided and confidence < approval_threshold, this
                callback is invoked with synthesis review data. The callback should
                return a dict with 'action' key ('accept', 'reject', or 'regenerate')
                and optional 'strategy' and 'user_input' for regeneration.
            approval_threshold: Confidence threshold below which approval is required.
                Default 0.6. Only applies if approval_callback is provided.

        Returns:
            Dict containing:
                - synthesis_content: Final synthesized output text
                - confidence: Synthesis confidence score (0.0-1.0)
                - temperature: Temperature used for synthesis
                - tokens_used: Number of tokens used
                - max_tokens: Token budget allocated
                - agents_synthesized: Number of agent outputs included
                - timestamp: Synthesis timestamp
                - degraded: Whether the result is degraded (Issue #18)
                - degraded_reason: Human-readable reason for degradation
                - successful_agents: List of successful agent IDs
                - failed_agents: List of failed agent IDs

        Raises:
            RuntimeError: If no LLM client available for synthesis
        """
        if not self.llm_client:
            raise RuntimeError("Synthesis requires LLM client")

        logger.info("=" * 60)
        logger.info("SYNTHESIS ENGINE STARTING")
        logger.info("=" * 60)

        # Gather recent agent messages AND system action results
        messages = self._get_recent_messages(
            limit=max_messages,
            message_types=[MessageType.STATUS_UPDATE, MessageType.SYSTEM_ACTION_RESULT]
        )

        if not messages:
            logger.warning("No agent messages available for synthesis")
            return {
                "synthesis_content": "No agent outputs available for synthesis.",
                "confidence": 0.0,
                "temperature": 0.0,
                "tokens_used": 0,
                "max_tokens": 0,
                "agents_synthesized": 0,
                "timestamp": time.time(),
                "error": "no_messages"
            }

        # Calculate agent confidence with optional reasoning-quality weighting
        # Agents with low reasoning quality scores have reduced influence
        confidences = []
        weighted_confidences = []
        weights = []
        critic_count = 0
        reasoning_weighted_count = 0

        # Initialize reasoning_evals if not provided
        if reasoning_evals is None:
            reasoning_evals = {}

        for msg in messages:
            if msg.message_type == MessageType.STATUS_UPDATE:
                conf = msg.content.get('confidence', 0.0)
                agent_type = msg.content.get('agent_type', 'unknown')
                agent_id = msg.sender_id

                # Calculate reasoning-based weight for this agent
                reasoning_weight = 1.0  # Default weight
                if agent_id in reasoning_evals:
                    eval_data = reasoning_evals[agent_id]
                    reasoning_weight = eval_data.get('reasoning_quality_score', 0.7)
                    # Penalize agents flagged for re-evaluation
                    if eval_data.get('re_evaluation_needed'):
                        reasoning_weight *= 0.5
                        logger.debug(f"Agent {agent_id} flagged for re-evaluation, weight halved")
                    reasoning_weighted_count += 1

                if conf > 0:
                    confidences.append(conf)
                    weighted_confidences.append(conf * reasoning_weight)
                    weights.append(reasoning_weight)

                if agent_type == 'critic':
                    critic_count += 1

        # Use weighted average if reasoning evals were provided
        if weighted_confidences and sum(weights) > 0:
            avg_agent_confidence = sum(weighted_confidences) / sum(weights)
            if reasoning_weighted_count > 0:
                logger.info(f"  Reasoning weights applied: {reasoning_weighted_count} agents evaluated")
        else:
            avg_agent_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Calculate standard deviation of agent confidences
        if len(confidences) >= 2:
            variance = sum((c - avg_agent_confidence) ** 2 for c in confidences) / len(confidences)
            confidence_std = variance ** 0.5
        else:
            confidence_std = 0.0  # Single agent or no agents, no variance

        # Synthesis confidence is pure agent confidence (no validation weighting)
        synthesis_confidence = avg_agent_confidence

        # Boost for critic validation
        if critic_count >= 1:
            synthesis_confidence = min(1.0, synthesis_confidence * 1.1)

        # Calculate adaptive synthesis parameters (with std dev for disagreement detection)
        temperature = self.calculate_synthesis_temperature(avg_agent_confidence, confidence_std)
        max_tokens = self.calculate_synthesis_tokens(len(messages), task_complexity)

        logger.info(f"Synthesis Parameters:")
        logger.info(f"  Task complexity: {task_complexity}")
        logger.info(f"  Agent messages: {len(messages)}")
        logger.info(f"  Average agent confidence: {avg_agent_confidence:.2f}")
        logger.info(f"  Confidence std dev: {confidence_std:.2f}")
        logger.info(f"  Synthesis confidence: {synthesis_confidence:.2f}")
        logger.info(f"  Critics present: {critic_count}")
        logger.info(f"  Adaptive temperature: {temperature}")
        logger.info(f"  Adaptive token budget: {max_tokens}")

        # Build synthesis prompt
        user_prompt = self.build_synthesis_prompt(task_description, messages, task_complexity)

        # Truth-seeking system prompt
        system_prompt = """You are the Central Post of the Felix helical multi-agent system - a truth-seeking synthesis engine.

Felix agents operate along a helical geometry:
- Top of helix: Broad exploration (research agents)
- Middle spiral: Focused analysis (analysis agents)
- Bottom convergence: Critical validation (critic agents)

Your role is NOT to simply concatenate or summarize agent outputs. Your role is to REASON about them, VALIDATE them, and SYNTHESIZE TRUTH.

Your synthesis must:

1. **Validate Facts**: Cross-check claims between agents. If agents disagree, identify the disagreement and reason about which source is more authoritative or recent.

2. **Prioritize Critic Feedback**: CRITIC agents provide quality control. Their concerns must be explicitly addressed, not ignored. If critics identified issues, explain how you resolved them or acknowledge limitations.

3. **Acknowledge Contradictions**: If agent outputs contradict each other, DO NOT paper over the conflict. Explicitly note the contradiction and either:
   - Resolve it with reasoning (e.g., "Source A is more recent/authoritative")
   - Acknowledge uncertainty (e.g., "Conflicting information exists, confidence is low")

4. **Express Appropriate Uncertainty**: If validation is weak, sources are questionable, or agents disagree, EXPRESS THIS. Use phrases like:
   - "Based on available information, but with low confidence..."
   - "Sources disagree on this point..."
   - "This claim could not be verified..."

5. **Identify Gaps**: If important information is missing or agents didn't address key aspects of the query, say so.

6. **Reason Transparently**: Don't just state conclusions. Briefly explain WHY you believe certain facts over others (recency, authority, consensus, verification).

7. **Preserve Helical Insights**: Still integrate exploration (research), analysis, and validation (critic) phases - but critically, not blindly.

Your output should reflect JUSTIFIED confidence, not reflexive confidence. If the agent outputs are low-quality, your synthesis confidence should reflect that reality."""

        # Call LLM for synthesis with retry and fallback
        start_time = time.time()
        synthesis_attempts = 0
        max_attempts = 3
        last_error = None
        llm_response = None

        # Track original parameters for retry adjustments
        original_temperature = temperature
        original_max_tokens = max_tokens

        while synthesis_attempts < max_attempts:
            synthesis_attempts += 1
            try:
                # Use streaming if callback provided, otherwise standard completion
                if streaming_callback and hasattr(self.llm_client, 'complete_streaming'):
                    logger.info("Synthesis using streaming mode")

                    # Wrap callback to convert StreamingChunk to plain text
                    # Pass agent name for full mode compatibility (expects 2 args)
                    def text_callback(chunk):
                        chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        streaming_callback("synthesis_engine", chunk_text)

                    llm_response = self.llm_client.complete_streaming(
                        agent_id="synthesis_engine",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        callback=text_callback,
                        batch_interval=0.1
                    )
                else:
                    llm_response = self.llm_client.complete(
                        agent_id="synthesis_engine",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                # Success! Break out of retry loop
                break

            except Exception as e:
                last_error = e
                logger.warning(f"Synthesis attempt {synthesis_attempts}/{max_attempts} failed: {e}")

                if synthesis_attempts < max_attempts:
                    # Retry with adjusted parameters
                    temperature = min(1.0, temperature * 1.2)  # Increase temp by 20%
                    max_tokens = int(max_tokens * 0.8)  # Reduce tokens by 20%
                    logger.info(f"Retrying with temp={temperature:.2f}, tokens={max_tokens}")
                else:
                    # All attempts failed, use fallback
                    logger.error(f"All {max_attempts} synthesis attempts failed, using fallback")
                    llm_response = self._create_fallback_synthesis(
                        task_description=task_description,
                        messages=messages,
                        task_complexity=task_complexity,
                        error=last_error
                    )

        synthesis_time = time.time() - start_time

        # Compute meta-confidence with coverage adjustment (Phase 7)
        meta_confidence_result = self.compute_meta_confidence(
            agent_confidence=synthesis_confidence,
            coverage_report=coverage_report
        )

        logger.info(f"✓ Synthesis complete in {synthesis_time:.2f}s")
        logger.info(f"  Attempts: {synthesis_attempts}")
        logger.info(f"  Tokens used: {llm_response.tokens_used} / {original_max_tokens}")
        logger.info(f"  Content length: {len(llm_response.content)} chars")
        if meta_confidence_result['coverage_adjustment'] != 0:
            logger.info(f"  Meta-confidence: {meta_confidence_result['meta_confidence']:.2f} "
                       f"(adjustment: {meta_confidence_result['coverage_adjustment']:+.2f})")
        if meta_confidence_result['epistemic_caveats']:
            logger.info(f"  Epistemic caveats: {len(meta_confidence_result['epistemic_caveats'])}")
        if synthesis_attempts >= max_attempts:
            logger.warning(f"  ⚠️  Used fallback synthesis (all LLM attempts failed)")

        # Assess degradation (Issue #18)
        used_fallback = synthesis_attempts >= max_attempts
        is_degraded, degraded_reason, degraded_reasons = self.assess_degradation(
            successful_agents=successful_agents or [],
            failed_agents=failed_agents or [],
            synthesis_confidence=synthesis_confidence,
            coverage_report=coverage_report,
            used_fallback=used_fallback
        )

        # Generate degraded response prefix if needed
        final_content = llm_response.content
        if is_degraded and synthesis_confidence < 0.5:
            final_content = self._generate_degraded_response(
                original_content=llm_response.content,
                degraded_reason=degraded_reason,
                confidence=synthesis_confidence
            )

        # Call validation functions (imported at line 35, now actually used!)
        # These functions are designed for knowledge entries but can work with synthesis output
        validation_score = None
        validation_flags = None
        try:
            # Wrap synthesis content for validation
            # - source_agent: "synthesis_engine" (this is synthesis output)
            # - domain: "workflow_task" (agent-generated content)
            # - confidence_level: use the synthesis_confidence
            validation_score = calculate_validation_score(
                content=final_content,
                source_agent="synthesis_engine",
                domain="workflow_task",
                confidence_level=synthesis_confidence,
                existing_knowledge=None
            )
            validation_flags = get_validation_flags(
                content=final_content,
                source_agent="synthesis_engine",
                domain="workflow_task",
                existing_knowledge=None
            )

            logger.info(f"  Validation score: {validation_score:.2f}")
            if validation_flags:
                logger.warning(f"  Validation flags: {validation_flags}")
        except Exception as e:
            logger.warning(f"  Validation check failed: {e}")

        logger.info("=" * 60)

        # Build result dict
        result = {
            "synthesis_content": final_content,
            "confidence": synthesis_confidence,  # Pure agent confidence (no validation weighting)
            "meta_confidence": meta_confidence_result['meta_confidence'],  # Phase 7: Coverage-adjusted
            "coverage_adjustment": meta_confidence_result['coverage_adjustment'],
            "epistemic_caveats": meta_confidence_result['epistemic_caveats'],
            "caveat_summary": meta_confidence_result['caveat_summary'],
            "temperature": original_temperature,  # Return original temperature, not retry-adjusted
            "tokens_used": llm_response.tokens_used,
            "max_tokens": original_max_tokens,
            "agents_synthesized": len(messages),
            "avg_agent_confidence": avg_agent_confidence,
            "critic_count": critic_count,
            "synthesis_time": synthesis_time,
            "synthesis_attempts": synthesis_attempts,  # Track retry attempts
            "used_fallback": used_fallback,  # Fallback indicator
            "timestamp": time.time(),
            # Degradation tracking (Issue #18)
            "degraded": is_degraded,
            "degraded_reason": degraded_reason,
            "degraded_reasons": degraded_reasons,
            "successful_agents": successful_agents or [],
            "failed_agents": failed_agents or [],
            # Validation results (new - addressing audit gap)
            "validation_score": validation_score,
            "validation_flags": validation_flags,
        }

        # Confidence gating with user approval (addressing human-in-the-loop gap)
        if approval_callback and synthesis_confidence < approval_threshold:
            logger.info(f"  Confidence {synthesis_confidence:.2f} below threshold {approval_threshold:.2f}")
            logger.info("  Requesting user approval for synthesis...")

            # Build review data for approval callback
            review_data = {
                'type': 'synthesis_review',
                'confidence': synthesis_confidence,
                'meta_confidence': meta_confidence_result['meta_confidence'],
                'content_preview': final_content[:500] if len(final_content) > 500 else final_content,
                'full_content': final_content,
                'degraded': is_degraded,
                'degraded_reason': degraded_reason,
                'validation_score': validation_score,
                'validation_flags': validation_flags or [],
                'agent_count': len(messages),
                'task_description': task_description,
                'options': [
                    {'id': 'accept', 'label': 'Accept as-is'},
                    {'id': 'regenerate_focused', 'label': 'Regenerate (more focused)',
                     'strategy': 'parameter_adjust'},
                    {'id': 'regenerate_context', 'label': 'Add context and regenerate',
                     'strategy': 'context_injection', 'requires_input': True},
                    {'id': 'regenerate_agents', 'label': 'Spawn more agents',
                     'strategy': 'spawn_more_agents'},
                    {'id': 'regenerate_search', 'label': 'Search web and regenerate',
                     'strategy': 'web_search_boost'},
                    {'id': 'regenerate_knowledge', 'label': 'Expand knowledge search',
                     'strategy': 'knowledge_expand'},
                    {'id': 'reject', 'label': 'Reject synthesis'}
                ]
            }

            # Invoke approval callback and wait for user decision
            try:
                decision = approval_callback(review_data)

                if decision.get('action') == 'accept':
                    logger.info("  User accepted synthesis as-is")
                    result['user_approved'] = True

                elif decision.get('action') == 'reject':
                    logger.info("  User rejected synthesis")
                    result['synthesis_content'] = None
                    result['user_declined'] = True
                    result['decline_reason'] = decision.get('reason', 'User rejected synthesis')
                    # Still log the audit for the rejected synthesis
                    result['user_approved'] = False

                elif decision.get('action', '').startswith('regenerate'):
                    logger.info(f"  User requested regeneration: {decision.get('strategy')}")
                    result['regeneration_requested'] = True
                    result['regeneration_strategy'] = decision.get('strategy')
                    result['regeneration_user_input'] = decision.get('user_input')
                    result['user_approved'] = False
                    # Note: Caller should handle regeneration using RegenerationExecutor

                else:
                    logger.warning(f"  Unknown approval decision: {decision}")
                    result['user_approved'] = True  # Default to accept

            except Exception as e:
                logger.error(f"  Approval callback failed: {e}")
                result['approval_error'] = str(e)
                result['user_approved'] = True  # Continue on callback failure
        else:
            # No approval needed (confidence above threshold or no callback)
            result['user_approved'] = True

        # Log full synthesis audit trail
        # Extract reasoning weights from reasoning_evals if provided
        reasoning_weights = None
        if reasoning_evals:
            reasoning_weights = {
                agent_id: eval_data.get('reasoning_quality_score', 1.0)
                for agent_id, eval_data in reasoning_evals.items()
            }

        self._log_synthesis_audit(
            workflow_id=workflow_id,
            task_description=task_description,
            task_complexity=task_complexity,
            messages=messages,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            result=result,
            validation_score=validation_score,
            validation_flags=validation_flags,
            reasoning_weights=reasoning_weights
        )

        return result

    def _create_fallback_synthesis(self,
                                    task_description: str,
                                    messages: List[Message],
                                    task_complexity: str,
                                    error: Exception) -> Any:
        """
        Create fallback synthesis response when LLM fails.

        Uses simple concatenation of agent outputs with metadata note.
        Ensures synthesis never completely fails.

        Args:
            task_description: Original task
            messages: Agent messages
            task_complexity: Task complexity level
            error: Error that caused fallback

        Returns:
            Mock LLM response object with content and tokens_used
        """
        # Build simple synthesis from agent outputs
        fallback_parts = [
            f"⚠️  LLM Synthesis Failed ({type(error).__name__})",
            f"",
            f"Task: {task_description}",
            f"",
            f"Agent Outputs Summary:",
            f""
        ]

        for i, msg in enumerate(messages, 1):
            if msg.message_type == MessageType.STATUS_UPDATE:
                agent_type = msg.content.get('agent_type', 'unknown')
                content = msg.content.get('content', '')
                confidence = msg.content.get('confidence', 0.0)

                fallback_parts.append(f"{i}. {agent_type.upper()} (conf: {confidence:.2f})")
                # Truncate long content for fallback readability
                if len(content) > 200:
                    fallback_parts.append(f"   {content[:200]}...")
                else:
                    fallback_parts.append(f"   {content}")
                fallback_parts.append("")

        fallback_parts.append("---")
        fallback_parts.append(f"Note: This is a fallback synthesis due to LLM error. {len(messages)} agent outputs summarized.")

        fallback_content = "\n".join(fallback_parts)

        # Create mock response object
        MockResponse = namedtuple('LLMResponse', ['content', 'tokens_used'])
        return MockResponse(
            content=fallback_content,
            tokens_used=len(fallback_content) // 4  # Rough token estimate
        )

    def assess_degradation(
        self,
        successful_agents: List[str],
        failed_agents: List[str],
        synthesis_confidence: float,
        coverage_report: Optional[Any],
        used_fallback: bool
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Assess if synthesis result should be marked as degraded.

        A synthesis is degraded when agent failures or other factors make
        the result unreliable and the user should be warned.

        Args:
            successful_agents: List of agent IDs that contributed successfully
            failed_agents: List of agent IDs that failed
            synthesis_confidence: Confidence score from synthesis (0.0-1.0)
            coverage_report: Optional coverage analysis report
            used_fallback: Whether fallback synthesis was used

        Returns:
            Tuple of (is_degraded, human_readable_reason, list_of_reason_codes)
        """
        reasons = []
        explanations = []

        total_agents = len(successful_agents) + len(failed_agents)

        # Check agent failure rate (>= 50% failed)
        if total_agents > 0:
            failure_rate = len(failed_agents) / total_agents
            if failure_rate >= 0.5:
                reasons.append("agent_failures")
                explanations.append(
                    f"{len(failed_agents)} of {total_agents} agents failed"
                )

        # Check for critical agent types in failures (research, analysis)
        critical_types = {'research', 'analysis'}
        for agent_id in failed_agents:
            # Agent IDs are typically like "research_agent_abc123"
            agent_type = agent_id.split('_')[0] if '_' in agent_id else ''
            if agent_type in critical_types:
                if "agent_failures" not in reasons:
                    reasons.append("agent_failures")
                explanations.append(f"Critical {agent_type} agent failed")
                break  # Only report once

        # Check LLM availability (fallback used)
        if used_fallback:
            reasons.append("llm_unavailable")
            explanations.append("LLM unavailable, used fallback synthesis")

        # Check confidence threshold (< 0.4)
        if synthesis_confidence < 0.4:
            reasons.append("low_confidence")
            explanations.append(f"Low synthesis confidence ({synthesis_confidence:.2f})")

        # Check knowledge coverage (< 0.3)
        if coverage_report:
            coverage_score = getattr(coverage_report, 'overall_coverage_score', 1.0)
            if coverage_score < 0.3:
                reasons.append("insufficient_coverage")
                explanations.append(f"Insufficient knowledge coverage ({coverage_score:.2f})")

        is_degraded = len(reasons) > 0
        human_reason = "; ".join(explanations) if explanations else None

        if is_degraded:
            logger.warning(f"⚠️ Synthesis degraded: {human_reason}")

        return is_degraded, human_reason, reasons

    def _generate_degraded_response(
        self,
        original_content: str,
        degraded_reason: str,
        confidence: float
    ) -> str:
        """
        Prepend degradation notice to synthesis output when reliability is low.

        For very low confidence, adds "I cannot provide a reliable answer" prefix.
        For moderately low confidence, adds a caution note.

        Args:
            original_content: The original synthesis content
            degraded_reason: Human-readable reason for degradation
            confidence: Synthesis confidence score (0.0-1.0)

        Returns:
            Content with appropriate degradation prefix
        """
        if confidence < 0.3:
            prefix = (
                "**I cannot provide a reliable answer to this question.**\n\n"
                f"Reason: {degraded_reason}\n\n"
                "The following is based on limited information and should be "
                "treated with caution:\n\n---\n\n"
            )
        elif confidence < 0.5:
            prefix = (
                "**Note: This response has reduced reliability.**\n\n"
                f"Reason: {degraded_reason}\n\n---\n\n"
            )
        else:
            prefix = ""

        return prefix + original_content

    def calculate_synthesis_temperature(self, avg_confidence: float, confidence_std: float = 0.0) -> float:
        """
        Calculate adaptive temperature for synthesis based on agent confidence consensus.

        High confidence + low variance → focused synthesis (0.2)
        Medium confidence + moderate variance → balanced synthesis (0.3)
        Low confidence + high variance → creative integration (0.4+)

        High variance (agents disagree) increases temperature for creative reasoning.

        Args:
            avg_confidence: Average confidence from agent outputs (0.0-1.0)
            confidence_std: Standard deviation of confidences (0.0-1.0)

        Returns:
            Temperature value (0.2-0.6)
        """
        # Base temperature from average confidence
        if avg_confidence >= 0.9:
            base_temp = 0.2  # High confidence → very focused
        elif avg_confidence >= 0.75:
            base_temp = 0.3  # Medium confidence → balanced
        else:
            base_temp = 0.4  # Lower confidence → more creative integration

        # Increase temperature for high variance (agents disagree significantly)
        # std > 0.25 indicates real disagreement, needs creative synthesis
        variance_adjustment = confidence_std * 0.5

        # Cap at 0.6 to avoid excessive randomness
        return min(0.6, base_temp + variance_adjustment)

    def calculate_synthesis_tokens(self, agent_count: int, task_complexity: str = "COMPLEX") -> int:
        """
        Calculate adaptive token budget for synthesis based on number of agents and task complexity.

        More agents → more content to synthesize → larger budget
        Simpler tasks → less synthesis needed → smaller budget

        Args:
            agent_count: Number of agent outputs to synthesize
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX")

        Returns:
            Token budget (200-3000)
        """
        # Simple factual queries need minimal synthesis
        if task_complexity == "SIMPLE_FACTUAL":
            return 200  # Just answer the question directly

        # Medium complexity gets moderate token budget
        if task_complexity == "MEDIUM":
            return 800 if agent_count < 5 else 1200

        # Complex tasks get full token budget based on team size
        if agent_count >= 10:
            return 3000  # Many agents → comprehensive synthesis
        elif agent_count >= 5:
            return 2000  # Medium team → balanced synthesis
        else:
            return 1500  # Small team → focused synthesis

    def compute_meta_confidence(self, agent_confidence: float,
                                coverage_report: Optional[Any] = None,
                                caveat_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Compute meta-confidence that accounts for knowledge coverage gaps.

        Meta-confidence represents epistemic self-awareness: Felix knows what it
        knows AND what it doesn't know. The result includes confidence adjustments
        and epistemic caveats for areas of uncertainty.

        Args:
            agent_confidence: Average confidence from agent synthesis
            coverage_report: Optional CoverageReport from KnowledgeCoverageAnalyzer
            caveat_threshold: Confidence threshold below which to generate caveats

        Returns:
            Dict containing:
                - meta_confidence: Coverage-adjusted confidence score
                - coverage_adjustment: How much coverage affected confidence
                - epistemic_caveats: List of uncertainty warnings
                - caveat_summary: Human-readable summary of limitations
                - should_inject_caveats: Whether caveats should appear in output
        """
        # Default to pure agent confidence if no coverage data
        if coverage_report is None:
            return {
                'meta_confidence': agent_confidence,
                'coverage_adjustment': 0.0,
                'epistemic_caveats': [],
                'caveat_summary': '',
                'should_inject_caveats': False
            }

        # Get coverage score (0.0-1.0)
        coverage_score = getattr(coverage_report, 'overall_coverage_score', 1.0)

        # Compute weighted coverage factor
        # Coverage < 0.5 significantly reduces confidence
        # Coverage > 0.8 provides slight confidence boost (capped at 1.05x)
        if coverage_score < 0.3:
            coverage_factor = 0.6  # Severe penalty for major gaps
        elif coverage_score < 0.5:
            coverage_factor = 0.75  # Moderate penalty
        elif coverage_score < 0.7:
            coverage_factor = 0.9  # Slight penalty
        else:
            coverage_factor = min(1.05, 0.9 + coverage_score * 0.15)  # Slight boost

        # Compute meta-confidence
        meta_confidence = agent_confidence * coverage_factor
        coverage_adjustment = meta_confidence - agent_confidence

        # Generate epistemic caveats for weak/missing domains
        caveats = []
        missing = getattr(coverage_report, 'missing_domains', [])
        weak = getattr(coverage_report, 'weak_domains', [])

        if missing:
            caveats.append(f"No knowledge available for: {', '.join(missing)}")
        if weak:
            caveats.append(f"Limited knowledge in: {', '.join(weak)}")
        if meta_confidence < 0.5:
            caveats.append("Overall knowledge coverage is low - response may be incomplete")

        # Generate human-readable summary
        if caveats:
            caveat_summary = "⚠️ EPISTEMIC LIMITATIONS: " + "; ".join(caveats)
        else:
            caveat_summary = ""

        # Determine if caveats should appear in output
        should_inject = (meta_confidence < caveat_threshold) or len(caveats) > 0

        return {
            'meta_confidence': meta_confidence,
            'coverage_adjustment': coverage_adjustment,
            'epistemic_caveats': caveats,
            'caveat_summary': caveat_summary,
            'should_inject_caveats': should_inject
        }

    def generate_epistemic_prompt_section(self, meta_confidence_result: Dict[str, Any]) -> str:
        """
        Generate a prompt section instructing the LLM to express appropriate uncertainty.

        When coverage gaps exist, this section instructs the synthesizer to:
        - Acknowledge limitations
        - Use appropriate hedging language
        - Suggest areas where more information may be needed

        Args:
            meta_confidence_result: Output from compute_meta_confidence()

        Returns:
            Prompt text to inject, or empty string if no caveats needed
        """
        if not meta_confidence_result.get('should_inject_caveats'):
            return ""

        caveats = meta_confidence_result.get('epistemic_caveats', [])
        meta_conf = meta_confidence_result.get('meta_confidence', 1.0)

        prompt_parts = [
            "\n---",
            "⚠️ EPISTEMIC SELF-AWARENESS REQUIRED",
            "",
            "Knowledge coverage analysis indicates gaps. You MUST:",
        ]

        if caveats:
            prompt_parts.append("")
            prompt_parts.append("Identified gaps:")
            for caveat in caveats:
                prompt_parts.append(f"  • {caveat}")

        prompt_parts.extend([
            "",
            "Required behavior:",
            "1. Acknowledge areas where information may be incomplete",
            "2. Use hedging language (\"based on available information\", \"this appears to be\")",
            "3. Explicitly state if certain aspects could not be verified",
            "4. Suggest what additional information would strengthen the response",
            "",
            f"Current meta-confidence: {meta_conf:.2f}",
            "---",
            ""
        ])

        return "\n".join(prompt_parts)

    def _sanitize_content(self, content: str, max_length: int = 10000) -> str:
        """
        Sanitize user/agent/system content to prevent prompt injection.

        Removes/escapes:
        - System prompt markers (```system, <|system|>, etc.)
        - Role markers (Assistant:, Human:, User:, System:)
        - Instruction override attempts ("You are", "Ignore previous", etc.)
        - Excessive newlines/whitespace
        - Control characters (except newline, tab, carriage return)
        - Truncates to max_length

        Args:
            content: Content to sanitize
            max_length: Max characters (default 10k)

        Returns:
            Sanitized content
        """
        if not content or not isinstance(content, str):
            return ""

        # Remove system prompt injection attempts
        content = re.sub(r'```\s*system\s*```?', '[removed]', content, flags=re.IGNORECASE)
        content = re.sub(r'<\|system\|>.*?<\|/system\|>', '[removed]', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<\|im_start\|>system.*?<\|im_end\|>', '[removed]', content, flags=re.DOTALL | re.IGNORECASE)

        # Remove role markers that could confuse the LLM
        content = re.sub(r'^(Assistant|Human|User|System|AI):\s*', '', content, flags=re.MULTILINE | re.IGNORECASE)

        # Remove instruction override attempts
        content = re.sub(r'\b(You are now|Ignore (all )?previous|Disregard|Forget)\b', '[removed]', content, flags=re.IGNORECASE)
        content = re.sub(r'\bSYSTEM:', '[removed]', content, flags=re.IGNORECASE)
        content = re.sub(r'<\|endoftext\|>', '[removed]', content, flags=re.IGNORECASE)

        # Normalize excessive whitespace (max 3 consecutive newlines, max 2 spaces)
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        content = re.sub(r' {3,}', '  ', content)

        # Remove control characters except \n \r \t
        content = ''.join(c for c in content if c.isprintable() or c in '\n\r\t')

        # Truncate if needed
        if len(content) > max_length:
            content = content[:max_length] + '\n...[truncated for safety]'

        return content

    def build_synthesis_prompt(self, task_description: str, messages: List[Message],
                                task_complexity: str = "COMPLEX") -> str:
        """
        Build synthesis prompt from task description and agent messages using PromptManager.

        All agent content, stdout, stderr, and commands are sanitized to prevent
        prompt injection attacks.

        Args:
            task_description: Original task description
            messages: List of agent messages to synthesize
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX")

        Returns:
            Formatted synthesis prompt with sanitized content
        """
        # Sanitize all message content to prevent prompt injection
        sanitized_messages = []
        for msg in messages:
            # Create a copy of the message with sanitized content
            sanitized_content = msg.content.copy()

            if msg.message_type == MessageType.STATUS_UPDATE:
                # Sanitize agent output
                if 'content' in sanitized_content:
                    sanitized_content['content'] = self._sanitize_content(sanitized_content['content'])

            elif msg.message_type == MessageType.SYSTEM_ACTION_RESULT:
                # Sanitize command execution results
                if 'command' in sanitized_content:
                    sanitized_content['command'] = self._sanitize_content(sanitized_content['command'], max_length=500)
                if 'stdout' in sanitized_content:
                    sanitized_content['stdout'] = self._sanitize_content(sanitized_content['stdout'], max_length=5000)
                if 'stderr' in sanitized_content:
                    sanitized_content['stderr'] = self._sanitize_content(sanitized_content['stderr'], max_length=5000)

            # Create new message with sanitized content
            sanitized_msg = Message(
                sender_id=msg.sender_id,
                message_type=msg.message_type,
                content=sanitized_content,
                timestamp=msg.timestamp
            )
            sanitized_messages.append(sanitized_msg)

        # Build message list section
        prompt_parts = [
            f"Original Task: {task_description}",
            "",
            "Agent Communications to Synthesize:",
            ""
        ]

        # Add each agent output and system action result with metadata
        for i, msg in enumerate(sanitized_messages, 1):
            if msg.message_type == MessageType.STATUS_UPDATE:
                agent_type = msg.content.get('agent_type', 'unknown')
                content = msg.content.get('content', '')
                confidence = msg.content.get('confidence', 0.0)

                prompt_parts.append(
                    f"{i}. {agent_type.upper()} Agent (confidence: {confidence:.2f}):"
                )
                prompt_parts.append(content)
                prompt_parts.append("")

            elif msg.message_type == MessageType.SYSTEM_ACTION_RESULT:
                command = msg.content.get('command', '')
                stdout = msg.content.get('stdout', '')
                stderr = msg.content.get('stderr', '')
                success = msg.content.get('success', False)
                exit_code = msg.content.get('exit_code', -1)

                prompt_parts.append(f"{i}. SYSTEM COMMAND EXECUTION:")
                prompt_parts.append(f"   Command: {command}")
                prompt_parts.append(f"   Success: {success}")
                prompt_parts.append(f"   Exit Code: {exit_code}")
                if stdout:
                    prompt_parts.append(f"   Output: {stdout}")
                if stderr:
                    prompt_parts.append(f"   Errors: {stderr}")
                prompt_parts.append("")

        prompt_parts.append("---")
        prompt_parts.append("")

        # Get synthesis instructions from PromptManager
        synthesis_instructions = self._get_synthesis_instructions(task_complexity)
        prompt_parts.append(synthesis_instructions)

        return "\n".join(prompt_parts)

    def _get_synthesis_instructions(self, task_complexity: str) -> str:
        """
        Get synthesis instructions from PromptManager based on task complexity.

        Args:
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX")

        Returns:
            Synthesis instructions string
        """
        # Map complexity to prompt key
        prompt_key_map = {
            "SIMPLE_FACTUAL": "synthesis_simple_factual",
            "MEDIUM": "synthesis_medium",
            "COMPLEX": "synthesis_complex"
        }

        prompt_key = prompt_key_map.get(task_complexity, "synthesis_complex")

        # Try to get from PromptManager if available
        if self.prompt_manager:
            try:
                prompt_template = self.prompt_manager.get_prompt(prompt_key)
                if prompt_template:
                    return prompt_template.template
            except Exception as e:
                logger.warning(f"Failed to get synthesis prompt from PromptManager: {e}. Using fallback.")

        # Fallback inline templates
        if task_complexity == "SIMPLE_FACTUAL":
            return """🎯 SIMPLE FACTUAL QUERY DETECTED

This is a straightforward factual question. Your synthesis should:
- Provide a DIRECT, CONCISE answer in 1-3 sentences
- Maximum length: 50-100 words
- State the key fact or information clearly
- NO philosophical analysis, NO elaborate discussion
- NO exploration of implications or deeper meanings
- Just answer the question directly

Example format: "The current date and time is [answer]. (Source: [if applicable])\""""
        elif task_complexity == "MEDIUM":
            return """Create a focused synthesis (3-5 paragraphs) that directly addresses the question.
Balance completeness with conciseness.

Guidelines:
- Target length: 3-5 paragraphs (200-400 words)
- Stay focused on the specific question asked
- Provide context where needed, but avoid tangents
- Use clear structure with topic sentences
- Prioritize practical information over abstract discussion"""
        else:
            return """Create a comprehensive final synthesis that integrates all agent findings above.

Guidelines:
- Focus on answering the core question directly
- Use structured sections if there are multiple aspects
- Be thorough but avoid unnecessary elaboration
- Stay grounded in the task requirements
- Avoid philosophical tangents unless directly relevant to the task
- Target length: 5-10 paragraphs for most complex tasks
- Prioritize clarity and actionable insights over abstract discussion"""
