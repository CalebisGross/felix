"""
Felix Agent - The Core Identity Layer

This module implements the FelixAgent class, which represents Felix's unified
identity across all interaction modes. Whether the user is asking a simple
question or requesting complex multi-agent orchestration, Felix always responds
AS Felix - not as the raw underlying LLM.

The FelixAgent:
- Embodies Felix's consistent identity (from config/chat_system_prompt.md)
- Routes requests based on complexity or explicit mode selection
- Handles direct inference for simple queries (no specialist agents)
- Orchestrates specialists for complex tasks (full workflow)
- Maintains knowledge brain integration
- Supports system command execution via trust system
"""

import time
import logging
import re
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

# Felix framework imports for full integration
from src.communication.message_types import Message, MessageType
from src.memory.task_memory import TaskComplexity, TaskOutcome
from src.memory.knowledge_store import KnowledgeQuery, ConfidenceLevel

logger = logging.getLogger(__name__)


@dataclass
class FelixResponse:
    """Response from FelixAgent processing."""
    content: str
    mode_used: str  # "direct", "light", "full"
    complexity: str  # "SIMPLE_FACTUAL", "MEDIUM", "COMPLEX"
    confidence: float = 0.0
    thinking_steps: Optional[List[Dict[str, Any]]] = None
    knowledge_sources: Optional[List[str]] = None
    execution_time: float = 0.0
    error: Optional[str] = None


class FelixAgent:
    """
    The unified Felix persona - always responds AS Felix.

    FelixAgent is the "face" of Felix that users interact with. It:
    - Maintains Felix's consistent identity regardless of mode
    - Routes requests to appropriate processing (direct, light, or full)
    - Integrates with knowledge brain for context
    - Handles system command execution via trust system
    - Streams responses back to the UI

    The key insight: Felix is NOT the specialist agents (Research, Analysis,
    Critic). Felix is the conductor - the specialists are Felix's internal
    thought processes. The response always comes FROM Felix.
    """

    def __init__(self, felix_system):
        """
        Initialize FelixAgent with access to Felix system components.

        Args:
            felix_system: Initialized FelixSystem instance providing:
                - central_post: Hub-spoke communication
                - lm_client: LLM client for inference
                - knowledge_store: Knowledge brain
                - task_memory: Learning system
                - system_command_manager: Trust-based execution
                - config: FelixConfig settings
        """
        self.felix_system = felix_system
        self.central_post = felix_system.central_post
        self.lm_client = felix_system.lm_client
        self.knowledge_store = getattr(felix_system, 'knowledge_store', None)
        self.task_memory = getattr(felix_system, 'task_memory', None)
        self.system_command_manager = getattr(felix_system, 'system_command_manager', None)
        self.config = felix_system.config

        # Load Felix identity
        self.identity_prompt = self._load_identity()

        logger.info("FelixAgent initialized - Felix identity layer active")

    def _load_identity(self) -> str:
        """
        Load Felix's identity from the system prompt configuration.

        Returns:
            str: Felix's system prompt defining identity and behavior
        """
        # Look for system prompt in config directory
        project_root = Path(__file__).parent.parent.parent
        prompt_path = project_root / "config" / "chat_system_prompt.md"

        if prompt_path.exists():
            try:
                content = prompt_path.read_text(encoding='utf-8')
                # Replace datetime placeholder
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                content = content.replace("{{currentDateTime}}", current_datetime)
                logger.debug(f"Loaded Felix identity from {prompt_path}")
                return content
            except Exception as e:
                logger.warning(f"Failed to load Felix identity: {e}")

        # Fallback minimal identity
        return """You are Felix, an air-gapped multi-agent AI framework.
You are NOT ChatGPT, GPT-4, Claude, or any cloud-based AI.
You are Felix - a local, private, air-gapped AI assistant.
Always identify yourself as Felix when asked about your identity."""

    def classify_complexity(self, message: str) -> str:
        """
        Classify message complexity using SynthesisEngine patterns.

        Args:
            message: User's input message

        Returns:
            str: "SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX"
        """
        # Use SynthesisEngine's classification if available
        if hasattr(self.central_post, 'synthesis_engine'):
            return self.central_post.synthesis_engine.classify_task_complexity(message)

        # Fallback pattern matching
        message_lower = message.lower()

        # Simple factual patterns
        simple_patterns = [
            r'\b(what|when|who|where)\s+(is|are|was|were)\b',
            r'\bwhat\s+time\b',
            r'\bwhat\s+date\b',
            r'\bhello\b',
            r'\bhi\b',
            r'\bhow\s+are\s+you\b',
        ]

        for pattern in simple_patterns:
            if re.search(pattern, message_lower):
                return "SIMPLE_FACTUAL"

        # Complex patterns
        complex_patterns = [
            r'\b(create|build|implement|design|develop)\b.*\b(system|application|framework)\b',
            r'\b(analyze|investigate|research)\b.*\b(comprehensive|detailed|thorough)\b',
            r'\bstep[\s-]by[\s-]step\b',
            r'\bmulti[\s-]?step\b',
        ]

        for pattern in complex_patterns:
            if re.search(pattern, message_lower):
                return "COMPLEX"

        # Default to medium
        return "MEDIUM"

    def process(
        self,
        message: str,
        mode: str = "auto",
        streaming_callback: Optional[Callable] = None,
        knowledge_enabled: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        cancel_event: Optional[threading.Event] = None,
        approval_callback: Optional[Callable[[Dict], Dict]] = None,
        approval_threshold: float = 0.6
    ) -> FelixResponse:
        """
        Process a message through Felix.

        This is the main entry point for all interactions with Felix.
        Routes to appropriate processing based on mode and complexity.

        Args:
            message: User's input message
            mode: Processing mode:
                - "auto": Classify and route automatically
                - "direct": Force direct inference (Simple mode in GUI)
                - "full": Force full orchestration (Workflow mode in GUI)
            streaming_callback: Callback for streaming chunks
                - For direct mode: callback(chunk_text)
                - For full mode: callback(agent_name, chunk_text)
            knowledge_enabled: Whether to include knowledge brain context
            conversation_history: Previous messages for context
            cancel_event: Optional threading.Event to signal cancellation
            approval_callback: Callback for user approval of low-confidence synthesis
            approval_threshold: Confidence threshold below which approval is required

        Returns:
            FelixResponse with results
        """
        start_time = time.time()

        # Determine complexity
        if mode == "auto":
            complexity = self.classify_complexity(message)
        elif mode == "direct":
            complexity = "SIMPLE_FACTUAL"
        else:  # "full"
            complexity = "COMPLEX"

        logger.info(f"FelixAgent processing: mode={mode}, complexity={complexity}")

        try:
            # Gather knowledge context if enabled
            knowledge_context = ""
            knowledge_sources = []
            if knowledge_enabled:
                knowledge_context, knowledge_sources = self._gather_knowledge(message)

            # Route based on complexity/mode
            if mode == "direct" or (mode == "auto" and complexity == "SIMPLE_FACTUAL"):
                response = self._respond_directly(
                    message=message,
                    knowledge_context=knowledge_context,
                    conversation_history=conversation_history,
                    streaming_callback=streaming_callback,
                    cancel_event=cancel_event
                )
                mode_used = "direct"

            elif mode == "auto" and complexity == "MEDIUM":
                # For medium complexity, still use direct but could be enhanced later
                response = self._respond_directly(
                    message=message,
                    knowledge_context=knowledge_context,
                    conversation_history=conversation_history,
                    streaming_callback=streaming_callback,
                    cancel_event=cancel_event
                )
                mode_used = "direct"

            else:  # "full" or COMPLEX
                response = self._full_orchestration(
                    message=message,
                    knowledge_context=knowledge_context,
                    conversation_history=conversation_history,
                    streaming_callback=streaming_callback,
                    cancel_event=cancel_event
                )
                mode_used = "full"

            execution_time = time.time() - start_time

            return FelixResponse(
                content=response.get('content', ''),
                mode_used=mode_used,
                complexity=complexity,
                confidence=response.get('confidence', 0.0),
                thinking_steps=response.get('thinking_steps'),
                knowledge_sources=knowledge_sources if knowledge_sources else None,
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"FelixAgent error: {e}", exc_info=True)
            execution_time = time.time() - start_time
            return FelixResponse(
                content=f"Error: {str(e)}",
                mode_used="error",
                complexity=complexity,
                execution_time=execution_time,
                error=str(e)
            )

    def _gather_knowledge(self, message: str) -> tuple:
        """
        Gather relevant knowledge context for the message.

        Uses MemoryFacade for sophisticated retrieval when available,
        falling back to simple search otherwise.

        Args:
            message: User's input message

        Returns:
            tuple: (knowledge_context_string, list_of_sources)
        """
        if not self.knowledge_store:
            return "", []

        try:
            results = []

            # Prefer MemoryFacade for sophisticated retrieval (domain filtering, relevance, meta-learning)
            if self.central_post and hasattr(self.central_post, 'memory_facade') and self.central_post.memory_facade:
                query = KnowledgeQuery(
                    domains=None,  # Search all domains (including web_search results)
                    content_keywords=[message],
                    min_confidence=ConfidenceLevel.MEDIUM,
                    limit=10,
                    use_semantic_search=True  # Enable if embeddings available
                )
                results = self.central_post.memory_facade.retrieve_knowledge_with_query(query)
                logger.debug(f"Knowledge retrieval via MemoryFacade: {len(results)} entries")
            else:
                # Fallback to simple search
                results = self.knowledge_store.advanced_search(content=message, limit=10)
                logger.debug(f"Knowledge retrieval via simple search: {len(results)} entries")

            if not results:
                return "", []

            # Build context string from KnowledgeEntry objects
            context_parts = []
            sources = []

            for entry in results:
                # KnowledgeEntry has .content (Dict) and .source_agent (str)
                content = entry.content.get('text', '') if isinstance(entry.content, dict) else str(entry.content)
                source = entry.source_agent or 'Unknown'

                if content:
                    context_parts.append(f"[From {source}]:\n{content}")
                    if source not in sources:
                        sources.append(source)

            knowledge_context = "\n\n".join(context_parts)
            return knowledge_context, sources

        except Exception as e:
            logger.warning(f"Knowledge retrieval error: {e}")
            return "", []

    def _respond_directly(
        self,
        message: str,
        knowledge_context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        streaming_callback: Optional[Callable] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Dict[str, Any]:
        """
        Direct Felix inference - single-agent workflow through FULL Felix framework.

        This method uses ALL Felix infrastructure (not just SystemCommandManager):
        - AgentRegistry: Lifecycle tracking, performance metrics
        - CentralPost: Hub-spoke message routing (O(N) complexity)
        - SystemCommandManager: Trust-based command execution (SAFE/REVIEW/BLOCKED)
        - KnowledgeStore: Result persistence for future queries
        - TaskMemory: Pattern learning from execution
        - SynthesisEngine: Consistent output synthesis

        Simple mode = "single-agent Felix workflow" not "bypass Felix entirely"

        Args:
            message: User's input message
            knowledge_context: Knowledge brain context
            conversation_history: Previous messages
            streaming_callback: Callback for streaming chunks
            cancel_event: Optional threading.Event to signal cancellation

        Returns:
            Dict with 'content', 'confidence', 'command_results', 'meta_confidence'
        """
        start_time = time.time()
        workflow_id = f"direct_{uuid.uuid4().hex[:8]}"
        agent_id = f"felix_direct_{workflow_id[-8:]}"
        error_occurred = None

        # =========================================================
        # STEP 1: Register as agent in Felix framework
        # =========================================================
        try:
            self.central_post.register_agent_id(
                agent_id=agent_id,
                metadata={
                    'agent_type': 'direct',
                    'spawn_time': time.time(),
                    'capabilities': {'direct_inference': True, 'command_execution': True},
                    'workflow_id': workflow_id
                }
            )
            logger.info(f"Registered direct agent: {agent_id}")
        except Exception as e:
            logger.warning(f"Agent registration failed (continuing): {e}")

        try:
            # Build system prompt with Felix identity
            system_prompt = self.identity_prompt

            # Add knowledge context if available
            if knowledge_context:
                system_prompt += f"\n\n<knowledge_context>\nThe following information is relevant to this query:\n\n{knowledge_context}\n</knowledge_context>"

            # Build initial user prompt with history
            if conversation_history:
                # Check if history is large enough to warrant compression
                history_size = sum(len(m.get('content', '')) for m in conversation_history)
                if history_size > 3000 and self.central_post.context_compressor:
                    # Use compression for large histories
                    history_text = self._compress_history(conversation_history)
                    logger.debug(f"Compressed conversation history from {history_size} chars")
                else:
                    history_text = self._format_history(conversation_history)
                base_prompt = f"{history_text}\n\nCURRENT MESSAGE:\nUSER: {message}"
            else:
                base_prompt = message

            # Track command execution for context enrichment
            command_results = []
            max_iterations = 5  # Safety limit for command loops
            current_prompt = base_prompt
            content = ""

            for iteration in range(max_iterations):
                # Check for cancellation at start of each iteration
                if cancel_event and cancel_event.is_set():
                    logger.info("Direct mode cancelled by user")
                    break

                logger.debug(f"Direct mode iteration {iteration + 1}/{max_iterations}")

                # Get LLM response with streaming for real-time feedback
                # For simple queries, this IS the final output (no synthesis needed)
                # For command queries, synthesis will integrate results after
                content = self._get_llm_response(
                    system_prompt=system_prompt,
                    user_prompt=current_prompt,
                    streaming_callback=streaming_callback,  # Stream raw Felix output
                    cancel_event=cancel_event
                )

                # Check for cancellation after LLM response
                if cancel_event and cancel_event.is_set():
                    logger.info("Direct mode cancelled after LLM response")
                    break

                # =========================================================
                # STEP 2: Queue response as Message through CentralPost
                # =========================================================
                try:
                    msg = Message(
                        sender_id=agent_id,
                        message_type=MessageType.STATUS_UPDATE,
                        content={
                            'response': content,  # Full content for synthesis
                            'iteration': iteration,
                            'workflow_id': workflow_id
                        },
                        timestamp=time.time()
                    )
                    self.central_post.queue_message(msg)
                    # Process message so synthesis engine can access it
                    self.central_post.process_next_message()
                    logger.debug(f"Queued and processed message to CentralPost: {msg.message_id}")
                except Exception as e:
                    logger.warning(f"Message queuing failed (continuing): {e}")

                # Check for SYSTEM_ACTION_NEEDED patterns
                pattern = r'SYSTEM_ACTION_NEEDED:\s*([^\n]+)'
                commands = re.findall(pattern, content, re.IGNORECASE)

                if not commands:
                    # No more commands - we're done
                    logger.debug(f"No SYSTEM_ACTION_NEEDED patterns found, completing")
                    break

                logger.info(f"Found {len(commands)} SYSTEM_ACTION_NEEDED pattern(s)")

                # =========================================================
                # STEP 3: Execute commands through SystemCommandManager
                # =========================================================
                iteration_results = []
                for cmd in commands:
                    cmd = cmd.strip()
                    logger.info(f"Executing command via Felix framework: {cmd}")

                    try:
                        # Route through the ACTUAL Felix SystemCommandManager
                        action_id = self.central_post.system_command_manager.request_system_action(
                            agent_id=agent_id,
                            command=cmd,
                            context=f"Direct mode request: {message[:100]}",
                            workflow_id=workflow_id,
                            cwd=self.central_post.project_root
                        )

                        # Wait for result (handles SAFE auto-execute, REVIEW approval, BLOCKED denial)
                        result = self.central_post.system_command_manager.wait_for_approval(
                            action_id=action_id,
                            timeout=60.0  # 1 minute timeout for commands
                        )

                        if result:
                            iteration_results.append({
                                'command': cmd,
                                'success': result.success,
                                'stdout': result.stdout,
                                'stderr': result.stderr,
                                'exit_code': result.exit_code
                            })
                            command_results.append(iteration_results[-1])
                            # Command results will be incorporated into final synthesis

                    except Exception as e:
                        logger.error(f"Command execution error: {e}")
                        iteration_results.append({
                            'command': cmd,
                            'success': False,
                            'stdout': '',
                            'stderr': str(e),
                            'exit_code': -1
                        })
                        command_results.append(iteration_results[-1])

                # Build enriched context with command results for re-invocation
                results_context = self._format_command_results(iteration_results)

                # Update prompt with command results for next iteration
                current_prompt = f"""{base_prompt}

<previous_response>
{content}
</previous_response>

<command_execution_results>
{results_context}
</command_execution_results>

Based on the command results above, please continue your response.

You MAY output another SYSTEM_ACTION_NEEDED to:
- Retry with a DIFFERENT approach if the previous command failed
- Execute additional commands needed to complete the task
- Verify results of previous operations

Only stop issuing commands when the task is genuinely complete or you need user input."""

                # Continue to next iteration (no streaming during accumulation)

            # Calculate confidence based on command success rate
            if command_results:
                success_rate = sum(1 for r in command_results if r['success']) / len(command_results)
                confidence = 0.7 + (0.2 * success_rate)  # 0.7-0.9 based on success
            else:
                confidence = 0.8  # Default for no-command responses

            # Clean content of SYSTEM_ACTION_NEEDED patterns before synthesis
            # Commands were already executed during iteration loop - don't expose directives to user
            action_pattern = r'^SYSTEM_ACTION_NEEDED:\s*[^\n]+\n?'
            content = re.sub(action_pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
            content = re.sub(r'\n\s*\n', '\n', content).strip()  # Clean up extra whitespace

            # =========================================================
            # STEP 4: Store result in KnowledgeStore for future queries
            # =========================================================
            try:
                if self.knowledge_store and content:
                    self.central_post.store_agent_result_as_knowledge(
                        agent_id=agent_id,
                        content=content,
                        confidence=confidence,
                        domain="direct_inference"
                    )
                    logger.debug(f"Stored result in KnowledgeStore")
            except Exception as e:
                logger.warning(f"KnowledgeStore storage failed (continuing): {e}")

            # =========================================================
            # STEP 5: Record in TaskMemory for pattern learning
            # =========================================================
            execution_time = time.time() - start_time
            try:
                if self.task_memory:
                    self.task_memory.record_task_execution(
                        task_description=message[:200],
                        task_type="direct_chat",
                        complexity=TaskComplexity.SIMPLE,
                        outcome=TaskOutcome.SUCCESS if not error_occurred else TaskOutcome.FAILURE,
                        duration=execution_time,
                        agents_used=[agent_id],
                        strategies_used=["direct_response"],
                        context_size=len(message),
                        error_messages=[str(error_occurred)] if error_occurred else None,
                        success_metrics={
                            'commands_run': len(command_results),
                            'confidence': confidence
                        }
                    )
                    logger.debug(f"Recorded task in TaskMemory")
            except Exception as e:
                logger.warning(f"TaskMemory recording failed (continuing): {e}")

            # Record agent performance metrics
            try:
                if hasattr(self.felix_system, 'performance_tracker') and self.felix_system.performance_tracker:
                    self.felix_system.performance_tracker.record_agent_checkpoint(
                        agent_id=agent_id,
                        agent_type='direct',
                        spawn_time=start_time,
                        checkpoint=1.0,
                        confidence=confidence,
                        tokens_used=0,
                        processing_time=execution_time,
                        depth_ratio=1.0,
                        phase='synthesis',
                        content_preview=content[:500] if content else None
                    )
                    logger.debug("Recorded performance in AgentPerformanceTracker")
            except Exception as e:
                logger.debug(f"Performance tracking failed: {e}")

            # =========================================================
            # STEP 6: Synthesis - ONLY if commands were executed
            # Simple queries already streamed raw Felix output directly
            # Synthesis is only needed to integrate command execution results
            # =========================================================
            synthesis_result = None
            meta_confidence = None

            if command_results:
                # Commands were executed - use synthesis to integrate results
                # Don't stream (content already shown), just update final response
                try:
                    synthesis_result = self.central_post.synthesize_agent_outputs(
                        task_description=message,
                        max_messages=5,
                        task_complexity="SIMPLE_FACTUAL",
                        streaming_callback=None  # Don't re-stream, content already shown
                    )
                    if synthesis_result:
                        synthesized_content = synthesis_result.get('synthesis_content', '')
                        if synthesized_content:
                            content = synthesized_content
                            confidence = synthesis_result.get('confidence', confidence)
                        meta_confidence = synthesis_result.get('meta_confidence')
                        logger.info(f"SynthesisEngine integrated command results, confidence={confidence}")
                except Exception as e:
                    logger.warning(f"SynthesisEngine failed (using raw content): {e}")
            else:
                # Simple response - raw Felix output already streamed directly
                # No synthesis needed - this is the correct behavior
                logger.debug("Simple response - raw Felix output already streamed (no synthesis needed)")

            return {
                'content': content,
                'confidence': confidence,
                'command_results': command_results if command_results else None,
                'meta_confidence': meta_confidence,
                'execution_time': execution_time
            }

        except Exception as e:
            error_occurred = str(e)
            logger.error(f"Direct mode error: {e}", exc_info=True)
            raise

        finally:
            # =========================================================
            # STEP 7: Cleanup - Deregister agent
            # =========================================================
            try:
                self.central_post.agent_registry.deregister_agent(agent_id)
                logger.debug(f"Deregistered direct agent: {agent_id}")
            except Exception as e:
                logger.warning(f"Agent deregistration failed: {e}")

    def _get_llm_response(
        self,
        system_prompt: str,
        user_prompt: str,
        streaming_callback: Optional[Callable] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> str:
        """
        Get LLM response with optional streaming.

        Args:
            system_prompt: System prompt with Felix identity
            user_prompt: User message/context
            streaming_callback: Optional callback for streaming
            cancel_event: Optional threading.Event to signal cancellation

        Returns:
            str: Complete response content
        """
        accumulated_content = []

        def on_chunk(chunk):
            """Handle streaming chunks."""
            chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
            accumulated_content.append(chunk_text)
            if streaming_callback:
                streaming_callback(chunk_text)

        try:
            if hasattr(self.lm_client, 'complete_streaming') and streaming_callback:
                response = self.lm_client.complete_streaming(
                    agent_id="felix_direct",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    callback=on_chunk,
                    batch_interval=0.1,
                    cancel_event=cancel_event
                )
                content = ''.join(accumulated_content) if accumulated_content else (
                    response.content if hasattr(response, 'content') else str(response)
                )
            else:
                response = self.lm_client.complete(
                    agent_id="felix_direct",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7
                )
                content = response.content if hasattr(response, 'content') else str(response)
                if streaming_callback:
                    streaming_callback(content)

            return content

        except Exception as e:
            logger.error(f"LLM response error: {e}")
            raise

    def _format_command_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format command results for LLM context.

        Args:
            results: List of command result dictionaries

        Returns:
            str: Formatted results string
        """
        if not results:
            return "No commands were executed."

        formatted = []
        for r in results:
            status = "SUCCESS" if r['success'] else "FAILED"
            entry = f"Command: {r['command']}\nStatus: {status} (exit code: {r['exit_code']})"

            if r['stdout']:
                # Truncate very long output
                stdout = r['stdout'][:2000] + "..." if len(r['stdout']) > 2000 else r['stdout']
                entry += f"\nOutput:\n{stdout}"

            if r['stderr']:
                stderr = r['stderr'][:500] + "..." if len(r['stderr']) > 500 else r['stderr']
                entry += f"\nErrors:\n{stderr}"

            formatted.append(entry)

        return "\n\n---\n\n".join(formatted)

    def _full_orchestration(
        self,
        message: str,
        knowledge_context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        streaming_callback: Optional[Callable] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Dict[str, Any]:
        """
        Full multi-agent workflow orchestration.

        This uses the complete Felix framework with specialist agents,
        helix progression, and synthesis. The final output still comes
        FROM Felix (via SynthesisEngine).

        Args:
            message: User's input message
            knowledge_context: Knowledge brain context
            conversation_history: Previous messages
            streaming_callback: Callback for agent thinking steps
            cancel_event: Optional threading.Event to signal cancellation

        Returns:
            Dict with 'content', 'confidence', 'thinking_steps'
        """
        # Import workflow function
        from src.workflows.felix_workflow import run_felix_workflow

        thinking_steps = []

        # Wrap callback to collect thinking steps
        def workflow_callback(agent_name: str, chunk: str):
            """Handle workflow agent outputs."""
            thinking_steps.append({
                'agent': agent_name,
                'content': chunk,
                'timestamp': time.time()
            })
            if streaming_callback:
                streaming_callback(agent_name, chunk)

        # Run full workflow
        result = run_felix_workflow(
            felix_system=self.felix_system,
            task_input=message,
            streaming_callback=workflow_callback,
            cancel_event=cancel_event,
            approval_callback=approval_callback,
            approval_threshold=approval_threshold
        )

        # Extract results from centralpost_synthesis dict
        # The synthesis is a dict with 'synthesis_content', 'confidence', etc.
        synthesis = result.get('centralpost_synthesis') or {}
        if isinstance(synthesis, dict):
            content = synthesis.get('synthesis_content', '')
            confidence = synthesis.get('confidence', 0.0)
        else:
            # Fallback if synthesis is somehow a string
            content = str(synthesis) if synthesis else ''
            confidence = 0.0

        # Fallback: if no synthesis, try to get best agent response
        if not content and result.get('llm_responses'):
            best_response = max(result['llm_responses'], key=lambda r: r.get('confidence', 0.0))
            content = best_response.get('response', '')
            confidence = best_response.get('confidence', 0.0)
            logger.warning("No synthesis available, using best agent response as fallback")

        return {
            'content': content,
            'confidence': confidence,
            'thinking_steps': thinking_steps
        }

    def _format_history(self, history: List[Dict[str, str]], limit: int = 10) -> str:
        """
        Format conversation history for context.

        Args:
            history: List of message dicts with 'role' and 'content'
            limit: Maximum messages to include

        Returns:
            str: Formatted history string
        """
        if not history:
            return ""

        # Take last N messages
        recent = history[-limit:] if len(history) > limit else history

        formatted = ["CONVERSATION HISTORY:"]
        for msg in recent:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _compress_history(self, history: List[Dict[str, str]], target_size: int = 2000) -> str:
        """
        Compress conversation history using ContextCompressor.

        Uses hierarchical summarization to preserve important context while
        reducing size for large conversation histories.

        Args:
            history: List of message dicts with 'role' and 'content'
            target_size: Target size in characters for compressed output

        Returns:
            str: Compressed history string
        """
        if not history:
            return ""

        try:
            # Build context dict for compressor
            context_dict = {}
            for i, msg in enumerate(history):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                context_dict[f"msg_{i}_{role}"] = content

            # Compress using central_post's context_compressor
            from src.memory.context_compression import CompressionStrategy

            compressed = self.central_post.context_compressor.compress_context(
                context=context_dict,
                target_size=target_size,
                strategy=CompressionStrategy.HIERARCHICAL_SUMMARY
            )

            # Extract and format compressed content
            if compressed and compressed.content:
                # Build formatted output from compressed content
                # Hierarchical summary returns nested dicts: {'core': {}, 'supporting': {...}, 'auxiliary': {}}
                formatted = ["CONVERSATION HISTORY (compressed):"]

                # Extract strings from the nested structure
                def extract_strings(obj):
                    """Recursively extract string values from nested dicts."""
                    if isinstance(obj, str):
                        if obj.strip():
                            return [obj]
                        return []
                    elif isinstance(obj, dict):
                        result = []
                        for v in obj.values():
                            result.extend(extract_strings(v))
                        return result
                    elif isinstance(obj, (list, tuple)):
                        result = []
                        for item in obj:
                            result.extend(extract_strings(item))
                        return result
                    return []

                extracted = extract_strings(compressed.content)
                formatted.extend(extracted)

                logger.info(
                    f"Compressed history: {compressed.original_size} -> "
                    f"{compressed.compressed_size} chars "
                    f"(ratio: {compressed.compression_ratio:.2%})"
                )
                return "\n".join(formatted)

        except Exception as e:
            logger.warning(f"History compression failed, using standard format: {e}")

        # Fallback to standard formatting with limit
        return self._format_history(history, limit=5)
