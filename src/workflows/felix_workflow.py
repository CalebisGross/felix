"""
Felix Framework Workflow Implementation

This module implements proper workflows using the Felix framework architecture:
- CentralPost for O(N) hub-spoke communication
- AgentFactory for intelligent agent spawning
- Knowledge store for persistent memory
- Dynamic spawning based on confidence monitoring
- Helix-based agent progression

This is the proper integration, as opposed to linear_pipeline.py which
is a comparison baseline for benchmarking purposes.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable

# Import collaborative context builder for agent collaboration
from .context_builder import CollaborativeContextBuilder

# Import context compression for managing growing collaborative context
from src.memory.context_compression import (
    ContextCompressor,
    CompressionConfig,
    CompressionStrategy,
    CompressionLevel
)

# Use module-specific logger that propagates to GUI
# IMPORTANT: Use 'felix_workflows' to match GUI logger configuration
logger = logging.getLogger('felix_workflows')
logger.setLevel(logging.INFO)
logger.propagate = True  # Ensure logs reach GUI handlers


def _classify_task_complexity(task_input: str) -> str:
    """
    Classify task complexity to optimize workflow execution.

    Args:
        task_input: The task description from user

    Returns:
        Task complexity: "SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX"
    """
    import re

    task_lower = task_input.lower()

    # Simple factual patterns that can be answered quickly with web search
    simple_patterns = [
        r'\b(what|when|who|where)\s+(is|are|was|were)\s+(the\s+)?current',
        r'\bwhat\s+time\b',
        r'\bwhat\s+date\b',
        r'\btoday\'?s?\s+(date|time)',
        r'\bcurrent\s+(time|date|datetime)',
        r'\bwho\s+(won|is|was)\b',
        r'\bwhen\s+(did|is|was)\b',
        r'\bhow\s+many\b.*\b(now|current|today)',
        r'\blatest\s+(news|update)\b',
    ]

    # Check for simple factual patterns
    for pattern in simple_patterns:
        if re.search(pattern, task_lower):
            return "SIMPLE_FACTUAL"

    # Medium complexity: specific questions but may need analysis
    medium_patterns = [
        r'\bexplain\b',
        r'\bcompare\b',
        r'\bwhat\s+are\s+the\s+(benefits|advantages|disadvantages)',
        r'\bhow\s+does\b',
        r'\bhow\s+to\b',
        r'\blist\b',
        r'\bsummarize\b',
    ]

    for pattern in medium_patterns:
        if re.search(pattern, task_lower):
            return "MEDIUM"

    # Default to complex for open-ended, analytical tasks
    return "COMPLEX"


def run_felix_workflow(felix_system, task_input: str,
                       progress_callback: Optional[Callable[[str, float], None]] = None,
                       max_steps_override: Optional[int] = None,
                       parent_workflow_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a workflow using the Felix framework components.

    This properly integrates with the Felix system initialized by the GUI:
    - Uses felix_system.central_post for communication
    - Uses felix_system.agent_factory for spawning
    - Uses felix_system.agent_manager for registration
    - Uses felix_system.knowledge_store for memory
    - Uses felix_system.lm_client shared across all agents
    - Supports conversation continuity via parent_workflow_id

    Args:
        felix_system: Initialized FelixSystem instance from GUI
        task_input: Task description to process (or follow-up question if continuing)
        progress_callback: Optional callback(status_message, progress_percentage)
        max_steps_override: Optional override for max workflow steps (None = adaptive)
        parent_workflow_id: Optional ID of parent workflow to continue from

    Returns:
        Dictionary with workflow results and metadata
    """

    try:
        # Track workflow execution time for task memory
        workflow_start_time = time.time()

        logger.info("="*60)
        logger.info("FELIX WORKFLOW STARTING")
        logger.info(f"Task: {task_input}")
        logger.info(f"Streaming: {'ENABLED' if felix_system.config.enable_streaming else 'DISABLED'}")
        logger.info("="*60)

        # Get Felix components (already initialized by GUI)
        central_post = felix_system.central_post
        agent_factory = felix_system.agent_factory
        agent_manager = felix_system.agent_manager
        knowledge_store = felix_system.knowledge_store

        # Load parent workflow context if continuing conversation
        parent_context = None
        original_task_input = task_input
        if parent_workflow_id:
            logger.info("=" * 60)
            logger.info(f"CONTINUING FROM PARENT WORKFLOW #{parent_workflow_id}")
            from src.workflows.conversation_loader import ConversationContextLoader
            from src.memory.workflow_history import WorkflowHistory

            workflow_history = WorkflowHistory()
            context_loader = ConversationContextLoader(workflow_history)

            parent_result = context_loader.load_parent_workflow(parent_workflow_id)
            if parent_result:
                # Build continuation prompt that includes parent context
                task_input = context_loader.build_continuation_prompt(
                    parent_result,
                    follow_up_question=original_task_input,
                    max_context_length=1000
                )
                parent_context = parent_result
                logger.info(f"  ✓ Parent task: {parent_result['task_input'][:80]}...")
                logger.info(f"  ✓ Parent confidence: {parent_result['confidence']:.2f}")
                logger.info(f"  ✓ Follow-up: {original_task_input[:80]}...")
            else:
                logger.warning(f"  ⚠ Could not load parent workflow {parent_workflow_id}, proceeding as new workflow")
            logger.info("=" * 60)

        # Initialize context compressor for managing growing collaborative context
        compression_config = CompressionConfig(
            max_context_size=2000,  # Reasonable limit for collaborative context
            strategy=CompressionStrategy.ABSTRACTIVE_SUMMARY,  # Mentioned in CLAUDE.md (0.3 ratio)
            level=CompressionLevel.MODERATE,  # 60% compression
            relevance_threshold=0.3
        )
        context_compressor = ContextCompressor(config=compression_config)

        # Initialize collaborative context builder for agent collaboration
        # This enables agents to retrieve and build upon previous outputs
        context_builder = CollaborativeContextBuilder(
            central_post=central_post,
            knowledge_store=knowledge_store,
            context_compressor=context_compressor  # Now provides actual compression
        )
        logger.info("Initialized CollaborativeContextBuilder with compression for agent collaboration")

        if progress_callback:
            progress_callback("Initializing Felix workflow...", 0.0)

        # Create LLM task
        from src.agents.llm_agent import LLMTask
        task = LLMTask(
            task_id=f"workflow_{int(time.time()*1000)}",
            description=task_input,
            context="GUI workflow task - process through Felix agents"
        )

        # Set current task for CentralPost web search
        central_post.set_current_task(task_input)

        # Classify task complexity for optimization
        task_complexity = _classify_task_complexity(task_input)
        logger.info(f"Task complexity classification: {task_complexity}")

        # Check web search availability (Fix 5)
        logger.info("=" * 60)
        logger.info("🔍 WEB SEARCH AVAILABILITY CHECK")
        if central_post.web_search_client:
            logger.info("  ✓ Web search client: AVAILABLE")
            logger.info(f"  ✓ Provider: {getattr(central_post.web_search_client, 'provider', 'unknown')}")
            logger.info(f"  ✓ Max results: {getattr(central_post.web_search_client, 'max_results', 'unknown')}")
        else:
            logger.warning("  ✗ Web search client: NOT AVAILABLE")
            logger.warning("  → Research agents cannot access current information")
            logger.warning("  → Install: pip install ddgs")
            logger.warning("  → Enable in Settings → Web Search Configuration")
        logger.info("=" * 60)

        # Proactive web search for time/date queries (Fix 1)
        if task_complexity == "SIMPLE_FACTUAL":
            task_lower = task_input.lower()
            time_patterns = ['current', 'time', 'date', 'today', 'now', "what's the time", "what is the time"]
            is_time_query = any(pattern in task_lower for pattern in time_patterns)

            if is_time_query:
                logger.info("=" * 60)
                logger.info("⏰ TIME QUERY DETECTED - Triggering proactive web search")
                logger.info("=" * 60)

                if central_post.web_search_client:
                    logger.info("✓ Web search client available - performing search...")

                    # DIAGNOSTIC: Verify method and instance before calling
                    logger.info("=" * 60)
                    logger.info("🔬 PRE-CALL DIAGNOSTICS")
                    logger.info(f"  CentralPost type: {type(central_post).__name__}")
                    logger.info(f"  Web search client type: {type(central_post.web_search_client).__name__}")
                    logger.info(f"  _perform_web_search exists: {hasattr(central_post, '_perform_web_search')}")
                    logger.info(f"  _perform_web_search callable: {callable(getattr(central_post, '_perform_web_search', None))}")
                    logger.info(f"  Task input: '{task_input}'")
                    logger.info(f"  Task length: {len(task_input)} chars")
                    logger.info("=" * 60)

                    try:
                        # Trigger web search immediately
                        logger.info("🚀 INVOKING central_post._perform_web_search() NOW...")
                        result = central_post._perform_web_search(task_input)
                        logger.info(f"✓ Method returned: {result}")
                        logger.info("✓ Proactive web search completed - results stored in knowledge base")
                    except Exception as e:
                        logger.error(f"✗ Proactive web search failed: {e}", exc_info=True)
                else:
                    logger.warning("⚠ Web search requested but CLIENT NOT AVAILABLE!")
                    logger.warning("  Install: pip install ddgs")
                    logger.warning("  Enable in Settings → Web Search Configuration")

        # Track results
        results = {
            "task": task_input,
            "task_input": original_task_input if parent_workflow_id else task_input,  # Store original for history
            "parent_workflow_id": parent_workflow_id,  # Track parent for conversation threading
            "agents_spawned": [],
            "messages_processed": [],
            "llm_responses": [],
            "knowledge_entries": [],
            "status": "in_progress",
            "centralpost_synthesis": None  # Will store CentralPost synthesis output
        }

        # Track ALL processed messages for dynamic spawning (REAL messages, not mocks!)
        all_processed_messages = []

        # Time progression parameters with adaptive stepping
        if max_steps_override is not None:
            # User override
            total_steps = max_steps_override
            logger.info(f"Using user-specified max steps: {total_steps}")
        else:
            # Adaptive mode: adjust based on task complexity
            if task_complexity == "SIMPLE_FACTUAL":
                total_steps = 5  # Minimal steps for simple factual queries
                logger.info(f"Simple factual query detected - using minimal steps: {total_steps}")
            elif task_complexity == "MEDIUM":
                total_steps = felix_system.config.workflow_max_steps_medium
                logger.info(f"Medium complexity detected - using moderate steps: {total_steps}")
            else:
                total_steps = felix_system.config.workflow_max_steps_complex
                logger.info(f"Complex task detected - using maximum steps: {total_steps}")

        current_time = 0.0
        time_step = 1.0 / total_steps

        # Adaptive early stopping variables
        adaptive_mode = (max_steps_override is None)
        sample_period = min(5, total_steps // 2)  # Sample first 5 steps or half of total
        complexity_assessed = (task_complexity == "SIMPLE_FACTUAL")  # Skip assessment if already simple

        logger.info(f"Workflow will progress through up to {total_steps} time steps")
        if adaptive_mode and not complexity_assessed:
            logger.info(f"Adaptive mode: will assess complexity after {sample_period} steps")

        # Spawn initial research agent with fixed spawn_time=0.0
        logger.info("Creating initial research agent...")
        from src.agents.specialized_agents import ResearchAgent
        research_agent = ResearchAgent(
            agent_id="workflow_research_000",
            spawn_time=0.0,  # Fixed time, not random - ensures immediate spawning
            helix=felix_system.helix,
            llm_client=felix_system.lm_client,
            research_domain="task_analysis",
            token_budget_manager=felix_system.token_budget_manager,
            max_tokens=1200
        )

        # Register with agent_manager
        agent_manager.register_agent(research_agent)

        # CRITICAL: Create spoke connection for O(N) communication topology
        # Spoke creation handles central_post registration automatically
        registration_successful = False
        if felix_system.spoke_manager:
            spoke = felix_system.spoke_manager.create_spoke(research_agent)
            if spoke:
                logger.info(f"Created spoke connection for {research_agent.agent_id}")
                registration_successful = True
            else:
                logger.warning(f"⚠ Could not register {research_agent.agent_id} - agent cap reached")
        elif felix_system.central_post:
            # Fallback to direct registration if no spoke_manager
            connection_id = felix_system.central_post.register_agent(research_agent)
            if connection_id:
                logger.info(f"Registered {research_agent.agent_id} directly with central_post")
                registration_successful = True
            else:
                logger.warning(f"⚠ Could not register {research_agent.agent_id} - agent cap reached")

        # Only proceed if registration succeeded
        if not registration_successful:
            logger.error(f"Failed to register initial research agent - cannot proceed with workflow")
            return {
                "status": "failed",
                "error": "Could not register initial agent (agent cap reached)",
                "agents_spawned": [],
                "llm_responses": []
            }

        results["agents_spawned"].append(research_agent.agent_id)
        logger.info(f"Created research agent: {research_agent.agent_id} (spawn_time=0.0)")

        if progress_callback:
            progress_callback(f"Spawned {research_agent.agent_type} agent", 10.0)

        # Get all agents (will accumulate as we spawn more)
        active_agents = [research_agent]

        # Progress through time steps
        for step in range(total_steps):
            current_time = step * time_step
            progress_pct = (step / total_steps) * 100

            logger.info(f"\n--- Time Step {step}/{total_steps} (t={current_time:.2f}) ---")

            if progress_callback:
                progress_callback(f"Processing time step {step+1}/{total_steps}", progress_pct)

            # Process each agent that can spawn at this time
            for agent in active_agents:
                # Handle initial spawn
                if agent.can_spawn(current_time) and agent.state.value == "waiting":
                    logger.info(f"Agent {agent.agent_id} spawning at t={current_time:.2f}")
                    try:
                        # Spawn agent (sets progress=0.0, state=ACTIVE)
                        agent.spawn(current_time, task)
                        # Initialize checkpoint tracking
                        agent._last_checkpoint_processed = -1
                        logger.info(f"✓ Agent {agent.agent_id} spawned at checkpoint 0.0")
                    except Exception as e:
                        logger.error(f"Error spawning agent {agent.agent_id}: {e}", exc_info=True)
                        continue

                # Check for checkpoint crossing (continuous communication as agent descends)
                if hasattr(agent, 'should_process_at_checkpoint') and agent.should_process_at_checkpoint(current_time):
                    checkpoint = agent.get_current_checkpoint()
                    logger.info(f"📍 Agent {agent.agent_id} crossed checkpoint {checkpoint:.1f} (progress={agent.progress:.3f})")

                    try:
                        # Build enriched collaborative context for this checkpoint
                        # This retrieves previous agent outputs so the agent can build upon them
                        try:
                            enriched_context = context_builder.build_agent_context(
                                original_task=task_input,
                                agent_type=agent.agent_type,
                                agent_id=agent.agent_id,
                                current_time=current_time,
                                max_context_tokens=agent.max_tokens,
                                message_limit=3  # Reduced from 10 to fit within token budget
                            )

                            # Update task with enriched context for collaborative processing
                            task.context_history = enriched_context.context_history
                            task.knowledge_entries = enriched_context.knowledge_entries
                            logger.info(f"  ✓ Using collaborative context with {len(enriched_context.context_history)} previous outputs")

                        except Exception as ctx_error:
                            logger.warning(f"  ⚠ Collaborative context building failed, falling back to non-collaborative mode: {ctx_error}")
                            # Clear any partial context and continue without collaboration
                            task.context_history = None
                            task.knowledge_entries = None

                        # Process task through LLM at this checkpoint (with or without collaborative context)
                        # Pass central_post and streaming flag for real-time communication
                        result = agent.process_task_with_llm(
                            task,
                            current_time,
                            central_post=central_post,
                            enable_streaming=felix_system.config.enable_streaming
                        )

                        # Mark checkpoint as processed
                        agent.mark_checkpoint_processed()

                        logger.info(f"  ✓ Agent {agent.agent_id} completed checkpoint {checkpoint:.1f}: confidence={result.confidence:.2f}, stage={agent.processing_stage}")

                        # Store agent output with full metrics for GUI display
                        agent_manager.store_agent_output(
                            agent_id=agent.agent_id,
                            result=result
                        )

                        # Share result via CentralPost
                        message = agent.share_result_to_central(result)
                        central_post.queue_message(message)
                        results["messages_processed"].append(message.message_id)

                        # Track REAL message for dynamic spawning analysis
                        all_processed_messages.append(message)

                        # Store in knowledge base
                        if knowledge_store:
                            central_post.store_agent_result_as_knowledge(
                                agent_id=agent.agent_id,
                                content=result.content,
                                confidence=result.confidence,
                                domain="workflow_task"
                            )
                            results["knowledge_entries"].append(agent.agent_id)

                        # Track LLM response (for GUI display)
                        results["llm_responses"].append({
                            "agent_id": agent.agent_id,
                            "agent_type": agent.agent_type,
                            "confidence": result.confidence,
                            "response": result.content,
                            "time": current_time,
                            "checkpoint": checkpoint,  # NEW: Track which checkpoint this was
                            "progress": agent.progress   # NEW: Track agent progress
                        })

                        # ADAPTIVE THRESHOLD: Assess knowledge quality after research agent checkpoints
                        if agent.agent_type == "research" and knowledge_store:
                            try:
                                from src.memory.knowledge_store import KnowledgeQuery, ConfidenceLevel
                                from src.workflows.truth_assessment import assess_answer_confidence, detect_contradictions, detect_query_type, QueryType

                                # Retrieve recent knowledge entries with dynamic freshness based on query type
                                import time as time_module
                                current_time_ts = time_module.time()

                                # Detect query type to determine appropriate freshness window
                                query_type = detect_query_type(task_input)
                                freshness_limits = {
                                    QueryType.TIME: 300,           # 5 minutes for time queries
                                    QueryType.DATE: 3600,          # 1 hour for date queries
                                    QueryType.CURRENT_EVENT: 1800, # 30 minutes for current events
                                    QueryType.GENERAL_FACT: 86400, # 24 hours for general facts
                                    QueryType.ANALYSIS: 86400,     # 24 hours for analysis
                                }
                                max_age = freshness_limits.get(query_type, 3600)  # Default to 1 hour
                                time_window_start = current_time_ts - max_age

                                logger.info(f"📊 Knowledge Retrieval: query_type={query_type.value}, freshness_window={max_age}s ({max_age/60:.1f} min)")

                                knowledge_entries = knowledge_store.retrieve_knowledge(
                                    KnowledgeQuery(
                                        domains=["web_search"],
                                        min_confidence=ConfidenceLevel.HIGH,
                                        time_range=(time_window_start, current_time_ts),
                                        limit=5
                                    )
                                )

                                if knowledge_entries and len(knowledge_entries) > 0:
                                    # Assess if knowledge is trustable
                                    trustable, trust_score, trust_reason = assess_answer_confidence(
                                        knowledge_entries,
                                        task_input
                                    )

                                    # Check for contradictions
                                    has_contradictions, contradiction_reason = detect_contradictions(knowledge_entries)

                                    # Log assessment
                                    logger.info(f"📊 Knowledge Quality Assessment:")
                                    logger.info(f"   Entries: {len(knowledge_entries)} HIGH confidence")
                                    logger.info(f"   Trustable: {trustable} (score: {trust_score:.2f})")
                                    logger.info(f"   Reason: {trust_reason}")
                                    if has_contradictions:
                                        logger.info(f"   ⚠ Contradictions: {contradiction_reason}")

                                    # Decide on threshold adjustment
                                    if trustable and result.confidence >= 0.60 and not has_contradictions:
                                        # Lower threshold to allow research agent to complete
                                        new_threshold = 0.60
                                        central_post.update_confidence_threshold(
                                            new_threshold,
                                            f"Trustable knowledge available ({trust_reason})"
                                        )
                                        logger.info(f"✓ Research agent can complete with trustable knowledge")
                                    elif has_contradictions and result.confidence >= 0.55:
                                        # Need Analysis agent to resolve
                                        logger.info(f"⚠ Contradictions detected - Analysis agent needed")
                                        # Keep threshold high to force more processing
                                        central_post.update_confidence_threshold(0.75, "Contradictions need resolution")
                                    elif not trustable and result.confidence >= 0.55:
                                        # Knowledge needs validation
                                        logger.info(f"📋 Knowledge needs validation - continuing helix progression")

                            except Exception as e:
                                logger.warning(f"Could not perform adaptive threshold assessment: {e}")

                        # Note: Synthesis now performed by CentralPost, not synthesis agents

                        if progress_callback:
                            progress_callback(f"{agent.agent_type} checkpoint {checkpoint:.1f}", progress_pct + 5)

                    except Exception as e:
                        logger.error(f"Error processing agent {agent.agent_id} at checkpoint {checkpoint:.1f}: {e}", exc_info=True)

            # Process pending messages in CentralPost
            messages_this_step = 0
            while central_post.has_pending_messages():
                processed_msg = central_post.process_next_message()
                if processed_msg:
                    messages_this_step += 1

            if messages_this_step > 0:
                logger.info(f"Processed {messages_this_step} messages through CentralPost")

            # Adaptive complexity assessment (only once after sample period)
            if adaptive_mode and not complexity_assessed and step >= sample_period:
                # Assess task complexity based on confidence
                if len(central_post._recent_confidences) > 0:
                    avg_confidence = sum(central_post._recent_confidences) / len(central_post._recent_confidences)

                    logger.info("="*60)
                    logger.info("ADAPTIVE COMPLEXITY ASSESSMENT")
                    logger.info(f"Sample period: {sample_period} steps completed")
                    logger.info(f"Average confidence: {avg_confidence:.2f}")

                    # Determine complexity and adjust total_steps
                    original_steps = total_steps
                    if avg_confidence >= felix_system.config.workflow_simple_threshold:
                        # Simple task - use minimal steps
                        total_steps = felix_system.config.workflow_max_steps_simple
                        complexity = "SIMPLE"
                    elif avg_confidence >= felix_system.config.workflow_medium_threshold:
                        # Medium complexity - use moderate steps
                        total_steps = felix_system.config.workflow_max_steps_medium
                        complexity = "MEDIUM"
                    else:
                        # Complex task - keep maximum steps
                        total_steps = felix_system.config.workflow_max_steps_complex
                        complexity = "COMPLEX"

                    # Adjust if we've already passed the new target
                    if step >= total_steps:
                        total_steps = step + 1  # Allow current step to complete

                    # Recalculate time_step for remaining steps
                    time_step = 1.0 / total_steps

                    logger.info(f"Detected complexity: {complexity}")
                    logger.info(f"Adjusting total steps: {original_steps} → {total_steps}")
                    logger.info(f"Remaining steps: {total_steps - step - 1}")
                    logger.info("="*60)

                    complexity_assessed = True

            # Debug: Always log spawning check status
            logger.info(f"=== STEP {step}: Dynamic Spawning Check ===")
            logger.info(f"  Condition step >= 1: {step >= 1}")
            logger.info(f"  AgentFactory exists: {agent_factory is not None}")
            logger.info(f"  enable_dynamic_spawning: {agent_factory.enable_dynamic_spawning if agent_factory else 'N/A'}")
            logger.info(f"  Processed messages count: {len(all_processed_messages)}")
            logger.info(f"  Active agents count: {len(active_agents)}")

            # Dynamic spawning after we have some results (step 1+)
            # Skip dynamic spawning for simple factual queries - just use the initial research agent
            if step >= 1 and agent_factory.enable_dynamic_spawning and task_complexity != "SIMPLE_FACTUAL":
                # Fallback: If no messages yet, spawn based on minimum team size
                if len(all_processed_messages) == 0 and len(active_agents) < 3:
                    logger.info("No messages yet - spawning additional agents to reach minimum team size")

                    # Spawn analysis agent
                    analysis_agent = agent_factory.create_analysis_agent(
                        analysis_type="comparative",
                        spawn_time_range=(current_time, current_time + 0.01)
                    )
                    agent_manager.register_agent(analysis_agent)

                    # CRITICAL: Create spoke connection (handles central_post registration)
                    registration_successful = False
                    if felix_system.spoke_manager:
                        spoke = felix_system.spoke_manager.create_spoke(analysis_agent)
                        if spoke:
                            logger.info(f"Created spoke connection for {analysis_agent.agent_id}")
                            registration_successful = True
                    elif felix_system.central_post:
                        connection_id = felix_system.central_post.register_agent(analysis_agent)
                        if connection_id:
                            logger.info(f"Registered {analysis_agent.agent_id} directly with central_post")
                            registration_successful = True

                    if registration_successful:
                        active_agents.append(analysis_agent)
                        results["agents_spawned"].append(analysis_agent.agent_id)
                        logger.info(f"  → Spawned {analysis_agent.agent_type}: {analysis_agent.agent_id}")
                    else:
                        logger.warning(f"⚠ Agent cap reached during fallback spawning - skipping {analysis_agent.agent_id}")

                    # Also spawn critic agent for minimum team
                    if len(active_agents) < 3:
                        critic_agent = agent_factory.create_critic_agent(
                            spawn_time_range=(current_time, current_time + 0.01)
                        )
                        agent_manager.register_agent(critic_agent)

                        # CRITICAL: Create spoke connection (handles central_post registration)
                        registration_successful = False
                        if felix_system.spoke_manager:
                            spoke = felix_system.spoke_manager.create_spoke(critic_agent)
                            if spoke:
                                logger.info(f"Created spoke connection for {critic_agent.agent_id}")
                                registration_successful = True
                        elif felix_system.central_post:
                            connection_id = felix_system.central_post.register_agent(critic_agent)
                            if connection_id:
                                logger.info(f"Registered {critic_agent.agent_id} directly with central_post")
                                registration_successful = True

                        if registration_successful:
                            active_agents.append(critic_agent)
                            results["agents_spawned"].append(critic_agent.agent_id)
                            logger.info(f"  → Spawned {critic_agent.agent_type}: {critic_agent.agent_id}")
                        else:
                            logger.warning(f"⚠ Agent cap reached during fallback spawning - skipping {critic_agent.agent_id}")

                    if progress_callback:
                        progress_callback(f"Spawned fallback agents", progress_pct + 2)

                # Normal spawning with messages
                elif all_processed_messages:
                    # Use REAL messages from agents (not mocks!)
                    # These contain all fields: confidence, agent_type, position_info, etc.

                    # Log spawning analysis
                    recent_confidences = [
                        msg.content.get("confidence", 0.0)
                        for msg in all_processed_messages
                    ]
                    avg_confidence = sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0.0

                    logger.info(f"\n--- Spawning check at t={current_time:.2f} ---")
                    logger.info(f"  Messages: {len(all_processed_messages)}")
                    logger.info(f"  Avg confidence: {avg_confidence:.2f} (threshold: 0.7)")
                    logger.info(f"  Current team: {len(active_agents)} agents, types: {[a.agent_type for a in active_agents]}")

                # Call assess_team_needs with REAL messages
                new_agents = agent_factory.assess_team_needs(
                    all_processed_messages,  # REAL messages with all fields!
                    current_time,
                    active_agents
                )

                if new_agents:
                    logger.info(f"✓ Spawning {len(new_agents)} new agents based on confidence/gaps")
                    for new_agent in new_agents:
                        # Register with agent_manager
                        agent_manager.register_agent(new_agent)

                        # CRITICAL: Create spoke connection for O(N) communication
                        # Spoke creation handles central_post registration automatically
                        registration_successful = False
                        if felix_system.spoke_manager:
                            spoke = felix_system.spoke_manager.create_spoke(new_agent)
                            if spoke:
                                logger.info(f"Created spoke connection for {new_agent.agent_id}")
                                registration_successful = True
                        elif felix_system.central_post:
                            connection_id = felix_system.central_post.register_agent(new_agent)
                            if connection_id:
                                logger.info(f"Registered {new_agent.agent_id} directly with central_post")
                                registration_successful = True

                        if registration_successful:
                            active_agents.append(new_agent)
                            results["agents_spawned"].append(new_agent.agent_id)
                            logger.info(f"  → {new_agent.agent_type} agent: {new_agent.agent_id}")

                            if progress_callback:
                                progress_callback(f"Spawned {new_agent.agent_type} agent", progress_pct + 2)
                        else:
                            # Agent cap reached - stop trying to spawn more agents
                            logger.warning(f"⚠ Agent cap reached - cannot spawn {new_agent.agent_id}")
                            logger.info(f"→ Continuing with {len(active_agents)} existing agents to complete helical progression")
                            break
                else:
                    logger.info(f"✗ No spawning needed (sufficient confidence or at capacity)")

            # Advance simulation time
            felix_system.advance_time(time_step)

            # Check if we've reached confident consensus (early exit)
            if step >= 8:  # After minimum processing time (extended to ensure synthesis phase)
                if all_processed_messages:
                    # Get recent confidence scores
                    recent_confidence = [
                        msg.content.get("confidence", 0.0)
                        for msg in all_processed_messages[-5:]  # Last 5 messages
                    ]
                    avg_recent = sum(recent_confidence) / len(recent_confidence) if recent_confidence else 0.0

                    # Debug: Always log consensus check with depth-awareness
                    logger.info(f"--- Consensus Check (step {step}) ---")
                    logger.info(f"  Recent confidence scores: {[f'{c:.2f}' for c in recent_confidence]}")
                    logger.info(f"  Average: {avg_recent:.2f} (threshold: 0.80)")
                    logger.info(f"  Team size: {len(results['agents_spawned'])} (minimum: 3)")

                    # CentralPost synthesis status
                    if results['centralpost_synthesis']:
                        logger.info(f"  CentralPost Synthesis: Complete (confidence: {results['centralpost_synthesis']['confidence']:.2f})")
                    else:
                        logger.info(f"  CentralPost Synthesis: Pending (will trigger at consensus)")

                    # Trigger CentralPost synthesis when confidence threshold reached
                    if avg_recent >= 0.80 and len(results["agents_spawned"]) >= 3:
                        logger.info(f"\n✓ Confidence threshold reached - triggering CentralPost synthesis...")
                        logger.info(f"  Confidence: {avg_recent:.2f} >= threshold (0.80)")
                        logger.info(f"  Team size: {len(results['agents_spawned'])} >= minimum (3)")

                        try:
                            synthesis_result = central_post.synthesize_agent_outputs(
                                task_description=task_input,
                                max_messages=20,
                                task_complexity=task_complexity
                            )
                            results["centralpost_synthesis"] = synthesis_result
                            logger.info(f"✓ CentralPost synthesis complete!")
                            logger.info(f"  Synthesis confidence: {synthesis_result['confidence']:.2f}")
                            logger.info(f"  Agents synthesized: {synthesis_result['agents_synthesized']}")
                            logger.info(f"  Tokens used: {synthesis_result['tokens_used']} / {synthesis_result['max_tokens']}")

                            if progress_callback:
                                progress_callback(f"Synthesis complete!", 100.0)
                            break  # Exit after successful synthesis

                        except Exception as e:
                            logger.error(f"✗ CentralPost synthesis failed: {e}")
                            logger.info(f"  Continuing workflow to gather more agent outputs...")
                            # Don't break - continue to gather more agent outputs

                    # Note: Old synthesis agent spawning logic removed - CentralPost handles synthesis now

        # Final processing
        logger.info("\n--- Final Processing ---")
        if progress_callback:
            progress_callback("Finalizing results...", 95.0)

        # Process any remaining messages
        final_msg_count = 0
        while central_post.has_pending_messages():
            central_post.process_next_message()
            final_msg_count += 1

        if final_msg_count > 0:
            logger.info(f"Processed {final_msg_count} final messages")

        # Compile final results
        results["status"] = "completed"
        results["completed_agents"] = len(results["llm_responses"])  # Agents that processed
        results["total_agents"] = len(active_agents)

        # Ensure CentralPost synthesis has been performed
        if results["centralpost_synthesis"] is None:
            logger.info("Triggering final CentralPost synthesis...")
            try:
                synthesis_result = central_post.synthesize_agent_outputs(
                    task_description=task_input,
                    max_messages=20,
                    task_complexity=task_complexity
                )
                results["centralpost_synthesis"] = synthesis_result
                logger.info(f"✓ Final CentralPost synthesis complete")
            except Exception as e:
                logger.error(f"✗ CentralPost synthesis failed: {e}")
                # Fallback to highest confidence agent output
                if results["llm_responses"]:
                    highest_confidence_response = max(results["llm_responses"], key=lambda r: r.get("confidence", 0.0))
                    results["centralpost_synthesis"] = {
                        "synthesis_content": highest_confidence_response["response"],
                        "confidence": highest_confidence_response["confidence"],
                        "agents_synthesized": 1,
                        "fallback": True,
                        "fallback_agent_id": highest_confidence_response["agent_id"]
                    }
                    logger.warning(f"⚠ WARNING: Using fallback synthesis from {highest_confidence_response['agent_id']}")
                else:
                    logger.error(f"✗ ERROR: No agent outputs available for synthesis!")

        logger.info("="*60)
        logger.info("FELIX WORKFLOW COMPLETED")
        logger.info(f"Agents spawned: {len(results['agents_spawned'])}")
        logger.info(f"Messages processed: {len(results['messages_processed'])}")
        logger.info(f"Knowledge entries: {len(results['knowledge_entries'])}")
        logger.info(f"LLM responses: {len(results['llm_responses'])}")
        logger.info("="*60)

        if progress_callback:
            progress_callback("Workflow completed successfully!", 100.0)

        # Record task execution in task memory for pattern learning
        if felix_system.task_memory:
            try:
                from src.memory.task_memory import TaskComplexity, TaskOutcome

                # Calculate workflow duration
                workflow_duration = time.time() - workflow_start_time

                # Determine outcome based on results
                outcome = TaskOutcome.SUCCESS

                # Calculate average confidence
                avg_confidence = 0.0
                if results.get("llm_responses"):
                    confidences = [r.get("confidence", 0.0) for r in results["llm_responses"]]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # Determine complexity based on agents spawned
                agent_count = len(results["agents_spawned"])
                if agent_count <= 2:
                    complexity = TaskComplexity.SIMPLE
                elif agent_count <= 4:
                    complexity = TaskComplexity.MODERATE
                elif agent_count <= 7:
                    complexity = TaskComplexity.COMPLEX
                else:
                    complexity = TaskComplexity.VERY_COMPLEX

                # Get agent types used
                agents_used = [agent.agent_type for agent in active_agents] if active_agents else []

                # Record execution
                felix_system.task_memory.record_task_execution(
                    task_description=task_input,
                    task_type="workflow_task",
                    complexity=complexity,
                    outcome=outcome,
                    duration=workflow_duration,
                    agents_used=agents_used,
                    strategies_used=["felix_workflow"],
                    context_size=len(task_input),
                    error_messages=[],
                    success_metrics={"avg_confidence": avg_confidence, "agent_count": agent_count}
                )
                logger.info("Task execution recorded in task memory")
            except Exception as task_memory_error:
                # Don't fail workflow if task memory recording fails
                logger.warning(f"Could not record task execution in task memory: {task_memory_error}")

        return results

    except Exception as e:
        logger.error(f"Felix workflow error: {e}", exc_info=True)

        if progress_callback:
            progress_callback(f"Workflow failed: {str(e)}", 0.0)

        return {
            "status": "failed",
            "error": str(e),
            "agents_spawned": results.get("agents_spawned", []) if 'results' in locals() else [],
            "llm_responses": results.get("llm_responses", []) if 'results' in locals() else []
        }
