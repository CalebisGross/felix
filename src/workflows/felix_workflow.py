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
from datetime import datetime
from typing import Dict, Any, Optional, Callable

# Import collaborative context builder for agent collaboration
from .context_builder import CollaborativeContextBuilder

# Import concept registry for terminology consistency
from .concept_registry import ConceptRegistry

# Import learning systems for pattern recommendations and optimization
from src.learning import RecommendationEngine

# Import context compression for managing growing collaborative context
from src.memory.context_compression import (
    ContextCompressor,
    CompressionConfig,
    CompressionStrategy,
    CompressionLevel
)

# Import TaskComplexity for task classification
from src.memory.task_memory import TaskComplexity

# Use module-specific logger that propagates to GUI
# IMPORTANT: Use 'felix_workflows' to match GUI logger configuration
logger = logging.getLogger('felix_workflows')
logger.setLevel(logging.INFO)
logger.propagate = True  # Ensure logs reach GUI handlers


def _map_synthesis_complexity_to_task_complexity(synthesis_complexity: str) -> TaskComplexity:
    """
    Map SynthesisEngine complexity classification to TaskComplexity enum.

    SynthesisEngine returns: "SIMPLE_FACTUAL", "MEDIUM", "COMPLEX"
    TaskMemory expects: TaskComplexity.SIMPLE, .MODERATE, .COMPLEX, .VERY_COMPLEX

    Args:
        synthesis_complexity: String complexity from SynthesisEngine.classify_task_complexity()

    Returns:
        TaskComplexity enum value
    """
    mapping = {
        "SIMPLE_FACTUAL": TaskComplexity.SIMPLE,
        "SIMPLE": TaskComplexity.SIMPLE,
        "MEDIUM": TaskComplexity.MODERATE,
        "MODERATE": TaskComplexity.MODERATE,
        "COMPLEX": TaskComplexity.COMPLEX,
        "VERY_COMPLEX": TaskComplexity.VERY_COMPLEX
    }
    return mapping.get(synthesis_complexity.upper(), TaskComplexity.MODERATE)


def _store_file_discoveries(result_content: str, knowledge_store, agent_id: str = "system", task_id: str = "discovery") -> int:
    """
    Extract and store file path discoveries from command results.

    When agents use `find` commands, this function extracts the discovered paths
    and stores them in the knowledge store for future reuse. This enables
    meta-learning: the system remembers where files are located.

    Args:
        result_content: The agent's output containing find command results
        knowledge_store: KnowledgeStore instance for persistence
        agent_id: ID of the agent that made the discovery
        task_id: ID of the task context

    Returns:
        Number of file discoveries stored
    """
    import re
    import os
    from src.memory.knowledge_store import KnowledgeType, ConfidenceLevel

    if not knowledge_store:
        return 0

    # Pattern matches find command output: ./path/to/file.ext
    # Also matches relative paths without ./ prefix
    find_results = re.findall(r'(?:^|\s)(\.?/?[\w/.-]+\.(?:py|js|ts|java|cpp|c|h|go|rs|rb|txt|md|json|yaml|yml|xml|html|css|sh))(?:\s|$)', result_content, re.MULTILINE)

    # Deduplicate paths in case regex matches same file multiple times
    unique_paths = list(set(p.strip() for p in find_results if p.strip() and len(p.strip()) >= 3))

    stored_count = 0
    for path in unique_paths:
        filename = os.path.basename(path)
        directory = os.path.dirname(path)

        # Check if we already have this exact path stored
        from src.memory.knowledge_store import KnowledgeQuery
        existing = knowledge_store.retrieve_knowledge(
            KnowledgeQuery(
                domains=["file_locations"],
                content_keywords=[filename],
                limit=5
            )
        )

        # Skip if already stored
        already_exists = any(
            e.content.get('full_path') == path
            for e in existing
        )
        if already_exists:
            continue

        try:
            # Build tags list, filtering out empty strings and duplicates
            parent_dir = path.split('/')[-2] if '/' in path and len(path.split('/')) > 1 else ''
            tags = list(set(t for t in [filename, directory, parent_dir] if t))

            entry_id = knowledge_store.store_knowledge(
                knowledge_type=KnowledgeType.FILE_LOCATION,
                content={
                    'filename': filename,
                    'full_path': path,
                    'directory': directory,
                    'discovered_by': agent_id,
                    'task_context': task_id
                },
                confidence_level=ConfidenceLevel.VERIFIED,  # Found paths are verified
                source_agent=agent_id,
                domain="file_locations",
                tags=tags
            )

            if entry_id:
                logger.info(f"  📁 Learned file location: {filename} → {path}")
                stored_count += 1

        except Exception as e:
            logger.debug(f"Failed to store file discovery {path}: {e}")

    return stored_count


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

        # Track spawned agents for cleanup in finally block
        spawned_agent_ids = []

        # Generate unique workflow ID for approval rule scoping
        import uuid
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"

        logger.info("="*60)
        logger.info("FELIX WORKFLOW STARTING")
        logger.info(f"Task: {task_input}")
        logger.info(f"Workflow ID: {workflow_id}")
        logger.info(f"Streaming: {'ENABLED' if felix_system.config.enable_streaming else 'DISABLED'}")
        logger.info("="*60)

        # Get Felix components (already initialized by GUI)
        central_post = felix_system.central_post
        agent_factory = felix_system.agent_factory
        agent_manager = felix_system.agent_manager
        knowledge_store = felix_system.knowledge_store

        # Set workflow ID in CentralPost for approval scoping
        central_post.set_current_workflow(workflow_id)

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

        # Initialize ConceptRegistry for terminology consistency (Phase 1.4)
        concept_registry = ConceptRegistry(workflow_id=workflow_id)
        logger.info("Initialized ConceptRegistry for workflow-scoped concept tracking")

        # Initialize RecommendationEngine for learning-based optimization
        recommendation_engine = None
        if felix_system.task_memory and felix_system.config.enable_learning:
            try:
                recommendation_engine = RecommendationEngine(
                    task_memory=felix_system.task_memory,
                    enable_auto_apply=felix_system.config.learning_auto_apply
                )
                logger.info("✓ RecommendationEngine initialized - learning enabled")
            except Exception as e:
                logger.warning(f"Could not initialize RecommendationEngine: {e}")
                recommendation_engine = None
        else:
            if not felix_system.config.enable_learning:
                logger.info("Learning disabled in config - skipping RecommendationEngine")

        if progress_callback:
            progress_callback("Initializing Felix workflow...", 0.0)

        # Create LLM task with current date/time context
        from src.agents.llm_agent import LLMTask
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        if not current_datetime.endswith(('UTC', 'EST', 'PST', 'CST', 'MST')):
            # If timezone abbreviation is empty, add local time indicator
            current_datetime = f"{current_datetime.strip()} (local time)"

        task = LLMTask(
            task_id=f"workflow_{int(time.time()*1000)}",
            description=task_input,
            context=f"Current date/time: {current_datetime}\n\nGUI workflow task - process through Felix agents"
        )

        # Set current task for CentralPost web search
        central_post.set_current_task(task_input)

        # Classify task complexity for optimization (using SynthesisEngine)
        task_complexity = central_post.synthesis_engine.classify_task_complexity(task_input)
        logger.info(f"Task complexity classification: {task_complexity}")

        # Store complexity in task metadata for prompt pipeline to use
        # Defensive check: handle both LLMTask objects and dict representations
        if hasattr(task, 'metadata'):
            task.metadata['complexity'] = task_complexity
        elif isinstance(task, dict):
            if 'metadata' not in task:
                task['metadata'] = {}
            task['metadata']['complexity'] = task_complexity

        # Classify tool requirements for conditional tool instruction injection
        tool_requirements = central_post.synthesis_engine.classify_tool_requirements(task_input)
        logger.info(f"Tool requirements classification: file_ops={tool_requirements['needs_file_ops']}, "
                   f"web_search={tool_requirements['needs_web_search']}, "
                   f"system_commands={tool_requirements['needs_system_commands']}")

        # Classify task type for meta-learning differentiation (Phase 2 - Knowledge Gap Cartography)
        # Different task types benefit from different knowledge, enabling meta-learning boost
        classified_task_type = central_post.synthesis_engine.classify_task_type(task_input)
        logger.info(f"Task type classification: {classified_task_type}")

        # Phase 7: Analyze knowledge coverage for epistemic self-awareness
        # This identifies knowledge gaps BEFORE workflow execution
        coverage_report = None
        try:
            from src.knowledge.coverage_analyzer import KnowledgeCoverageAnalyzer
            if knowledge_store:
                coverage_analyzer = KnowledgeCoverageAnalyzer(knowledge_store)
                coverage_report = coverage_analyzer.analyze_coverage(task_input)

                logger.info("=" * 60)
                logger.info("📊 KNOWLEDGE COVERAGE ANALYSIS")
                logger.info(f"  Overall coverage: {coverage_report.overall_coverage_score:.2f}")
                if coverage_report.covered_domains:
                    logger.info(f"  ✓ Covered: {', '.join(coverage_report.covered_domains)}")
                if coverage_report.weak_domains:
                    logger.info(f"  ⚠ Weak: {', '.join(coverage_report.weak_domains)}")
                if coverage_report.missing_domains:
                    logger.info(f"  ✗ Missing: {', '.join(coverage_report.missing_domains)}")
                if coverage_report.gap_summary:
                    logger.info(f"  {coverage_report.gap_summary}")
                logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Coverage analysis failed (continuing without): {e}")

        # Phase 3.2: Initialize failure recovery manager
        from src.workflows.failure_recovery import FailureRecoveryManager, FailureType
        failure_recovery = FailureRecoveryManager()
        logger.info("✓ Failure recovery system initialized")

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
                        logger.info("🚀 INVOKING central_post.perform_web_search() NOW...")
                        result = central_post.perform_web_search(task_input)
                        logger.info(f"✓ Method returned: {result}")
                        logger.info("✓ Proactive web search completed - results stored in knowledge base")
                    except Exception as e:
                        logger.error(f"✗ Proactive web search failed: {e}", exc_info=True)
                else:
                    logger.warning("⚠ Web search requested but CLIENT NOT AVAILABLE!")
                    logger.warning("  Install: pip install ddgs")
                    logger.warning("  Enable in Settings → Web Search Configuration")

        # Fast path for greetings - respond immediately without agents
        import re
        if re.match(r'^\s*(hello|hi|hey|greetings?|howdy|yo)\s*[!.?]*\s*$', task_input.lower().strip()):
            logger.info("=" * 60)
            logger.info("GREETING DETECTED - Fast path (zero agents)")
            logger.info("=" * 60)

            greeting_response = "Hello! How can I help you today?"

            return {
                "status": "completed",
                "task": task_input,
                "task_input": task_input,
                "workflow_id": workflow_id,
                "task_complexity": task_complexity,
                "centralpost_synthesis": {
                    "synthesis_content": greeting_response,
                    "confidence": 1.0,
                    "complexity": task_complexity
                },
                "agents_spawned": [],
                "llm_responses": [],
                "steps_executed": 0,
                "processing_time": time.time() - workflow_start_time,
                "knowledge_entries": [],
                "knowledge_entry_ids": []
            }

        # Track results
        results = {
            "task": task_input,
            "task_input": original_task_input if parent_workflow_id else task_input,  # Store original for history
            "parent_workflow_id": parent_workflow_id,  # Track parent for conversation threading
            "agents_spawned": [],
            "messages_processed": [],
            "llm_responses": [],
            "knowledge_entries": [],
            "knowledge_entry_ids": [],  # Track actual knowledge IDs for meta-learning
            "status": "in_progress",
            "centralpost_synthesis": None  # Will store CentralPost synthesis output
        }

        # Track reasoning evaluations from CriticAgent for weighted synthesis (Phase 6)
        # Maps agent_id -> reasoning evaluation dict from CriticAgent.evaluate_reasoning_process()
        reasoning_evals = {}

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

        # Get pre-workflow recommendations from learning systems
        unified_recommendation = None
        recommendation_id = None
        learned_thresholds = {}
        if recommendation_engine:
            logger.info("=" * 60)
            logger.info("🧠 LEARNING SYSTEM - Pre-Workflow Recommendations")
            try:
                # Convert synthesis complexity string to TaskComplexity enum
                task_complexity_enum = _map_synthesis_complexity_to_task_complexity(task_complexity)
                unified_recommendation = recommendation_engine.get_pre_workflow_recommendations(
                    task_description=task_input,
                    task_type=classified_task_type,
                    task_complexity=task_complexity_enum
                )

                if unified_recommendation:
                    logger.info(f"  {unified_recommendation.recommendation_summary}")

                    # Store recommendation ID for outcome tracking
                    if unified_recommendation.workflow_recommendation:
                        recommendation_id = unified_recommendation.workflow_recommendation.recommendation_id
                        # Store recommendation for later outcome recording
                        recommendation_engine.pattern_learner.store_recommendation(
                            unified_recommendation.workflow_recommendation
                        )

                    # Extract learned thresholds for use during workflow
                    learned_thresholds = unified_recommendation.suggested_thresholds
                    logger.info(f"  Learned thresholds: {', '.join([f'{k}={v:.2f}' for k,v in learned_thresholds.items()])}")

                    # Apply high-confidence recommendations if enabled
                    if unified_recommendation.should_auto_apply:
                        applied = recommendation_engine.apply_high_confidence_recommendations(
                            unified_rec=unified_recommendation,
                            agent_factory=agent_factory,
                            config=felix_system.config
                        )
                        if applied.get('auto_applied'):
                            logger.info(f"  ✓ Auto-applied: {', '.join(applied.get('changes', []))}")
                else:
                    logger.info("  No learned patterns available - using standard configuration")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to get recommendations: {e}")
            logger.info("=" * 60)

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
            max_tokens=16000,  # Generous budget for 50K context window
            prompt_manager=felix_system.prompt_manager,
            prompt_optimizer=felix_system.prompt_optimizer
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
        spawned_agent_ids.append(research_agent.agent_id)  # Track for cleanup
        logger.info(f"Created research agent: {research_agent.agent_id} (spawn_time=0.0)")

        if progress_callback:
            progress_callback(f"Spawned {research_agent.agent_type} agent", 10.0)

        # Get all agents (will accumulate as we spawn more)
        active_agents = [research_agent]

        # Progress through time steps
        # Store original total for consistent progress calculation
        original_total_steps = total_steps
        for step in range(total_steps):
            current_time = step * time_step
            # Use original total for percentage, clamp to 95% to reserve 100% for completion
            progress_pct = min(95, (step / original_total_steps) * 100)

            logger.info(f"\n--- Time Step {step}/{original_total_steps} (t={current_time:.2f}) ---")

            if progress_callback:
                progress_callback(f"Processing time step {step+1}/{original_total_steps}", progress_pct)

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
                                max_context_tokens=40000,  # Context window budget (not agent response limit)
                                message_limit=3,  # Reduced from 10 to fit within token budget
                                tool_requirements=tool_requirements  # Pass tool requirements for conditional injection
                            )

                            # Update task with enriched context for collaborative processing
                            task.context_history = enriched_context.context_history
                            task.knowledge_entries = enriched_context.knowledge_entries
                            task.tool_instructions = enriched_context.tool_instructions  # Include conditional tool instructions
                            task.context_inventory = enriched_context.context_inventory  # NEW: Include context inventory for agent comprehension
                            task.existing_concepts = enriched_context.existing_concepts  # NEW: Include existing concepts for terminology consistency
                            logger.info(f"  ✓ Using collaborative context with {len(enriched_context.context_history)} previous outputs")
                            if enriched_context.tool_instructions:
                                logger.info(f"  ✓ Tool instructions included ({len(enriched_context.tool_instructions)} chars)")
                            if enriched_context.context_inventory:
                                logger.info(f"  ✓ Context awareness protocol enabled ({len(enriched_context.context_inventory)} chars)")
                            if enriched_context.existing_concepts and enriched_context.existing_concepts != "No concepts defined yet in this workflow.":
                                logger.info(f"  ✓ Concept registry active with terminology consistency enforcement")

                            # Track knowledge entry IDs for meta-learning
                            if enriched_context.knowledge_entries:
                                for entry in enriched_context.knowledge_entries:
                                    if hasattr(entry, 'knowledge_id') and entry.knowledge_id not in results["knowledge_entry_ids"]:
                                        results["knowledge_entry_ids"].append(entry.knowledge_id)

                            # Track tool instruction IDs for meta-learning
                            if enriched_context.tool_instruction_ids:
                                for tool_id in enriched_context.tool_instruction_ids:
                                    if tool_id not in results["knowledge_entry_ids"]:
                                        results["knowledge_entry_ids"].append(tool_id)
                                logger.info(f"  ✓ Tracking {len(enriched_context.tool_instruction_ids)} tool instruction IDs for meta-learning")

                        except Exception as ctx_error:
                            logger.warning(f"  ⚠ Collaborative context building failed, falling back to non-collaborative mode: {ctx_error}")
                            # Clear any partial context and continue without collaboration
                            task.context_history = None
                            task.knowledge_entries = None

                        # Process task through LLM at this checkpoint (with or without collaborative context)
                        # Pass central_post and streaming flag for real-time communication
                        # Phase 3.2: Wrap with failure recovery
                        try:
                            result = agent.process_task_with_llm(
                                task,
                                current_time,
                                central_post=central_post,
                                enable_streaming=felix_system.config.enable_streaming
                            )

                            # Check for low confidence that might need recovery
                            if result.confidence < 0.3:
                                logger.warning(f"⚠️ Agent {agent.agent_id} produced low confidence result: {result.confidence:.2f}")
                                failure_record = failure_recovery.record_failure(
                                    FailureType.LOW_CONFIDENCE,
                                    agent.agent_id,
                                    f"Confidence {result.confidence:.2f} below threshold 0.3",
                                    context={'result': result, 'agent_type': agent.agent_type}
                                )
                                # Recovery strategy: Spawn critic for validation
                                recovery_result = failure_recovery.attempt_recovery(failure_record)
                                if recovery_result['success'] and recovery_result.get('adjusted_parameters', {}).get('spawn_critic'):
                                    logger.info(f"  🔄 Recovery: {recovery_result['message']}")
                                    # Let dynamic spawning handle critic creation

                        except Exception as e:
                            logger.error(f"✗ Agent {agent.agent_id} processing failed: {e}")

                            # Record failure
                            failure_record = failure_recovery.record_failure(
                                FailureType.AGENT_ERROR,
                                agent.agent_id,
                                str(e),
                                context={
                                    'agent_type': agent.agent_type,
                                    'agent_params': {
                                        'temperature': getattr(agent, 'temperature', 0.7),
                                        'max_tokens': agent.max_tokens
                                    }
                                }
                            )

                            # Attempt recovery if not too many failures
                            if not failure_recovery.should_abandon_recovery(agent.agent_id):
                                recovery_result = failure_recovery.attempt_recovery(failure_record)
                                if recovery_result['success']:
                                    logger.info(f"  🔄 Recovery: {recovery_result['message']}")

                                    # Apply adjusted parameters and retry
                                    adjusted_params = recovery_result.get('adjusted_parameters', {})
                                    if adjusted_params.get('temperature'):
                                        agent.temperature = adjusted_params['temperature']
                                    if adjusted_params.get('max_tokens'):
                                        agent.max_tokens = int(adjusted_params['max_tokens'])

                                    # Retry processing
                                    try:
                                        result = agent.process_task_with_llm(
                                            task,
                                            current_time,
                                            central_post=central_post,
                                            enable_streaming=felix_system.config.enable_streaming
                                        )
                                        logger.info(f"  ✓ Recovery successful! Agent produced result with confidence {result.confidence:.2f}")
                                    except Exception as retry_error:
                                        logger.error(f"  ✗ Recovery failed: {retry_error}")
                                        continue  # Skip this agent and move on
                                else:
                                    logger.error(f"  ✗ Recovery not possible: {recovery_result['message']}")
                                    continue  # Skip this agent
                            else:
                                logger.error(f"  ✗ Abandoning recovery for {agent.agent_id} (too many failures)")
                                continue  # Skip this agent

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

                        # MULTI-STEP REASONING LOOP (Intelligent Agent Discovery)
                        # Agents can now perform multi-step reasoning: find → read → analyze
                        # This enables handling incomplete paths like "read central_post.py"
                        import re
                        reasoning_iteration = 0
                        reasoning_history = []  # Track outputs for stall detection
                        max_iterations = felix_system.config.agent_reasoning_max_iterations
                        confidence_threshold = felix_system.config.agent_reasoning_confidence_threshold

                        while reasoning_iteration < max_iterations:
                            # === EXIT CONDITION 1: Agent signals COMPLETE ===
                            if "REASONING_STATE: COMPLETE" in result.content:
                                logger.info(f"  ✓ Agent {agent.agent_id} signaled COMPLETE - reasoning finished")
                                break

                            # === EXIT CONDITION 2: Agent signals BLOCKED ===
                            if "REASONING_STATE: BLOCKED" in result.content:
                                logger.warning(f"  ⚠️ Agent {agent.agent_id} signaled BLOCKED - task may be impossible")
                                break

                            # === EXIT CONDITION 3: No more system actions needed ===
                            if "SYSTEM_ACTION_NEEDED:" not in result.content:
                                if reasoning_iteration > 0:
                                    logger.info(f"  ✓ No more system actions - reasoning complete after {reasoning_iteration} iteration(s)")
                                break

                            reasoning_iteration += 1
                            logger.info(f"  🧠 Reasoning iteration {reasoning_iteration}/{max_iterations} for agent {agent.agent_id}")

                            # === EXIT CONDITION 4: Stall detection (duplicate output) ===
                            content_hash = hash(result.content[:500])  # Hash first 500 chars for comparison
                            if content_hash in reasoning_history:
                                logger.warning(f"  ⚠️ Duplicate output detected - breaking reasoning loop to prevent infinite loop")
                                break
                            reasoning_history.append(content_hash)

                            # === EXIT CONDITION 5: Confidence threshold reached ===
                            if result.confidence >= confidence_threshold:
                                logger.info(f"  ✓ Confidence {result.confidence:.2f} >= {confidence_threshold} - reasoning complete")
                                break

                            # Process the system action
                            logger.debug(f"🐛 DEBUG: Agent output contains SYSTEM_ACTION_NEEDED:")
                            logger.debug(f"🐛 DEBUG: Full agent output:\n{result.content[:500]}")

                            # Let CentralPost process the system action message
                            logger.debug(f"🐛 DEBUG: Calling central_post.process_next_message() to execute command...")
                            central_post.process_next_message()
                            logger.debug(f"🐛 DEBUG: central_post.process_next_message() completed")

                            # Extract commands that were requested
                            pattern = r'SYSTEM_ACTION_NEEDED:\s*([^\n]+)'
                            commands = re.findall(pattern, result.content, re.IGNORECASE)
                            logger.debug(f"🐛 DEBUG: Extracted {len(commands)} command(s): {commands}")

                            if not commands:
                                logger.warning(f"  ⚠️ SYSTEM_ACTION_NEEDED: found but no commands extracted")
                                break

                            logger.info(f"  📋 Processing {len(commands)} system action(s): {', '.join(commands[:3])}{'...' if len(commands) > 3 else ''}")

                            # PHASE 2: Store file discoveries from find commands (meta-learning)
                            # This enables the system to remember where files are located
                            _store_file_discoveries(
                                result_content=result.content,
                                knowledge_store=knowledge_store,
                                agent_id=agent.agent_id,
                                task_id=task.task_id if hasattr(task, 'task_id') else 'workflow'
                            )

                            # Build updated context with command results
                            logger.debug(f"🐛 DEBUG: Building enriched context with command results...")
                            enriched_context = context_builder.build_agent_context(
                                original_task=task_input,
                                agent_type=agent.agent_type,
                                agent_id=agent.agent_id,
                                current_time=current_time,
                                max_context_tokens=40000,  # Context window budget (not agent response limit)
                                message_limit=3,
                                tool_requirements=tool_requirements
                            )

                            logger.debug(f"🐛 DEBUG: Enriched context built:")
                            logger.debug(f"🐛   - context_history length: {len(enriched_context.context_history)} chars")
                            logger.debug(f"🐛   - knowledge_entries: {len(enriched_context.knowledge_entries)} entries")
                            logger.debug(f"🐛   - tool_instructions: {len(enriched_context.tool_instructions)} chars")

                            # Update task with command results in context
                            task.context_history = enriched_context.context_history
                            task.knowledge_entries = enriched_context.knowledge_entries
                            task.tool_instructions = enriched_context.tool_instructions
                            task.context_inventory = enriched_context.context_inventory

                            logger.info(f"  🔄 Re-invoking agent {agent.agent_id} with command results...")

                            # Re-invoke the SAME agent with updated context
                            followup_result = agent.process_task_with_llm(
                                task,
                                current_time,
                                central_post=central_post,
                                enable_streaming=felix_system.config.enable_streaming
                            )

                            logger.info(f"  ✓ Iteration {reasoning_iteration} complete: confidence={followup_result.confidence:.2f}")
                            logger.debug(f"🐛 DEBUG: Does followup contain SYSTEM_ACTION_NEEDED?: {'SYSTEM_ACTION_NEEDED:' in followup_result.content}")

                            # Replace result with followup for next iteration
                            result = followup_result

                            # Share the followup result
                            followup_message = agent.share_result_to_central(result)
                            central_post.queue_message(followup_message)
                            results["messages_processed"].append(followup_message.message_id)

                            # Update stored agent output
                            agent_manager.store_agent_output(
                                agent_id=agent.agent_id,
                                result=result
                            )

                            # Update knowledge with followup result
                            if knowledge_store:
                                central_post.store_agent_result_as_knowledge(
                                    agent_id=agent.agent_id,
                                    content=result.content,
                                    confidence=result.confidence,
                                    domain="workflow_task"
                                )

                        # Log reasoning summary
                        if reasoning_iteration > 0:
                            logger.info(f"  ✓ Agent {agent.agent_id} completed {reasoning_iteration} reasoning iteration(s)")

                        # Store in knowledge base (skip if already stored during re-invocation)
                        if knowledge_store and "SYSTEM_ACTION_NEEDED:" not in result.content:
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

                        # Phase 7: Inter-agent Collaboration
                        # Agents assess collaboration opportunities and influence compatible peers
                        if len(active_agents) > 1:
                            try:
                                other_agents = [a for a in active_agents if a.agent_id != agent.agent_id]
                                if other_agents and hasattr(agent, 'assess_collaboration_opportunities'):
                                    opportunities = agent.assess_collaboration_opportunities(other_agents, current_time)
                                    collaboration_count = 0

                                    for opp in opportunities[:2]:  # Limit to top 2 opportunities
                                        if opp.get('compatibility', 0) > 0.6:
                                            target = opp.get('target_agent')
                                            if target and hasattr(agent, 'influence_agent_behavior'):
                                                influence_strength = opp['compatibility'] * 0.5
                                                agent.influence_agent_behavior(
                                                    target,
                                                    opp.get('recommended_influence', 'guidance'),
                                                    influence_strength
                                                )
                                                collaboration_count += 1
                                                logger.debug(f"  🤝 {agent.agent_id} → {target.agent_id}: "
                                                           f"{opp['recommended_influence']} (strength: {influence_strength:.2f})")

                                    if collaboration_count > 0:
                                        logger.info(f"  🤝 Inter-agent collaboration: {collaboration_count} peer influences applied")
                            except Exception as collab_error:
                                logger.debug(f"  Collaboration assessment skipped: {collab_error}")

                        # NEW: Extract and register concepts from agent response (Phase 1.4)
                        if concept_registry:
                            try:
                                import re

                                # Pattern 1: **Term**: Definition or **Term** - Definition
                                # Matches bold markdown terms followed by definition
                                pattern1 = r'\*\*([^*]+)\*\*[:\-]\s*([^\n]+(?:\n(?!\*\*)[^\n]+)*)'
                                matches1 = re.findall(pattern1, result.content, re.MULTILINE)

                                # Pattern 2: Term: Definition (at line start, title case)
                                # More conservative: only matches capitalized terms at line start
                                pattern2 = r'^([A-Z][A-Za-z\s]{2,30}):\s*([^\n]+(?:\n(?![A-Z])[^\n]+)*)'
                                matches2 = re.findall(pattern2, result.content, re.MULTILINE)

                                # Combine and deduplicate matches
                                all_concepts = {}
                                for name, definition in matches1 + matches2:
                                    name = name.strip()
                                    definition = definition.strip()
                                    # Filter out obviously non-conceptual patterns
                                    if len(name) > 2 and len(definition) > 10 and not name.lower().startswith('http'):
                                        # Keep longest definition if duplicate
                                        if name not in all_concepts or len(definition) > len(all_concepts[name]):
                                            all_concepts[name] = definition

                                # Register extracted concepts
                                if all_concepts:
                                    logger.info(f"  📚 Extracting {len(all_concepts)} concept(s) from {agent.agent_id}...")
                                    for name, definition in all_concepts.items():
                                        # Truncate very long definitions (likely false positives)
                                        if len(definition) > 300:
                                            definition = definition[:300] + "..."

                                        registered = concept_registry.register_concept(
                                            name=name,
                                            definition=definition,
                                            source_agent=agent.agent_id,
                                            confidence=result.confidence
                                        )
                                        if registered:
                                            logger.info(f"    ✓ Registered: '{name}'")
                                        else:
                                            logger.info(f"    ℹ️ Duplicate detected: '{name}'")

                            except Exception as e:
                                logger.error(f"  ✗ Concept extraction failed: {e}")

                        # NEW: Reasoning process evaluation for CriticAgent (Phase 1.2)
                        # Phase 6: Track reasoning evals for weighted synthesis contributions
                        if agent.agent_type == "critic" and len(results["llm_responses"]) > 1:
                            try:
                                # Evaluate the most recent non-critic agent's reasoning
                                previous_outputs = [r for r in results["llm_responses"] if r["agent_type"] != "critic"]
                                if previous_outputs:
                                    last_output = previous_outputs[-1]
                                    evaluated_agent_id = last_output["agent_id"]
                                    agent_output = {
                                        "result": last_output["response"],
                                        "confidence": last_output["confidence"],
                                        "agent_id": evaluated_agent_id
                                    }
                                    agent_metadata = {
                                        "agent_type": last_output["agent_type"],
                                        "checkpoint": last_output["checkpoint"],
                                        "time": last_output["time"]
                                    }

                                    logger.info(f"  🧠 CriticAgent evaluating reasoning process of {evaluated_agent_id}...")
                                    reasoning_eval = agent.evaluate_reasoning_process(agent_output, agent_metadata)

                                    logger.info(f"  ✓ Reasoning evaluation complete:")
                                    logger.info(f"    Quality score: {reasoning_eval['reasoning_quality_score']:.2f}")
                                    logger.info(f"    Logical coherence: {reasoning_eval['logical_coherence']:.2f}")
                                    logger.info(f"    Evidence quality: {reasoning_eval['evidence_quality']:.2f}")
                                    if reasoning_eval['identified_issues']:
                                        logger.info(f"    Issues: {', '.join(reasoning_eval['identified_issues'])}")
                                    if reasoning_eval['re_evaluation_needed']:
                                        logger.warning(f"    ⚠ Re-evaluation recommended")

                                    # Phase 6: Store reasoning eval for weighted synthesis
                                    reasoning_evals[evaluated_agent_id] = reasoning_eval

                                    # Phase 6: Store reasoning issues in task metadata for downstream agents
                                    if reasoning_eval.get('re_evaluation_needed'):
                                        if hasattr(task, 'metadata'):
                                            task.metadata['reasoning_issues'] = reasoning_eval['identified_issues']
                                            task.metadata['improvement_recommendations'] = reasoning_eval['improvement_recommendations']
                                        logger.info(f"    📋 Stored reasoning issues in task metadata for agent improvement")

                                    # Store reasoning evaluation in knowledge for meta-learning
                                    if knowledge_store:
                                        central_post.store_agent_result_as_knowledge(
                                            agent_id=f"{agent.agent_id}_reasoning_eval",
                                            content=f"Reasoning evaluation of {evaluated_agent_id}:\n" +
                                                   f"Quality: {reasoning_eval['reasoning_quality_score']:.2f}\n" +
                                                   f"Issues: {', '.join(reasoning_eval['identified_issues'])}\n" +
                                                   f"Recommendations: {', '.join(reasoning_eval['improvement_recommendations'])}",
                                            confidence=reasoning_eval['reasoning_quality_score'],
                                            domain="reasoning_evaluation"
                                        )
                            except Exception as e:
                                logger.error(f"  ✗ Reasoning evaluation failed: {e}")

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
                            progress_callback(f"{agent.agent_type} checkpoint {checkpoint:.1f}", min(99, progress_pct + 5))

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

            # EARLY TERMINATION: Check if system actions completed successfully
            # If task requires only system actions (like file creation), exit early
            # BUT: Require meaningful progress (multiple operations) to avoid premature exit
            if step >= 2:  # Wait at least 2 steps to allow multi-command workflows
                # Get all completed system action results from CentralPost
                action_results = central_post.get_action_results()

                if action_results:
                    # Check if ALL actions succeeded
                    # CommandResult has a 'success' attribute
                    all_succeeded = all(result.success for result in action_results)

                    # Smart termination: Require EITHER:
                    # - Multiple commands (2+) for multi-step tasks, OR
                    # - Later workflow stage (step >= 3) for single complex command
                    multiple_commands = len(action_results) >= 2
                    late_stage = step >= 3

                    should_terminate = all_succeeded and (multiple_commands or late_stage)

                    if should_terminate:
                        logger.info("=" * 60)
                        logger.info("⚡ EARLY TERMINATION TRIGGERED")
                        logger.info(f"  Reason: All system actions completed successfully")
                        logger.info(f"  Actions completed: {len(action_results)}")
                        logger.info(f"  Multiple commands: {multiple_commands}")
                        logger.info(f"  Late stage (step >= 3): {late_stage}")
                        logger.info(f"  Step: {step}/{total_steps}")
                        logger.info(f"  Skipping remaining {total_steps - step - 1} steps")
                        logger.info("=" * 60)

                        if progress_callback:
                            progress_callback("System actions complete - finishing workflow", 90.0)

                        # Break early - will proceed to final synthesis
                        break
                    elif all_succeeded and len(action_results) == 1:
                        logger.info(f"⏸ Single command completed at step {step}, continuing workflow for potential follow-ups...")

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
                        spawn_time_range=(current_time, min(current_time + 0.01, 1.0))
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
                        spawned_agent_ids.append(analysis_agent.agent_id)  # Track for cleanup
                        logger.info(f"  → Spawned {analysis_agent.agent_type}: {analysis_agent.agent_id}")

                        # Record scaling metric for H2 hypothesis validation
                        elapsed_time = time.time() - workflow_start_time
                        central_post.performance_monitor.record_scaling_metric(
                            agent_count=len(active_agents),
                            processing_time=elapsed_time
                        )
                    else:
                        logger.warning(f"⚠ Agent cap reached during fallback spawning - skipping {analysis_agent.agent_id}")

                    # Also spawn critic agent for minimum team
                    if len(active_agents) < 3:
                        critic_agent = agent_factory.create_critic_agent(
                            spawn_time_range=(current_time, min(current_time + 0.01, 1.0))
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
                            spawned_agent_ids.append(critic_agent.agent_id)  # Track for cleanup
                            logger.info(f"  → Spawned {critic_agent.agent_type}: {critic_agent.agent_id}")
                        else:
                            logger.warning(f"⚠ Agent cap reached during fallback spawning - skipping {critic_agent.agent_id}")

                    if progress_callback:
                        progress_callback(f"Spawned fallback agents", min(99, progress_pct + 2))

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

                # Call assess_team_needs with REAL messages and task description
                new_agents = agent_factory.assess_team_needs(
                    all_processed_messages,  # REAL messages with all fields!
                    current_time,
                    active_agents,
                    task_description=task_input  # NEW: Pass task for plugin-aware spawning
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
                            spawned_agent_ids.append(new_agent.agent_id)  # Track for cleanup
                            logger.info(f"  → {new_agent.agent_type} agent: {new_agent.agent_id}")

                            if progress_callback:
                                progress_callback(f"Spawned {new_agent.agent_type} agent", min(99, progress_pct + 2))
                        else:
                            # Agent cap reached - stop trying to spawn more agents
                            logger.warning(f"⚠ Agent cap reached - cannot spawn {new_agent.agent_id}")
                            logger.info(f"→ Continuing with {len(active_agents)} existing agents to complete helical progression")
                            break
                else:
                    logger.info(f"✗ No spawning needed (sufficient confidence or at capacity)")

            # Phase 3.1: Check for agent extension requests (dynamic checkpoint injection)
            if step >= total_steps - 2:  # Near end of workflow
                extension_requests = central_post.get_extension_requests()
                if extension_requests and not hasattr(results, '_extensions_granted'):
                    results['_extensions_granted'] = 0

                # Grant up to 2 extensions per workflow
                if extension_requests and results.get('_extensions_granted', 0) < 2:
                    logger.info("\n" + "="*60)
                    logger.info("🔄 AGENT PROCESSING EXTENSION REQUESTS")
                    logger.info("="*60)
                    logger.info(f"  Requests: {len(extension_requests)}")
                    for req in extension_requests:
                        logger.info(f"    - {req['agent_id']}: {req['reason']}")

                    # Add 3 additional steps
                    additional_steps = 3
                    old_total = total_steps
                    total_steps += additional_steps
                    time_step = 1.0 / total_steps  # Recalculate

                    results['_extensions_granted'] = results.get('_extensions_granted', 0) + 1

                    logger.info(f"  ✓ Extension granted ({results['_extensions_granted']}/2)")
                    logger.info(f"  ✓ Total steps extended: {old_total} → {total_steps}")
                    logger.info(f"  ✓ Time step recalculated: {time_step:.4f}")
                    logger.info("="*60)

                    # Clear requests after processing
                    central_post.clear_extension_requests()

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
                            # Phase 6: Pass reasoning evaluations for weighted synthesis
                            # Phase 7: Pass coverage report for meta-confidence
                            synthesis_result = central_post.synthesize_agent_outputs(
                                task_description=task_input,
                                max_messages=20,
                                task_complexity=task_complexity,
                                reasoning_evals=reasoning_evals if reasoning_evals else None,
                                coverage_report=coverage_report
                            )
                            results["centralpost_synthesis"] = synthesis_result
                            logger.info(f"✓ CentralPost synthesis complete!")
                            logger.info(f"  Synthesis confidence: {synthesis_result['confidence']:.2f}")
                            if synthesis_result.get('meta_confidence') != synthesis_result.get('confidence'):
                                logger.info(f"  Meta-confidence: {synthesis_result.get('meta_confidence', 0):.2f}")
                            logger.info(f"  Agents synthesized: {synthesis_result['agents_synthesized']}")
                            logger.info(f"  Tokens used: {synthesis_result['tokens_used']} / {synthesis_result['max_tokens']}")
                            if reasoning_evals:
                                logger.info(f"  Reasoning evals applied: {len(reasoning_evals)} agents weighted")
                            if synthesis_result.get('epistemic_caveats'):
                                logger.info(f"  Epistemic caveats: {len(synthesis_result['epistemic_caveats'])}")

                            # NEW: Broadcast synthesis feedback to agents for learning
                            # Pass knowledge IDs and task type for meta-learning boost
                            central_post.broadcast_synthesis_feedback(
                                synthesis_result=synthesis_result,
                                task_description=task_input,
                                workflow_id=workflow_id,
                                task_type=classified_task_type,
                                knowledge_entry_ids=results.get("knowledge_entry_ids", [])
                            )

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
                # Phase 6: Pass reasoning evaluations for weighted synthesis
                # Phase 7: Pass coverage report for meta-confidence
                synthesis_result = central_post.synthesize_agent_outputs(
                    task_description=task_input,
                    max_messages=20,
                    task_complexity=task_complexity,
                    reasoning_evals=reasoning_evals if reasoning_evals else None,
                    coverage_report=coverage_report
                )
                results["centralpost_synthesis"] = synthesis_result
                logger.info(f"✓ Final CentralPost synthesis complete")
                if synthesis_result.get('meta_confidence') != synthesis_result.get('confidence'):
                    logger.info(f"  Meta-confidence: {synthesis_result.get('meta_confidence', 0):.2f}")
                if reasoning_evals:
                    logger.info(f"  Reasoning evals applied: {len(reasoning_evals)} agents weighted")
                if synthesis_result.get('epistemic_caveats'):
                    logger.info(f"  Epistemic caveats: {len(synthesis_result['epistemic_caveats'])}")

                # NEW: Broadcast synthesis feedback to agents for learning
                # Pass knowledge IDs and task type for meta-learning boost
                central_post.broadcast_synthesis_feedback(
                    synthesis_result=synthesis_result,
                    task_description=task_input,
                    workflow_id=workflow_id,
                    task_type=classified_task_type,
                    knowledge_entry_ids=results.get("knowledge_entry_ids", [])
                )
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

        # NEW: Task completion detection (Phase 2.3)
        if results["centralpost_synthesis"]:
            try:
                from src.workflows.task_completion_detector import TaskCompletionDetector, CompletionStatus

                detector = TaskCompletionDetector()
                completion_status, completion_score, completion_reason = detector.detect_completion(
                    task_description=task_input,
                    synthesis_output=results["centralpost_synthesis"]["synthesis_content"],
                    synthesis_confidence=results["centralpost_synthesis"]["confidence"],
                    agent_count=len(results["agents_spawned"])
                )

                # Store completion analysis in results
                results["task_completion"] = {
                    "status": completion_status.value,
                    "score": completion_score,
                    "reason": completion_reason
                }

                logger.info("\n--- Task Completion Analysis ---")
                logger.info(f"Status: {completion_status.value.upper()}")
                logger.info(f"Score: {completion_score:.2f}")
                logger.info(f"Reason: {completion_reason}")

                # If task incomplete, log recommendation
                if completion_status == CompletionStatus.INCOMPLETE:
                    logger.warning("⚠️ Task appears INCOMPLETE - consider:")
                    logger.warning("  - Increasing max_steps for more processing time")
                    logger.warning("  - Clarifying task requirements")
                    logger.warning("  - Checking if agents had necessary tools/context")

            except Exception as e:
                logger.error(f"Task completion detection failed: {e}")
                results["task_completion"] = {
                    "status": "error",
                    "score": 0.0,
                    "reason": str(e)
                }

        logger.info("="*60)
        logger.info("FELIX WORKFLOW COMPLETED")
        logger.info(f"Agents spawned: {len(results['agents_spawned'])}")
        logger.info(f"Messages processed: {len(results['messages_processed'])}")
        logger.info(f"Knowledge entries: {len(results['knowledge_entries'])}")
        logger.info(f"LLM responses: {len(results['llm_responses'])}")
        if results.get("task_completion"):
            logger.info(f"Task completion: {results['task_completion']['status']} ({results['task_completion']['score']:.2f})")
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
                    task_type=classified_task_type,
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

        # Record workflow outcome for learning systems
        if recommendation_engine:
            try:
                logger.info("=" * 60)
                logger.info("🧠 LEARNING SYSTEM - Recording Workflow Outcome")

                # Calculate workflow duration
                workflow_duration = time.time() - workflow_start_time

                # Determine workflow success
                workflow_success = (
                    results.get("status") == "completed" and
                    results.get("centralpost_synthesis") is not None
                )

                # Get final confidence from synthesis
                final_confidence = 0.0
                if results.get("centralpost_synthesis"):
                    final_confidence = results["centralpost_synthesis"].get("confidence", 0.0)
                elif results.get("llm_responses"):
                    # Fallback to average agent confidence
                    confidences = [r.get("confidence", 0.0) for r in results["llm_responses"]]
                    final_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # Build agents_used list with confidence predictions
                agents_used_info = []
                if 'active_agents' in locals():
                    for agent in active_agents:
                        agent_info = {
                            'type': agent.agent_type,
                            'predicted_confidence': agent.confidence if hasattr(agent, 'confidence') else 0.0
                        }
                        agents_used_info.append(agent_info)

                # Get thresholds that were used (learned or standard)
                thresholds_used = learned_thresholds if learned_thresholds else {
                    'confidence_threshold': 0.8,
                    'team_expansion_threshold': 0.7,
                    'volatility_threshold': 0.15,
                    'web_search_threshold': 0.7
                }

                # Record outcome
                # Convert synthesis complexity string to TaskComplexity enum
                task_complexity_enum = _map_synthesis_complexity_to_task_complexity(task_complexity)
                recommendation_engine.record_workflow_outcome(
                    workflow_id=workflow_id,
                    task_type=classified_task_type,
                    task_complexity=task_complexity_enum,
                    agents_used=agents_used_info,
                    workflow_success=workflow_success,
                    workflow_duration=workflow_duration,
                    final_confidence=final_confidence,
                    thresholds_used=thresholds_used,
                    recommendation_id=recommendation_id
                )

                logger.info(f"  ✓ Recorded: success={workflow_success}, "
                           f"confidence={final_confidence:.2f}, "
                           f"duration={workflow_duration:.1f}s, "
                           f"agents={len(agents_used_info)}")
                logger.info("=" * 60)

            except Exception as learning_error:
                # Don't fail workflow if learning recording fails
                logger.warning(f"Could not record learning outcome: {learning_error}")

        # === KNOWLEDGE USAGE RECORDING (META-LEARNING) ===
        # Record which knowledge entries were used and how helpful they were
        if results["knowledge_entry_ids"] and felix_system.knowledge_store:
            try:
                logger.info("=" * 60)
                logger.info("RECORDING KNOWLEDGE USAGE (META-LEARNING)")

                # Determine usefulness score based on workflow outcome
                final_confidence = results.get("centralpost_synthesis", {}).get("confidence", 0.0)
                workflow_success = final_confidence >= 0.7  # Consider 0.7+ as success

                if workflow_success:
                    if final_confidence >= 0.8:
                        useful_score = 0.9  # Very helpful
                    elif final_confidence >= 0.6:
                        useful_score = 0.7  # Helpful
                    else:
                        useful_score = 0.5  # Somewhat helpful
                else:
                    useful_score = 0.3  # Not very helpful

                # Record usage for meta-learning boost
                felix_system.knowledge_store.record_knowledge_usage(
                    workflow_id=workflow_id,
                    knowledge_ids=results["knowledge_entry_ids"],
                    task_type=classified_task_type,  # Now properly classified for meta-learning
                    task_complexity=task_complexity,
                    useful_score=useful_score,
                    retrieval_method="adaptive"  # Using our new adaptive system
                )

                logger.info(f"  ✓ Recorded usage for {len(results['knowledge_entry_ids'])} knowledge entries")
                logger.info(f"  Usefulness score: {useful_score:.2f} (confidence={final_confidence:.2f})")
                logger.info("=" * 60)

            except Exception as knowledge_error:
                # Don't fail workflow if knowledge recording fails
                logger.warning(f"Could not record knowledge usage: {knowledge_error}")

        # === PERFORMANCE MONITORING ===
        # Record final workflow metrics and log performance summary
        try:
            workflow_duration = time.time() - workflow_start_time
            agent_count = len(active_agents) if 'active_agents' in locals() else 0

            # Record final workflow processing time
            central_post.performance_monitor.record_processing_time(workflow_duration)

            # Get comprehensive performance summary
            performance_summary = central_post.performance_monitor.get_performance_summary(
                active_connections=agent_count
            )

            # Log performance metrics (for H1/H2/H3 hypothesis validation)
            logger.info("=" * 60)
            logger.info("WORKFLOW PERFORMANCE METRICS")
            logger.info(f"  Duration: {workflow_duration:.2f}s")
            logger.info(f"  Agent count: {agent_count}")
            logger.info(f"  Messages processed: {performance_summary.get('total_messages_processed', 0)}")
            logger.info(f"  Message throughput: {performance_summary.get('message_throughput', 0):.2f} msgs/sec")
            logger.info(f"  Avg processing time: {performance_summary.get('avg_processing_time', 0):.3f}s")
            if performance_summary.get('average_overhead_ratio', 0) > 0:
                logger.info(f"  Overhead ratio: {performance_summary.get('average_overhead_ratio', 0):.3f}")
            logger.info("=" * 60)

            # Add performance metrics to results for GUI/analysis
            results['performance_metrics'] = performance_summary

        except Exception as perf_error:
            # Don't fail workflow if performance logging fails
            logger.warning(f"Could not log performance metrics: {perf_error}")

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

    finally:
        # Cleanup: Remove all spokes and deregister agents for this workflow
        if 'spawned_agent_ids' in locals() and 'felix_system' in locals():
            logger.info(f"Cleaning up {len(spawned_agent_ids)} workflow agents...")
            for agent_id in spawned_agent_ids:
                # Remove spoke
                if felix_system.spoke_manager:
                    removed = felix_system.spoke_manager.remove_spoke(agent_id)
                    if removed:
                        logger.info(f"  ✓ Removed spoke for {agent_id}")
                    else:
                        logger.debug(f"  - Spoke not found for {agent_id}")

                # Deregister from agent manager
                if felix_system.agent_manager and agent_id in felix_system.agent_manager.agents:
                    felix_system.agent_manager.deregister_agent(agent_id)
                    logger.info(f"  ✓ Deregistered agent {agent_id}")

            logger.info("✓ Workflow cleanup complete")

        # Clear workflow ID and approval rules in CentralPost
        if 'central_post' in locals() and 'workflow_id' in locals():
            central_post.clear_current_workflow()
            logger.info(f"✓ Cleared workflow approval rules for {workflow_id}")
