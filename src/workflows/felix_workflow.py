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


def run_felix_workflow(felix_system, task_input: str,
                       progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """
    Run a workflow using the Felix framework components.

    This properly integrates with the Felix system initialized by the GUI:
    - Uses felix_system.central_post for communication
    - Uses felix_system.agent_factory for spawning
    - Uses felix_system.agent_manager for registration
    - Uses felix_system.knowledge_store for memory
    - Uses felix_system.lm_client shared across all agents

    Args:
        felix_system: Initialized FelixSystem instance from GUI
        task_input: Task description to process
        progress_callback: Optional callback(status_message, progress_percentage)

    Returns:
        Dictionary with workflow results and metadata
    """

    try:
        # Track workflow execution time for task memory
        workflow_start_time = time.time()

        logger.info("="*60)
        logger.info("FELIX WORKFLOW STARTING")
        logger.info(f"Task: {task_input}")
        logger.info("="*60)

        # Get Felix components (already initialized by GUI)
        central_post = felix_system.central_post
        agent_factory = felix_system.agent_factory
        agent_manager = felix_system.agent_manager
        knowledge_store = felix_system.knowledge_store

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

        # Track results
        results = {
            "task": task_input,
            "agents_spawned": [],
            "messages_processed": [],
            "llm_responses": [],
            "knowledge_entries": [],
            "status": "in_progress",
            "final_synthesis": None  # Will store the final synthesis output
        }

        # Track ALL processed messages for dynamic spawning (REAL messages, not mocks!)
        all_processed_messages = []

        # Time progression parameters
        total_steps = 20  # Increased from 10 to allow consensus building
        current_time = 0.0
        time_step = 1.0 / total_steps

        logger.info(f"Workflow will progress through {total_steps} time steps")

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
        if felix_system.spoke_manager:
            felix_system.spoke_manager.create_spoke(research_agent)
            logger.info(f"Created spoke connection for {research_agent.agent_id}")
        elif felix_system.central_post:
            # Fallback to direct registration if no spoke_manager
            felix_system.central_post.register_agent(research_agent)
            logger.info(f"Registered {research_agent.agent_id} directly with central_post")

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
                if agent.can_spawn(current_time) and agent.state.value == "waiting":
                    logger.info(f"Agent {agent.agent_id} spawning at t={current_time:.2f}")

                    try:
                        # Spawn agent
                        agent.spawn(current_time, task)

                        # Build enriched collaborative context for this agent
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
                            logger.info(f"✓ Using collaborative context with {len(enriched_context.context_history)} previous outputs")

                        except Exception as ctx_error:
                            logger.warning(f"⚠ Collaborative context building failed, falling back to non-collaborative mode: {ctx_error}")
                            # Clear any partial context and continue without collaboration
                            task.context_history = None
                            task.knowledge_entries = None

                        # Process task through LLM (with or without collaborative context)
                        result = agent.process_task_with_llm(task, current_time)

                        logger.info(f"Agent {agent.agent_id} completed: confidence={result.confidence:.2f}")

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
                            "response": result.content,  # Changed from "content_preview" to "response"
                            "time": current_time
                        })

                        # Track synthesis agent outputs for final synthesis
                        if agent.agent_type.lower() == "synthesis":
                            # Keep the most recent or highest confidence synthesis
                            if results["final_synthesis"] is None or result.confidence > results["final_synthesis"]["confidence"]:
                                results["final_synthesis"] = {
                                    "agent_id": agent.agent_id,
                                    "content": result.content,
                                    "confidence": result.confidence,
                                    "time": current_time
                                }
                                logger.info(f"Updated final synthesis from {agent.agent_id} (confidence: {result.confidence:.2f})")

                        if progress_callback:
                            progress_callback(f"{agent.agent_type} agent completed", progress_pct + 5)

                    except Exception as e:
                        logger.error(f"Error processing agent {agent.agent_id}: {e}", exc_info=True)

            # Process pending messages in CentralPost
            messages_this_step = 0
            while central_post.has_pending_messages():
                processed_msg = central_post.process_next_message()
                if processed_msg:
                    messages_this_step += 1

            if messages_this_step > 0:
                logger.info(f"Processed {messages_this_step} messages through CentralPost")

            # Debug: Always log spawning check status
            logger.info(f"=== STEP {step}: Dynamic Spawning Check ===")
            logger.info(f"  Condition step >= 1: {step >= 1}")
            logger.info(f"  AgentFactory exists: {agent_factory is not None}")
            logger.info(f"  enable_dynamic_spawning: {agent_factory.enable_dynamic_spawning if agent_factory else 'N/A'}")
            logger.info(f"  Processed messages count: {len(all_processed_messages)}")
            logger.info(f"  Active agents count: {len(active_agents)}")

            # Dynamic spawning after we have some results (step 1+)
            if step >= 1 and agent_factory.enable_dynamic_spawning:
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
                    if felix_system.spoke_manager:
                        felix_system.spoke_manager.create_spoke(analysis_agent)
                        logger.info(f"Created spoke connection for {analysis_agent.agent_id}")
                    elif felix_system.central_post:
                        felix_system.central_post.register_agent(analysis_agent)
                        logger.info(f"Registered {analysis_agent.agent_id} directly with central_post")

                    active_agents.append(analysis_agent)
                    results["agents_spawned"].append(analysis_agent.agent_id)
                    logger.info(f"  → Spawned {analysis_agent.agent_type}: {analysis_agent.agent_id}")

                    # Also spawn critic agent for minimum team
                    if len(active_agents) < 3:
                        critic_agent = agent_factory.create_critic_agent(
                            spawn_time_range=(current_time, current_time + 0.01)
                        )
                        agent_manager.register_agent(critic_agent)

                        # CRITICAL: Create spoke connection (handles central_post registration)
                        if felix_system.spoke_manager:
                            felix_system.spoke_manager.create_spoke(critic_agent)
                            logger.info(f"Created spoke connection for {critic_agent.agent_id}")
                        elif felix_system.central_post:
                            felix_system.central_post.register_agent(critic_agent)
                            logger.info(f"Registered {critic_agent.agent_id} directly with central_post")

                        active_agents.append(critic_agent)
                        results["agents_spawned"].append(critic_agent.agent_id)
                        logger.info(f"  → Spawned {critic_agent.agent_type}: {critic_agent.agent_id}")

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
                        if felix_system.spoke_manager:
                            felix_system.spoke_manager.create_spoke(new_agent)
                            logger.info(f"Created spoke connection for {new_agent.agent_id}")
                        elif felix_system.central_post:
                            felix_system.central_post.register_agent(new_agent)
                            logger.info(f"Registered {new_agent.agent_id} directly with central_post")

                        active_agents.append(new_agent)
                        results["agents_spawned"].append(new_agent.agent_id)

                        logger.info(f"  → {new_agent.agent_type} agent: {new_agent.agent_id}")

                        if progress_callback:
                            progress_callback(f"Spawned {new_agent.agent_type} agent", progress_pct + 2)
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

                    # Debug: Always log consensus check
                    logger.info(f"--- Consensus Check (step {step}) ---")
                    logger.info(f"  Recent confidence scores: {[f'{c:.2f}' for c in recent_confidence]}")
                    logger.info(f"  Average: {avg_recent:.2f} (threshold: 0.80)")
                    logger.info(f"  Team size: {len(results['agents_spawned'])} (minimum: 3)")
                    logger.info(f"  Synthesis output: {'Yes' if results['final_synthesis'] else 'No'}")

                    # Exit if high confidence + sufficient agents + synthesis agent has run
                    if avg_recent >= 0.80 and len(results["agents_spawned"]) >= 3 and results["final_synthesis"] is not None:
                        logger.info(f"\n✓ Confident consensus reached with synthesis output!")
                        logger.info(f"  Confidence: {avg_recent:.2f} >= threshold (0.80)")
                        logger.info(f"  Team size: {len(results['agents_spawned'])} >= minimum (3)")
                        logger.info(f"  Synthesis agent: {results['final_synthesis']['agent_id']}")
                        if progress_callback:
                            progress_callback(f"Consensus reached!", 100.0)
                        break  # Early exit - consensus achieved!
                    elif avg_recent >= 0.80 and results["final_synthesis"] is None:
                        logger.info(f"  ⚠ Confidence threshold met but no synthesis output yet - continuing...")
                        # Force spawn synthesis agent if approaching consensus without one
                        synthesis_agents = [a for a in active_agents if a.agent_type.lower() == "synthesis"]
                        if not synthesis_agents:
                            logger.info(f"  → Spawning synthesis agent to generate final output...")
                            synthesis_agent = felix_system.central_post.create_synthesis_agent()
                            active_agents.append(synthesis_agent)
                            results["agents_spawned"].append(synthesis_agent.agent_id)

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

        # Check for synthesis agent output (required for proper final output)
        if results["final_synthesis"] is None:
            if results["llm_responses"]:
                # Use highest confidence output as last resort, but log warning
                highest_confidence_response = max(results["llm_responses"], key=lambda r: r.get("confidence", 0.0))
                results["final_synthesis"] = {
                    "agent_id": highest_confidence_response["agent_id"],
                    "content": highest_confidence_response["response"],
                    "confidence": highest_confidence_response["confidence"],
                    "time": highest_confidence_response["time"]
                }
                logger.warning(f"⚠ WARNING: No synthesis agent output! Using fallback from {highest_confidence_response['agent_id']}")
                logger.warning(f"   This indicates workflow completed before synthesis phase - consider increasing max_steps")
            else:
                logger.error(f"✗ ERROR: No agent outputs available for final synthesis!")

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
