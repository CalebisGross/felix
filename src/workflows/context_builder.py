"""
Collaborative Context Builder for Felix Framework

Builds enriched context for agents by:
- Retrieving recent messages from CentralPost
- Compressing accumulated context when needed
- Integrating relevant knowledge from memory
- Creating agent-type-specific prompts
"""

import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Use 'felix_workflows' logger to match GUI configuration and ensure visibility
logger = logging.getLogger('felix_workflows')


@dataclass
class EnrichedContext:
    """Container for enriched agent context."""
    task_description: str
    context_history: List[Dict[str, Any]]
    original_context_size: int
    compressed_context_size: int
    compression_ratio: float
    knowledge_entries: List[Any]
    message_count: int
    tool_instructions: str = ""  # Conditional tool instructions based on task requirements
    tool_instruction_ids: List[str] = None  # IDs of tool instruction knowledge entries (for meta-learning)
    context_inventory: str = ""  # NEW: Explicit inventory of available resources for agent comprehension
    existing_concepts: str = ""  # NEW: Existing concepts from concept registry for terminology consistency
    version: int = 0  # Phase 2.4: Context version number for race-free synchronization

    def __post_init__(self):
        if self.tool_instruction_ids is None:
            self.tool_instruction_ids = []


class CollaborativeContextBuilder:
    """
    Builds enriched context for agents from accumulated messages.

    This enables true collaborative multi-agent workflows where each agent
    builds upon the work of previous agents rather than working in isolation.
    """

    def __init__(self, central_post, knowledge_store=None, context_compressor=None, workflow_id: Optional[str] = None):
        """
        Initialize collaborative context builder.

        Args:
            central_post: CentralPost instance for message retrieval and memory access
            knowledge_store: DEPRECATED - Access through central_post.memory_facade instead
            context_compressor: Optional ContextCompressor (fallback to memory_facade if None)
            workflow_id: Optional workflow ID for concept registry
        """
        self.central_post = central_post

        # Access memory systems through MemoryFacade for consistency
        self.memory_facade = central_post.memory_facade if hasattr(central_post, 'memory_facade') else None

        # Fallback for backward compatibility (will be removed in future)
        self.knowledge_store = knowledge_store if self.memory_facade is None else None

        # Use memory_facade's compressor if available, otherwise use provided one
        if self.memory_facade and hasattr(self.memory_facade, 'context_compressor'):
            self.context_compressor = self.memory_facade.context_compressor
        else:
            self.context_compressor = context_compressor

        # Track context building statistics
        self.contexts_built = 0
        self.total_compression_ratio = 0.0

        # Phase 2.4: Context versioning for race-free synchronization
        self._context_version = 0
        self._context_lock = threading.RLock()  # Reentrant lock for nested calls
        self._last_message_count = 0  # Track when context needs updating

        # NEW: Concept registry for terminology consistency
        from src.workflows.concept_registry import ConceptRegistry
        workflow_id = workflow_id or f"workflow_{id(self)}"
        self.concept_registry = ConceptRegistry(workflow_id)

        # NEW: Context relevance evaluator for filtering irrelevant facts
        from src.workflows.context_relevance import ContextRelevanceEvaluator
        self.relevance_evaluator = ContextRelevanceEvaluator()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count using 4 chars ≈ 1 token heuristic.

        This is a standard approximation for English text. More accurate
        than character count, doesn't require full tokenizer library.

        Args:
            text: Text to estimate token count for

        Returns:
            Estimated number of tokens
        """
        return len(text) // 4

    @staticmethod
    def _get_knowledge_limit_for_complexity(task_complexity: str) -> int:
        """
        Get adaptive knowledge retrieval limit based on task complexity.

        Simple factual queries don't need many knowledge entries, while
        complex analytical tasks benefit from richer context.

        Args:
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", "COMPLEX")

        Returns:
            Knowledge entry limit (8, 15, or 25)
        """
        if task_complexity == "SIMPLE_FACTUAL":
            return 8  # Slight increase from 5 for simple queries
        elif task_complexity == "MEDIUM":
            return 15  # 3x for moderate analysis tasks
        else:  # COMPLEX
            return 25  # 5x for deep analytical work

    def _get_known_file_locations(self, task_description: str) -> str:
        """
        Retrieve known file locations relevant to task.

        This enables meta-learning: files discovered by previous find commands
        are remembered and provided to agents to avoid redundant discovery.

        Args:
            task_description: The task that may mention filenames

        Returns:
            Formatted string of known file locations, or empty string if none
        """
        import re

        # Extract filenames mentioned in task (e.g., "central_post.py", "config.yaml")
        filename_pattern = r'[\w_-]+\.(?:py|js|ts|java|cpp|c|h|go|rs|rb|txt|md|json|yaml|yml|xml|html|css|sh)'
        filenames = re.findall(filename_pattern, task_description, re.IGNORECASE)

        if not filenames:
            return ""

        # Get access to knowledge store
        memory_system = self.memory_facade if self.memory_facade else self.knowledge_store
        if not memory_system:
            return ""

        known_paths = []
        try:
            from src.memory.knowledge_store import KnowledgeQuery

            for filename in set(filenames):  # Deduplicate
                # Query for file_locations domain
                query = KnowledgeQuery(
                    domains=["file_locations"],
                    content_keywords=[filename],
                    limit=3
                )

                if self.memory_facade:
                    entries = self.memory_facade.retrieve_knowledge_with_query(query)
                else:
                    entries = self.knowledge_store.retrieve_knowledge(query)

                for entry in entries:
                    if isinstance(entry.content, dict):
                        full_path = entry.content.get('full_path', '')
                        if full_path:
                            known_paths.append(f"  {filename} → {full_path}")
                            break  # Take first match per filename

        except Exception as e:
            logger.debug(f"Failed to retrieve file locations: {e}")
            return ""

        if known_paths:
            return "📁 KNOWN FILE LOCATIONS (from previous discoveries):\n" + "\n".join(known_paths)
        return ""

    def build_context_inventory(self,
                                tool_instructions: str,
                                tool_instruction_ids: List[str],
                                knowledge_entries: List[Any],
                                context_history: List[Dict[str, Any]],
                                task_description: str = "",
                                coverage_report: Optional[Any] = None) -> str:
        """
        Build explicit context inventory for agent prompts to improve comprehension.

        This creates a clear "what you already have" checklist that forces agents
        to acknowledge available resources before making redundant requests.

        Args:
            tool_instructions: Tool instruction text (if any)
            tool_instruction_ids: IDs of tool instructions
            knowledge_entries: Knowledge entries retrieved
            context_history: Previous agent outputs
            task_description: Original task for file location lookup (meta-learning)
            coverage_report: Optional CoverageReport for epistemic awareness (Phase 5)

        Returns:
            Formatted inventory string for prompt injection
        """
        inventory_lines = [
            "=" * 60,
            "📦 CONTEXT INVENTORY (WHAT YOU ALREADY HAVE)",
            "=" * 60
        ]

        # Tool availability
        # Tools are always available via explicit retrieval or fallback (MINIMAL_TOOLS_FALLBACK)
        # Never say "None available" since fallback provides system commands
        tool_count = len(tool_instruction_ids) if tool_instruction_ids else "system"
        inventory_lines.append(f"✅ TOOLS AVAILABLE: {tool_count} tool instruction(s) loaded")
        inventory_lines.append("   Review the tool instructions section above")

        # Knowledge availability (web search and workflow data)
        web_search_entries = [k for k in knowledge_entries if k.domain == "web_search"]
        workflow_entries = [k for k in knowledge_entries if k.domain == "workflow_task"]

        if web_search_entries:
            inventory_lines.append(f"✅ WEB SEARCH DATA: {len(web_search_entries)} result(s) available")
            inventory_lines.append("   DO NOT request web search - data is already here")

            # Show sources if available
            sources = set()
            for entry in web_search_entries[:3]:  # Sample first 3
                if isinstance(entry.content, dict) and 'source_url' in entry.content:
                    sources.add(entry.content['source_url'][:50])
            if sources:
                inventory_lines.append(f"   Sources: {', '.join(list(sources)[:2])}...")
        else:
            inventory_lines.append("❌ WEB SEARCH DATA: No results yet")
            inventory_lines.append("   You MAY request web search if needed for current information")

        # Previous agent outputs
        if context_history:
            inventory_lines.append(f"✅ PREVIOUS AGENT OUTPUTS: {len(context_history)} message(s)")
            agent_types = set([h.get('agent_type', 'unknown') for h in context_history])
            inventory_lines.append(f"   Agent types: {', '.join(agent_types)}")
            inventory_lines.append("   BUILD ON THIS WORK - don't repeat what they found")
        else:
            inventory_lines.append("❌ PREVIOUS OUTPUTS: You are the first agent")
            inventory_lines.append("   You are pioneering this workflow")

        # Workflow knowledge
        if workflow_entries:
            inventory_lines.append(f"📚 WORKFLOW KNOWLEDGE: {len(workflow_entries)} stored insight(s)")
            inventory_lines.append("   Previous workflows have relevant information")

        # Known file locations (meta-learning from previous discoveries)
        if task_description:
            known_locations = self._get_known_file_locations(task_description)
            if known_locations:
                inventory_lines.append("")
                inventory_lines.append(known_locations)
                inventory_lines.append("   USE THESE PATHS DIRECTLY - no need to search again")

        # Knowledge coverage gaps (Phase 5 - Epistemic Awareness)
        if coverage_report:
            inventory_lines.append("")
            if coverage_report.missing_domains or coverage_report.weak_domains:
                inventory_lines.append("⚠️ KNOWLEDGE GAPS DETECTED:")
                if coverage_report.missing_domains:
                    inventory_lines.append(f"   ❌ Missing: {', '.join(coverage_report.missing_domains)}")
                if coverage_report.weak_domains:
                    inventory_lines.append(f"   ⚠️ Weak: {', '.join(coverage_report.weak_domains)}")
                inventory_lines.append(f"   Overall coverage: {coverage_report.overall_coverage_score:.1%}")
                if "trigger_web_search" in coverage_report.recommendations:
                    inventory_lines.append("   → Consider web search for missing domain knowledge")
            else:
                inventory_lines.append(f"✅ KNOWLEDGE COVERAGE: {coverage_report.overall_coverage_score:.1%}")
                if coverage_report.covered_domains:
                    inventory_lines.append(f"   Covered: {', '.join(coverage_report.covered_domains)}")

        inventory_lines.append("=" * 60)

        return "\n".join(inventory_lines)

    def build_agent_context(self,
                           original_task: str,
                           agent_type: str,
                           agent_id: str,
                           current_time: float,
                           max_context_tokens: int = 40000,
                           message_limit: int = 10,
                           tool_requirements: Optional[Dict[str, bool]] = None) -> EnrichedContext:
        """
        Build enriched context for agent including previous agent outputs.

        Phase 2.4: Context versioning ensures race-free synchronization.
        Lock prevents concurrent context building that could miss recent messages.

        Args:
            original_task: The original user task
            agent_type: Type of agent (research, critic, analysis, synthesis)
            agent_id: ID of the agent receiving context
            current_time: Current simulation time
            max_context_tokens: Maximum tokens for context (for compression)
            message_limit: Maximum messages to retrieve
            tool_requirements: Optional dict specifying which tools are needed
                              {'needs_file_ops': bool, 'needs_web_search': bool, 'needs_system_commands': bool}

        Returns:
            EnrichedContext with task description, accumulated context, and conditional tool instructions
        """
        logger.info(f"="*60)
        logger.info(f"Building collaborative context for {agent_type} agent: {agent_id}")
        logger.info(f"  Task: {original_task[:80]}..." if len(original_task) > 80 else f"  Task: {original_task}")

        # Retrieve tool instructions based on task requirements (conditional injection)
        tool_instructions = ""
        tool_instruction_ids = []
        if tool_requirements and self.memory_facade:
            try:
                logger.info("🔧 TOOL INSTRUCTION RETRIEVAL")
                logger.info(f"  Tool requirements: {tool_requirements}")
                tool_instructions, tool_instruction_ids = self.memory_facade.retrieve_tool_instructions(tool_requirements)
                if tool_instructions:
                    logger.info(f"  ✓ Retrieved tool instructions ({len(tool_instructions)} chars)")
                    logger.info(f"  ✓ Tracking {len(tool_instruction_ids)} tool instruction IDs for meta-learning")
                else:
                    logger.info("  ℹ No tool instructions needed or available for this task")
            except Exception as e:
                logger.warning(f"  ⚠ Tool instruction retrieval failed: {e}")
                tool_instructions = ""
                tool_instruction_ids = []

        # Retrieve recent messages from CentralPost (exclude self)
        from src.communication.central_post import MessageType
        recent_messages = self.central_post.get_recent_messages(
            limit=message_limit,
            since_time=None,  # Get all messages
            message_types=[MessageType.STATUS_UPDATE, MessageType.SYSTEM_ACTION_RESULT],  # Include agent outputs AND system command results
            exclude_sender=agent_id  # Don't include own previous messages
        )

        logger.info(f"  Retrieved {len(recent_messages)} messages from CentralPost")
        if recent_messages:
            logger.info(f"  Previous agents: {[msg.sender_id for msg in recent_messages]}")

        # CRITICAL: Also retrieve ALL system commands executed in this workflow
        # This ensures agents can see what's already been done, even if messages were processed
        workflow_system_actions = []

        # DEBUG: Trace workflow_id retrieval
        has_workflow_attr = hasattr(self.central_post, '_current_workflow_id')
        workflow_id_value = self.central_post._current_workflow_id if has_workflow_attr else None
        logger.debug(f"🐛 DEBUG context_builder: has_workflow_attr={has_workflow_attr}, workflow_id_value={workflow_id_value}")

        if has_workflow_attr and workflow_id_value:
            workflow_id = self.central_post._current_workflow_id
            try:
                # DEBUG: Check what's in the cache before retrieval
                scm = self.central_post.system_command_manager
                all_workflow_ids = list(scm._executed_commands.keys()) if hasattr(scm, '_executed_commands') else []
                logger.debug(f"🐛 DEBUG context_builder: available workflow_ids in cache = {all_workflow_ids}")
                if workflow_id in scm._executed_commands:
                    cached_count = len(scm._executed_commands[workflow_id])
                    logger.debug(f"🐛 DEBUG context_builder: found {cached_count} cached commands for workflow_id={workflow_id}")
                else:
                    logger.debug(f"🐛 DEBUG context_builder: workflow_id={workflow_id} NOT FOUND in cache!")

                executed_commands = self.central_post.system_command_manager.get_workflow_executed_commands(workflow_id)
                workflow_system_actions = executed_commands
                logger.info(f"  Retrieved {len(executed_commands)} executed system commands for workflow {workflow_id}")
                logger.debug(f"🐛 DEBUG: workflow_system_actions length after assignment = {len(workflow_system_actions)}")
                if executed_commands:
                    logger.debug(f"🐛 DEBUG: First command info keys = {list(executed_commands[0].keys())}")
                    logger.debug(f"🐛 DEBUG: First command = {executed_commands[0].get('command', 'NO COMMAND KEY')[:100]}")
                for i, cmd_info in enumerate(executed_commands, 1):
                    result = cmd_info['result']
                    status = "✓" if result.success else "✗"
                    logger.info(f"    {i}. {status} {cmd_info['command'][:50]}...")
            except Exception as e:
                logger.warning(f"  Could not retrieve workflow system actions: {e}")

        # Build context history from messages
        context_history = []
        original_size = self.estimate_tokens(original_task)

        logger.info(f"  Building context history in chronological order...")
        for i, msg in enumerate(reversed(recent_messages), 1):  # Chronological order
            msg_content = msg.content

            # Handle different message types
            if msg.message_type == MessageType.STATUS_UPDATE:
                # Agent response
                agent_response = msg_content.get("content", "")
                agent_conf = msg_content.get("confidence", 0.0)
                sender_type = msg_content.get("agent_type", "unknown")

            elif msg.message_type == MessageType.SYSTEM_ACTION_RESULT:
                # System command result
                command = msg_content.get("command", "")
                stdout = msg_content.get("stdout", "")
                stderr = msg_content.get("stderr", "")
                success = msg_content.get("success", False)
                exit_code = msg_content.get("exit_code", -1)

                # Format as readable context entry
                agent_response = f"System Command Execution:\nCommand: {command}\nSuccess: {success}\nExit Code: {exit_code}"
                if stdout:
                    # Intelligent truncation: preserve head + tail for large outputs (match Path 2 limit)
                    max_output = 50000  # 50KB limit to prevent context overflow
                    if len(stdout) > max_output:
                        head_size = 40000
                        tail_size = 10000
                        truncated_chars = len(stdout) - max_output
                        stdout = f"{stdout[:head_size]}\n\n... [truncated {truncated_chars} chars] ...\n\n{stdout[-tail_size:]}"
                    agent_response += f"\nOutput: {stdout}"
                if stderr:
                    # Same truncation for stderr
                    if len(stderr) > 5000:
                        stderr = f"{stderr[:4000]}\n... [truncated] ...\n{stderr[-1000:]}"
                    agent_response += f"\nErrors: {stderr}"

                agent_conf = 1.0  # High confidence for actual command results
                sender_type = "system_action"

            else:
                # Unknown message type, skip
                continue

            context_entry = {
                "agent_id": msg.sender_id,
                "agent_type": sender_type,
                "response": agent_response,
                "confidence": agent_conf,
                "timestamp": msg.timestamp
            }

            context_history.append(context_entry)
            original_size += self.estimate_tokens(str(agent_response))

            # Log each context entry being added
            response_preview = agent_response[:60] + "..." if len(agent_response) > 60 else agent_response
            logger.info(f"    {i}. {sender_type} ({msg.sender_id}): conf={agent_conf:.2f}, '{response_preview}'")

        # Add workflow system actions to context (if not already included in messages)
        # These are commands that were executed but might have been processed before this agent spawned
        logger.debug(f"🐛 DEBUG: BEFORE check - workflow_system_actions is {'TRUTHY' if workflow_system_actions else 'EMPTY/FALSY'}, len={len(workflow_system_actions) if workflow_system_actions else 0}")
        logger.debug(f"🐛 DEBUG: context_history has {len(context_history)} entries BEFORE adding system actions")
        if workflow_system_actions:
            logger.info(f"  Adding {len(workflow_system_actions)} executed system commands to context...")
            for cmd_info in workflow_system_actions:
                result = cmd_info['result']
                command = cmd_info['command']

                # Create a formatted context entry for the system action
                action_response = f"System Command Executed:\nCommand: {command}\nSuccess: {result.success}\nExit Code: {result.exit_code}"
                if hasattr(result, 'stdout') and result.stdout:
                    # Intelligent truncation: preserve head + tail for large outputs
                    max_output = 50000  # 50KB limit to prevent context overflow
                    stdout = result.stdout
                    if len(stdout) > max_output:
                        head_size = 40000
                        tail_size = 10000
                        truncated_chars = len(stdout) - max_output
                        action_response += f"\nOutput: {stdout[:head_size]}\n\n... [truncated {truncated_chars} chars] ...\n\n{stdout[-tail_size:]}"
                    else:
                        action_response += f"\nOutput: {stdout}"
                if hasattr(result, 'stderr') and result.stderr:
                    # Same intelligent truncation for stderr
                    stderr = result.stderr
                    if len(stderr) > 5000:
                        action_response += f"\nErrors: {stderr[:4000]}\n... [truncated] ...\n{stderr[-1000:]}"
                    else:
                        action_response += f"\nErrors: {stderr}"

                context_entry = {
                    "agent_id": "system_executor",
                    "agent_type": "system_action",
                    "response": action_response,
                    "confidence": 1.0,  # System actions have definitive results
                    "timestamp": cmd_info.get('timestamp', 0)
                }

                context_history.append(context_entry)
                original_size += self.estimate_tokens(action_response)

                status_emoji = "✓" if result.success else "✗"
                logger.info(f"    {status_emoji} {command[:50]}...")

        logger.debug(f"🐛 DEBUG: context_history has {len(context_history)} entries AFTER adding system actions")

        # TOKEN BUDGET: With 50K context window, we have generous budgets
        # Reserve space for collaborative instructions (~500 tokens)
        available_tokens = max_context_tokens - 500
        cumulative_tokens = self.estimate_tokens(original_task)

        # Add all context entries - with 40K budget we shouldn't need to drop anything
        enforced_context_history = []
        for entry in reversed(context_history):  # Newest first
            entry_size = self.estimate_tokens(str(entry['response']))
            entry_type = entry.get('agent_type', 'unknown')

            if cumulative_tokens + entry_size <= available_tokens:
                # Entry fits within budget - add it
                enforced_context_history.insert(0, entry)  # Maintain chronological order
                cumulative_tokens += entry_size
                logger.debug(f"  ✓ Added {entry_type} entry ({entry_size} tokens)")
            else:
                # Don't skip system_action entries - truncate to fit remaining budget
                remaining_tokens = available_tokens - cumulative_tokens - 100  # Buffer for safety

                if remaining_tokens > 1000 and entry_type == 'system_action':
                    # Truncate to fit: agents should always see SOMETHING from file reads
                    max_chars = remaining_tokens * 4  # ~4 chars per token
                    response = entry['response']

                    if len(response) > max_chars:
                        head_size = int(max_chars * 0.8)
                        tail_size = int(max_chars * 0.2)
                        truncated_chars = len(response) - max_chars
                        entry['response'] = f"{response[:head_size]}\n\n... [truncated {truncated_chars} chars to fit budget] ...\n\n{response[-tail_size:]}"

                    enforced_context_history.insert(0, entry)
                    cumulative_tokens = available_tokens - 100
                    logger.info(f"  ⚠ Truncated {entry_type} entry to fit budget ({remaining_tokens} tokens available)")
                else:
                    # Only skip if truly no space or not a system_action
                    logger.warning(f"  ⚠️ Skipped {entry_type} entry: {entry_size} tokens would exceed budget ({cumulative_tokens}/{available_tokens})")

        # Report results
        if len(enforced_context_history) < len(context_history):
            logger.info(f"  Token budget: kept {len(enforced_context_history)}/{len(context_history)} entries ({cumulative_tokens}/{available_tokens} tokens)")
        else:
            logger.debug(f"  Token budget: all {len(context_history)} entries fit ({cumulative_tokens}/{available_tokens} tokens)")

        context_history = enforced_context_history
        original_size = cumulative_tokens

        # Apply compression if context is too large
        compressed_size = original_size
        compression_ratio = 1.0

        if self.context_compressor and original_size > max_context_tokens:
            logger.info(f"  Context size {original_size} exceeds limit {max_context_tokens}, applying compression...")

            # Combine all context history into a single text for compression
            combined_context = "\n\n".join([
                f"[{entry.get('agent_type', 'Unknown')}]: {entry.get('response', '')}"
                for entry in context_history
            ])

            try:
                # Apply actual compression using the configured strategy
                compressed_result = self.context_compressor.compress_context(
                    combined_context,
                    max_tokens=int(max_context_tokens * 0.8)  # Leave 20% buffer for headers
                )

                # Store both compressed and original for reference
                if compressed_result and compressed_result.compressed_text:
                    # Create compressed context entry
                    compressed_entry = {
                        "agent_type": "system_compression",
                        "response": compressed_result.compressed_text,
                        "confidence": 0.9,  # High confidence for compressed content
                        "original_entries": len(context_history),
                        "compression_method": compressed_result.method.value
                    }

                    # Keep most recent 2 entries uncompressed + compressed summary
                    recent_entries = context_history[-2:] if len(context_history) > 2 else context_history
                    context_history = [compressed_entry] + recent_entries

                    compressed_size = self.estimate_tokens(compressed_result.compressed_text)
                    for entry in recent_entries:
                        compressed_size += self.estimate_tokens(str(entry['response']))

                    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                    logger.info(f"  ✓ Compressed {compressed_result.original_entries} entries to {compressed_size} tokens (ratio: {compression_ratio:.2f})")
                    logger.info(f"  Compression method: {compressed_result.method.value}")
                else:
                    # Fallback to truncation if compression fails
                    logger.warning("  Compression returned empty result, falling back to truncation")
                    context_history = context_history[-5:]
                    compressed_size = sum(self.estimate_tokens(str(entry['response'])) for entry in context_history)
                    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

            except Exception as e:
                logger.warning(f"  Compression failed: {e}, falling back to truncation")
                context_history = context_history[-5:]
                compressed_size = sum(self.estimate_tokens(str(entry['response'])) for entry in context_history)
                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

        # Retrieve relevant knowledge (if available)
        logger.info("")
        logger.info("🔍 KNOWLEDGE RETRIEVAL")
        logger.info("="*60)

        knowledge_entries = []
        # Use MemoryFacade if available, fallback to knowledge_store for backward compatibility
        memory_system = self.memory_facade if self.memory_facade else self.knowledge_store
        if memory_system:
            logger.info("  ✓ Memory system available (via " + ("MemoryFacade" if self.memory_facade else "knowledge_store") + ")")
            try:
                import time as time_module
                from src.memory.knowledge_store import KnowledgeQuery, ConfidenceLevel
                from src.workflows.truth_assessment import detect_query_type, QueryType

                # Detect task complexity for adaptive knowledge limits
                # Use SynthesisEngine's classification for consistency (YAML-configurable patterns)
                task_complexity = self.central_post.synthesis_engine.classify_task_complexity(original_task)
                knowledge_limit = self._get_knowledge_limit_for_complexity(task_complexity)

                # Detect query type to determine appropriate freshness window
                current_time = time_module.time()
                query_type = detect_query_type(original_task)
                freshness_limits = {
                    QueryType.TIME: 300,           # 5 minutes for time queries
                    QueryType.DATE: 3600,          # 1 hour for date queries
                    QueryType.CURRENT_EVENT: 1800, # 30 minutes for current events
                    QueryType.GENERAL_FACT: 86400, # 24 hours for general facts
                    QueryType.ANALYSIS: 86400,     # 24 hours for analysis
                }
                max_age = freshness_limits.get(query_type, 3600)  # Default to 1 hour
                time_window_start = current_time - max_age

                logger.info(f"  📝 Query parameters:")
                logger.info(f"     - Domains: web_search, workflow_task")
                logger.info(f"     - Min confidence: MEDIUM")
                logger.info(f"     - Task complexity: {task_complexity} (adaptive limit: {knowledge_limit} entries)")
                logger.info(f"     - Query type: {query_type.value} (freshness: {max_age}s / {max_age/60:.1f} min)")
                logger.info(f"     - Time range: {int(time_window_start)} to {int(current_time)}")

                # CRITICAL: Retrieve from BOTH web_search and workflow_task domains
                # Web search results are stored in "web_search" domain (central_post.py)
                # Agent results are stored in "workflow_task" domain
                logger.info("  🔍 Calling memory system to retrieve knowledge...")

                # Use MemoryFacade if available, fallback to direct knowledge_store
                if self.memory_facade:
                    logger.info("  → Using MemoryFacade.retrieve_knowledge_with_query()")
                    relevant_knowledge = self.memory_facade.retrieve_knowledge_with_query(
                        KnowledgeQuery(
                            domains=None,  # Query ALL domains (Knowledge Brain + workflow)
                            min_confidence=ConfidenceLevel.MEDIUM,
                            time_range=None,  # No time restriction - include indexed Knowledge Brain docs
                            limit=knowledge_limit,  # Adaptive limit based on task complexity
                            task_type=agent_type,  # For meta-learning boost
                            task_complexity=task_complexity  # For future semantic search
                        )
                    )
                else:
                    # Fallback for backward compatibility
                    logger.info("  → Using knowledge_store.retrieve_knowledge() (fallback)")
                    relevant_knowledge = self.knowledge_store.retrieve_knowledge(
                        KnowledgeQuery(
                            domains=None,  # Query ALL domains (Knowledge Brain + workflow)
                            min_confidence=ConfidenceLevel.MEDIUM,
                            time_range=None,  # No time restriction - include indexed Knowledge Brain docs
                            limit=knowledge_limit,  # Adaptive limit based on task complexity
                            task_type=agent_type,
                            task_complexity=task_complexity
                        )
                    )
                knowledge_entries = relevant_knowledge

                logger.info(f"  ✓ Retrieved {len(knowledge_entries)} entries from knowledge store")

                if len(knowledge_entries) == 0:
                    logger.warning("")
                    logger.warning("  ⚠️ NO KNOWLEDGE ENTRIES FOUND!")
                    logger.warning("  ⚠️ This means either:")
                    logger.warning("     1. Web search hasn't run yet")
                    logger.warning("     2. Web search failed to store knowledge")
                    logger.warning(f"     3. Knowledge is older than freshness window ({max_age/60:.1f} min)")
                    logger.warning("  ⚠️ Agents will NOT have web search data!")
                else:
                    # Sort web_search entries first (they're typically most relevant for current queries)
                    # Note: Meta-learning boost already applied optimal ranking
                    knowledge_entries = sorted(knowledge_entries, key=lambda ke: (ke.domain != "web_search", ke.created_at), reverse=False)

                    # NEW: Filter by contextual relevance to avoid irrelevant facts
                    pre_filter_count = len(knowledge_entries)
                    knowledge_entries = self.relevance_evaluator.filter_by_relevance(
                        knowledge_entries,
                        original_task,
                        threshold=0.3  # Keep items with >30% relevance (lowered to be more permissive)
                    )
                    if len(knowledge_entries) < pre_filter_count:
                        logger.info(f"  🎯 Relevance filtering: {len(knowledge_entries)}/{pre_filter_count} entries kept")

                    # === TOKEN BUDGET SAFETY: Trim knowledge entries if needed ===
                    # Reserve token budget for knowledge entries (max 30% of available context)
                    knowledge_token_budget = int(max_context_tokens * 0.3)
                    total_knowledge_tokens = 0
                    trimmed_entries = []

                    for entry in knowledge_entries:
                        # Estimate tokens for this entry
                        content_str = str(entry.content)
                        entry_tokens = self.estimate_tokens(content_str)

                        # Keep entry if within budget
                        if total_knowledge_tokens + entry_tokens <= knowledge_token_budget:
                            trimmed_entries.append(entry)
                            total_knowledge_tokens += entry_tokens
                        else:
                            logger.info(f"  ⚠ Token budget reached: trimming remaining entries "
                                      f"({total_knowledge_tokens}/{knowledge_token_budget} tokens used)")
                            break

                    # Report if entries were trimmed
                    if len(trimmed_entries) < len(knowledge_entries):
                        logger.info(f"  🔧 Token safety: kept {len(trimmed_entries)}/{len(knowledge_entries)} "
                                  f"knowledge entries within budget ({total_knowledge_tokens} tokens)")

                    knowledge_entries = trimmed_entries

                    logger.info("")
                    logger.info(f"  📚 Knowledge Entries (sorted with web_search first):")
                    for i, ke in enumerate(knowledge_entries, 1):
                        # Extract content from dictionary if present
                        if isinstance(ke.content, dict):
                            # Prefer 'result' key which contains extracted web search info
                            content_str = ke.content.get('result', str(ke.content))
                        else:
                            content_str = str(ke.content)

                        # Use longer display for web_search domain
                        max_chars = 200 if ke.domain == "web_search" else 50
                        truncated = content_str[:max_chars] + "..." if len(content_str) > max_chars else content_str

                        # Add emoji for web_search entries
                        prefix = "🌐" if ke.domain == "web_search" else "📝"

                        logger.info(f"    {i}. {prefix} [{ke.domain}] {ke.confidence_level.value}")
                        logger.info(f"       Content: {truncated}")

                        # Log additional metadata for web_search entries
                        if ke.domain == "web_search" and isinstance(ke.content, dict):
                            if 'source_url' in ke.content:
                                logger.info(f"       Source: {ke.content['source_url']}")
                            if 'deep_search_used' in ke.content:
                                logger.info(f"       Deep search: {ke.content['deep_search_used']}")

            except Exception as e:
                logger.error(f"  ❌ Knowledge retrieval FAILED: {e}", exc_info=True)
                logger.error(f"  ❌ Agents will NOT have any knowledge entries!")
        else:
            logger.warning("  ⚠️ Knowledge store NOT AVAILABLE")
            logger.warning("  ⚠️ Agents will NOT have access to stored knowledge")

        logger.info("="*60)

        # Track statistics
        self.contexts_built += 1
        self.total_compression_ratio += compression_ratio

        # NEW: Build context inventory for agent comprehension
        logger.info("")
        logger.info("📦 BUILDING CONTEXT INVENTORY")
        context_inventory = self.build_context_inventory(
            tool_instructions=tool_instructions,
            tool_instruction_ids=tool_instruction_ids,
            knowledge_entries=knowledge_entries,
            context_history=context_history,
            task_description=original_task  # For meta-learning file location lookup
        )
        logger.info("  ✓ Context inventory generated for agent prompt")
        logger.info(f"    - Tools: {'Retrieved' if tool_instructions else 'Fallback will be used'}")
        logger.info(f"    - Tool requirements: {tool_requirements if tool_requirements else 'All (fallback)'}")
        logger.info(f"    - Web search data: {len([k for k in knowledge_entries if k.domain == 'web_search'])} entries")
        logger.info(f"    - Previous outputs: {len(context_history)} messages")

        # Get existing concepts from concept registry for terminology consistency
        existing_concepts = self.get_existing_concepts_for_prompt(max_concepts=10)
        logger.info(f"  ✓ Retrieved {len(self.concept_registry.get_all_concepts())} existing concepts from registry")

        # Phase 2.4: Increment version if new messages processed
        current_message_count = len(recent_messages) + len(workflow_system_actions)
        if current_message_count > self._last_message_count:
            self._context_version += 1
            self._last_message_count = current_message_count
            logger.info(f"  📌 Context version incremented to {self._context_version} ({current_message_count} messages)")

        enriched = EnrichedContext(
            task_description=original_task,
            context_history=context_history,
            original_context_size=original_size,
            compressed_context_size=compressed_size,
            compression_ratio=compression_ratio,
            knowledge_entries=knowledge_entries,
            message_count=len(recent_messages),
            tool_instructions=tool_instructions,  # Include conditional tool instructions
            tool_instruction_ids=tool_instruction_ids,  # Include IDs for meta-learning
            context_inventory=context_inventory,  # NEW: Include inventory for agent comprehension
            existing_concepts=existing_concepts,  # NEW: Include existing concepts for terminology consistency
            version=self._context_version  # Phase 2.4: Include version number
        )

        logger.info(f"  ✓ Collaborative context built successfully:")
        logger.info(f"    - Version: {self._context_version}")
        logger.info(f"    - Messages: {len(context_history)}")
        logger.info(f"    - Knowledge entries: {len(knowledge_entries)}")
        logger.info(f"    - Context size: {compressed_size} tokens (budget: {max_context_tokens})")
        logger.info(f"    - Compression ratio: {compression_ratio:.2f}")
        logger.info(f"="*60)

        return enriched

    def check_concept_before_use(self, concept_name: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if a concept has already been defined before agent uses it.

        Args:
            concept_name: Concept name to check
            agent_id: Agent ID requesting the check

        Returns:
            Dictionary with concept info if exists, None otherwise
        """
        concept = self.concept_registry.get_concept(concept_name)
        if concept:
            logger.debug(f"  ℹ️  {agent_id} checking concept '{concept_name}': "
                        f"Found (defined by {concept.source_agent})")
            return {
                'exists': True,
                'definition': concept.definition,
                'source_agent': concept.source_agent,
                'confidence': concept.confidence
            }
        return None

    def register_agent_concept(self, concept_name: str, definition: str,
                               agent_id: str, confidence: float = 0.7) -> bool:
        """
        Register a concept defined by an agent.

        Args:
            concept_name: Concept name
            definition: Concept definition
            agent_id: Agent defining the concept
            confidence: Agent's confidence in definition

        Returns:
            True if newly registered, False if duplicate
        """
        return self.concept_registry.register_concept(
            name=concept_name,
            definition=definition,
            source_agent=agent_id,
            confidence=confidence
        )

    def get_existing_concepts_for_prompt(self, max_concepts: int = 10) -> str:
        """
        Get formatted string of existing concepts for agent prompts.

        Args:
            max_concepts: Maximum number of concepts to include

        Returns:
            Formatted string of concepts
        """
        all_concepts = self.concept_registry.get_all_concepts()

        if not all_concepts:
            return "No concepts defined yet in this workflow."

        # Sort by usage count (most used first)
        sorted_concepts = sorted(all_concepts, key=lambda c: c.usage_count, reverse=True)
        top_concepts = sorted_concepts[:max_concepts]

        concept_lines = []
        for concept in top_concepts:
            concept_lines.append(
                f"  - **{concept.name}**: {concept.definition[:150]}... "
                f"(defined by {concept.source_agent})"
            )

        return "\n".join(concept_lines)

    def export_concept_registry(self, filepath: str = "analysis/improvement_registry.md") -> None:
        """
        Export concept registry to markdown file.

        Args:
            filepath: Path to output file
        """
        self.concept_registry.export_to_markdown(filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """Get context building statistics."""
        avg_compression = (self.total_compression_ratio / self.contexts_built
                          if self.contexts_built > 0 else 0.0)

        # Include concept registry stats
        registry_summary = self.concept_registry.get_summary()

        return {
            "contexts_built": self.contexts_built,
            "average_compression_ratio": avg_compression,
            "concept_registry": registry_summary
        }
