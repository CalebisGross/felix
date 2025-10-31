"""
Felix System Manager for GUI Integration

This module provides a unified system manager that initializes and coordinates
all Felix components, providing proper integration between the GUI and the
core Felix architecture.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory, Message, MessageType
from src.communication.spoke import SpokeManager
from src.llm.lm_studio_client import LMStudioClient
from src.llm.token_budget import TokenBudgetManager
from src.llm.web_search_client import WebSearchClient
from src.memory.knowledge_store import KnowledgeStore
from src.memory.task_memory import TaskMemory
from src.memory.context_compression import ContextCompressor, CompressionConfig, CompressionStrategy, CompressionLevel
from src.agents import ResearchAgent, AnalysisAgent, CriticAgent, PromptOptimizer
from src.agents.system_agent import SystemAgent
from src.agents.agent import AgentState

# Knowledge Brain imports
try:
    from src.knowledge import (
        KnowledgeDaemon, DaemonConfig,
        KnowledgeRetriever,
        EmbeddingProvider
    )
    KNOWLEDGE_BRAIN_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Knowledge Brain not available: {e}")
    KNOWLEDGE_BRAIN_AVAILABLE = False
    KnowledgeDaemon = None
    DaemonConfig = None
    KnowledgeRetriever = None
    EmbeddingProvider = None

logger = logging.getLogger(__name__)


@dataclass
class FelixConfig:
    """Configuration for Felix system."""
    # LM Studio settings
    lm_host: str = '127.0.0.1'
    lm_port: int = 1234

    # Helix geometry
    helix_top_radius: float = 3.0
    helix_bottom_radius: float = 0.5
    helix_height: float = 8.0
    helix_turns: float = 2.0

    # System limits
    max_agents: int = 25  # Increased from 15 to allow sufficient agents for collaboration
    base_token_budget: int = 2500

    # Memory
    memory_db_path: str = "felix_memory.db"
    knowledge_db_path: str = "felix_knowledge.db"

    # Context compression
    compression_target_length: int = 100
    compression_ratio: float = 0.3
    compression_strategy: str = "abstractive"

    # Features
    enable_metrics: bool = True
    enable_memory: bool = True
    enable_dynamic_spawning: bool = True
    enable_compression: bool = True
    enable_spoke_topology: bool = True
    verbose_llm_logging: bool = True  # Log detailed LLM requests/responses
    enable_streaming: bool = True  # Enable incremental token streaming for real-time communication
    streaming_batch_interval: float = 0.1  # Send partial updates every 100ms

    # Web search settings
    web_search_enabled: bool = False  # Enable web search for CentralPost (confidence-based)
    web_search_provider: str = "duckduckgo"  # Search provider: "duckduckgo" or "searxng"
    web_search_max_results: int = 5  # Maximum results per search query
    web_search_max_queries: int = 3  # Maximum queries per search session
    searxng_url: Optional[str] = None  # SearxNG instance URL (if using SearxNG)
    web_search_blocked_domains: Optional[List[str]] = None  # Domains to filter from results

    # Web search trigger configuration
    web_search_confidence_threshold: float = 0.7  # Trigger search when avg confidence < this
    web_search_min_samples: int = 1  # Minimum confidence scores before checking average
    web_search_cooldown: float = 10.0  # Seconds between web searches

    # Workflow early stopping configuration
    workflow_max_steps_simple: int = 5  # Max steps for simple tasks (confidence >= 0.75)
    workflow_max_steps_medium: int = 10  # Max steps for medium tasks (confidence >= 0.50)
    workflow_max_steps_complex: int = 20  # Max steps for complex tasks (confidence < 0.50)
    workflow_simple_threshold: float = 0.75  # Confidence threshold for simple tasks
    workflow_medium_threshold: float = 0.50  # Confidence threshold for medium tasks

    # Learning system configuration
    enable_learning: bool = True  # Enable adaptive learning systems
    learning_auto_apply: bool = True  # Auto-apply high-confidence recommendations (≥95%)
    learning_min_samples_patterns: int = 10  # Minimum samples for pattern recommendations
    learning_min_samples_calibration: int = 10  # Minimum samples for confidence calibration
    learning_min_samples_thresholds: int = 15  # Minimum samples for threshold learning (reduced from 20)

    # Knowledge Brain configuration
    enable_knowledge_brain: bool = False  # Enable autonomous knowledge brain (requires setup)
    knowledge_watch_dirs: List[str] = None  # Directories to watch for documents
    knowledge_embedding_mode: str = "auto"  # Embedding mode: auto/lm_studio/tfidf/fts5
    knowledge_auto_augment: bool = True  # Auto-augment workflows with relevant knowledge
    knowledge_daemon_enabled: bool = True  # Enable background daemon
    knowledge_refinement_interval: int = 3600  # Refinement interval in seconds (1 hour)
    knowledge_processing_threads: int = 2  # Number of processing threads
    knowledge_max_memory_mb: int = 512  # Maximum memory for processing (MB)
    knowledge_chunk_size: int = 1000  # Characters per chunk
    knowledge_chunk_overlap: int = 200  # Character overlap between chunks

    # Context Retrieval configuration (adaptive knowledge limits)
    knowledge_limit_simple: int = 8  # Knowledge entries for SIMPLE_FACTUAL tasks
    knowledge_limit_medium: int = 15  # Knowledge entries for MEDIUM complexity tasks
    knowledge_limit_complex: int = 25  # Knowledge entries for COMPLEX tasks
    enable_meta_learning_boost: bool = True  # Apply historical usefulness ranking
    meta_learning_min_usages: int = 2  # Minimum usages before boost applies
    enable_adaptive_limits: bool = True  # Use task complexity for adaptive limits


class AgentManager:
    """Manages active agents in the Felix system."""

    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_id -> agent instance
        self.agent_outputs: Dict[str, Dict[str, Any]] = {}  # agent_id -> output data
        self._agent_counter = 0

    def register_agent(self, agent) -> None:
        """Register a new agent."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")

    def deregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Deregistered agent: {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str):
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List:
        """Get all active agents."""
        return list(self.agents.values())

    def get_agent_count(self) -> int:
        """Get number of active agents."""
        return len(self.agents)

    def get_next_agent_id(self, agent_type: str) -> str:
        """Generate next agent ID."""
        agent_id = f"{agent_type}_{self._agent_counter:03d}"
        self._agent_counter += 1
        return agent_id

    def store_agent_output(self, agent_id: str, result) -> None:
        """
        Store comprehensive agent output including prompts and metrics for display in GUI.

        Args:
            agent_id: Agent identifier
            result: LLMResult object containing all processing data
        """
        self.agent_outputs[agent_id] = {
            "output": result.content,
            "confidence": result.confidence,
            "timestamp": result.timestamp,
            "system_prompt": result.system_prompt,
            "user_prompt": result.user_prompt,
            "temperature": result.temperature_used,
            "tokens_used": result.llm_response.tokens_used if result.llm_response else 0,
            "token_budget": result.token_budget_allocated,
            "processing_time": result.processing_time,
            "position_info": result.position_info,
            "collaborative_count": result.collaborative_context_count,
            "model": result.llm_response.model if result.llm_response else "unknown",
            "processing_stage": result.processing_stage
        }
        logger.debug(f"Stored output for agent {agent_id} (confidence: {result.confidence:.2f}, "
                    f"tokens: {result.llm_response.tokens_used if result.llm_response else 0}/{result.token_budget_allocated})")

    def get_agent_output(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get stored output for an agent."""
        return self.agent_outputs.get(agent_id)


class FelixSystem:
    """
    Unified Felix system manager that coordinates all components.

    This class provides a single point of integration between the GUI
    and the Felix architecture, ensuring all components work together.
    """

    def __init__(self, config: Optional[FelixConfig] = None):
        """Initialize Felix system with configuration."""
        self.config = config or FelixConfig()
        self.running = False

        # Core components (initialized on start)
        self.helix: Optional[HelixGeometry] = None
        self.lm_client: Optional[LMStudioClient] = None
        self.web_search_client: Optional[WebSearchClient] = None
        self.central_post: Optional[CentralPost] = None
        self.spoke_manager: Optional[SpokeManager] = None
        self.agent_factory: Optional[AgentFactory] = None
        self.token_budget_manager: Optional[TokenBudgetManager] = None
        self.prompt_optimizer: Optional[PromptOptimizer] = None

        # Agent management
        self.agent_manager = AgentManager()

        # Memory systems
        self.knowledge_store: Optional[KnowledgeStore] = None
        self.task_memory: Optional[TaskMemory] = None
        self.context_compressor: Optional[ContextCompressor] = None

        # Knowledge Brain components
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.knowledge_retriever: Optional[KnowledgeRetriever] = None
        self.knowledge_daemon: Optional[KnowledgeDaemon] = None

        # Current simulation time
        self._current_time = 0.0

    def start(self) -> bool:
        """
        Start the Felix system and initialize all components.

        Returns:
            True if system started successfully, False otherwise
        """
        if self.running:
            logger.warning("Felix system already running")
            return True

        try:
            logger.info("Starting Felix system...")

            # Initialize helix geometry
            self.helix = HelixGeometry(
                top_radius=self.config.helix_top_radius,
                bottom_radius=self.config.helix_bottom_radius,
                height=self.config.helix_height,
                turns=self.config.helix_turns
            )
            logger.info(f"Helix geometry initialized: {self.helix}")

            # Initialize LM Studio client
            base_url = f"http://{self.config.lm_host}:{self.config.lm_port}/v1"
            self.lm_client = LMStudioClient(
                base_url=base_url,
                verbose_logging=self.config.verbose_llm_logging
            )

            # Test LM Studio connection
            if not self.lm_client.test_connection():
                logger.error(f"Failed to connect to LM Studio at {base_url}")
                logger.error("Please ensure LM Studio is running with a model loaded")
                return False

            logger.info(f"Connected to LM Studio at {base_url}")

            # Initialize web search client (if enabled)
            if self.config.web_search_enabled:
                try:
                    self.web_search_client = WebSearchClient(
                        provider=self.config.web_search_provider,
                        max_results=self.config.web_search_max_results,
                        cache_enabled=True,
                        searxng_url=self.config.searxng_url,
                        blocked_domains=self.config.web_search_blocked_domains
                    )
                    blocked_info = f" (blocking: {', '.join(self.web_search_client.blocked_domains)})" if self.web_search_client.blocked_domains else ""
                    logger.info(f"Web search client initialized (provider: {self.config.web_search_provider}{blocked_info})")
                except ImportError as e:
                    logger.error(f"Failed to initialize web search client: {e}")
                    logger.error("Install ddgs: pip install ddgs")
                    self.web_search_client = None
                except Exception as e:
                    logger.error(f"Web search client initialization failed: {e}")
                    self.web_search_client = None
            else:
                logger.info("Web search disabled in configuration")

            # Initialize token budget manager
            self.token_budget_manager = TokenBudgetManager(
                base_budget=self.config.base_token_budget
            )
            logger.info("Token budget manager initialized")

            # Initialize memory systems
            if self.config.enable_memory:
                self.knowledge_store = KnowledgeStore(self.config.knowledge_db_path)
                self.task_memory = TaskMemory(self.config.memory_db_path)
                logger.info("Memory systems initialized")

            # Initialize context compressor
            if self.config.enable_compression:
                # Map config strategy to CompressionStrategy enum
                strategy = (CompressionStrategy.ABSTRACTIVE_SUMMARY
                           if self.config.compression_strategy == "abstractive"
                           else CompressionStrategy.HIERARCHICAL_SUMMARY)

                compression_config = CompressionConfig(
                    max_context_size=self.config.compression_target_length * 10,  # Target length in words, convert to tokens
                    strategy=strategy,
                    level=CompressionLevel.MODERATE
                )
                self.context_compressor = ContextCompressor(config=compression_config)
                logger.info(f"Context compressor initialized (strategy: {strategy.value}, max_context: {compression_config.max_context_size})")

            # Initialize Knowledge Brain components (if enabled)
            if self.config.enable_knowledge_brain and KNOWLEDGE_BRAIN_AVAILABLE:
                try:
                    logger.info("Initializing Knowledge Brain...")

                    # 1. Create embedding provider
                    self.embedding_provider = EmbeddingProvider(
                        lm_studio_client=self.lm_client,
                        db_path=self.config.knowledge_db_path
                    )
                    logger.info(f"  Embedding provider initialized (tier: {self.embedding_provider.active_tier.value})")

                    # 2. Create knowledge retriever
                    self.knowledge_retriever = KnowledgeRetriever(
                        knowledge_store=self.knowledge_store,
                        embedding_provider=self.embedding_provider,
                        enable_meta_learning=True
                    )
                    logger.info("  Knowledge retriever initialized")

                    # 3. Create daemon config
                    daemon_config = DaemonConfig(
                        watch_directories=self.config.knowledge_watch_dirs or ['./knowledge_sources'],
                        enable_batch_processing=True,
                        enable_refinement=True,
                        enable_file_watching=True,
                        refinement_interval=self.config.knowledge_refinement_interval,
                        processing_threads=self.config.knowledge_processing_threads,
                        max_memory_mb=self.config.knowledge_max_memory_mb,
                        chunk_size=self.config.knowledge_chunk_size,
                        chunk_overlap=self.config.knowledge_chunk_overlap
                    )

                    # 4. Create knowledge daemon
                    self.knowledge_daemon = KnowledgeDaemon(
                        config=daemon_config,
                        knowledge_store=self.knowledge_store,
                        llm_client=self.lm_client
                    )
                    logger.info("  Knowledge daemon initialized")

                    # 5. Start daemon if enabled
                    if self.config.knowledge_daemon_enabled:
                        self.knowledge_daemon.start()
                        logger.info("  Knowledge daemon started")

                    logger.info("✓ Knowledge Brain fully operational")

                except Exception as e:
                    logger.error(f"Failed to initialize Knowledge Brain: {e}")
                    logger.error("Continuing without Knowledge Brain...")
                    self.embedding_provider = None
                    self.knowledge_retriever = None
                    self.knowledge_daemon = None
            elif self.config.enable_knowledge_brain and not KNOWLEDGE_BRAIN_AVAILABLE:
                logger.warning("Knowledge Brain enabled but dependencies not available")

            # Initialize prompt optimizer
            self.prompt_optimizer = PromptOptimizer()
            logger.info("Prompt optimizer initialized")

            # Initialize central post
            self.central_post = CentralPost(
                max_agents=self.config.max_agents,
                enable_metrics=self.config.enable_metrics,
                enable_memory=self.config.enable_memory,
                memory_db_path=self.config.memory_db_path,
                llm_client=self.lm_client,  # For CentralPost synthesis capability
                web_search_client=self.web_search_client,  # For Research agents
                web_search_confidence_threshold=self.config.web_search_confidence_threshold,
                web_search_min_samples=self.config.web_search_min_samples,
                web_search_cooldown=self.config.web_search_cooldown,
                knowledge_store=self.knowledge_store  # CRITICAL: Share the same knowledge_store instance!
            )
            logger.info("Central post initialized with synthesis capability and shared knowledge store")

            # Initialize spoke manager (for O(N) communication topology)
            if self.config.enable_spoke_topology:
                self.spoke_manager = SpokeManager(self.central_post)
                logger.info("Spoke manager initialized")

            # Initialize agent factory
            self.agent_factory = AgentFactory(
                helix=self.helix,
                llm_client=self.lm_client,
                token_budget_manager=self.token_budget_manager,
                enable_dynamic_spawning=self.config.enable_dynamic_spawning,
                max_agents=self.config.max_agents,
                token_budget_limit=self.config.base_token_budget * self.config.max_agents,
                web_search_client=self.web_search_client,
                max_web_queries=self.config.web_search_max_queries
            )
            logger.info("Agent factory initialized")

            self.running = True
            logger.info("Felix system started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start Felix system: {e}", exc_info=True)
            self.running = False
            return False

    def stop(self) -> None:
        """Stop the Felix system and cleanup resources."""
        if not self.running:
            logger.warning("Felix system not running")
            return

        try:
            logger.info("Stopping Felix system...")

            # Deregister all agents
            agent_ids = list(self.agent_manager.agents.keys())
            for agent_id in agent_ids:
                self.agent_manager.deregister_agent(agent_id)
                # Spoke manager handles deregistration
                if self.spoke_manager:
                    self.spoke_manager.remove_spoke(agent_id)
                elif self.central_post:
                    # Fallback if spoke manager not used
                    self.central_post.deregister_agent(agent_id)

            # Shutdown spoke manager
            if self.spoke_manager:
                self.spoke_manager.shutdown_all()
                logger.info("Spoke manager shutdown")

            # Shutdown central post
            if self.central_post:
                self.central_post.shutdown()

            # Stop knowledge daemon
            if self.knowledge_daemon:
                try:
                    self.knowledge_daemon.stop()
                    logger.info("Knowledge daemon stopped")
                except Exception as e:
                    logger.warning(f"Error stopping knowledge daemon: {e}")

            # Close LM client if it has a close method
            if self.lm_client and hasattr(self.lm_client, 'close_async'):
                try:
                    asyncio.run(self.lm_client.close_async())
                except Exception as e:
                    logger.warning(f"Error closing LM client: {e}")

            self.running = False
            logger.info("Felix system stopped")

        except Exception as e:
            logger.error(f"Error stopping Felix system: {e}", exc_info=True)

    def spawn_agent(self, agent_type: str, domain: str = "general",
                   spawn_time: Optional[float] = None) -> Optional[Any]:
        """
        Spawn a new agent of the specified type.

        Args:
            agent_type: Type of agent ("research", "analysis", "synthesis", "critic")
            domain: Domain/focus for the agent
            spawn_time: Optional spawn time (defaults to current time)

        Returns:
            Spawned agent instance or None if spawning failed
        """
        if not self.running:
            logger.error("Cannot spawn agent: Felix system not running")
            return None

        try:
            # Use current time if not specified
            if spawn_time is None:
                spawn_time = self._current_time

            # Generate agent ID
            agent_id = self.agent_manager.get_next_agent_id(agent_type.lower())

            # Create agent based on type
            agent = None
            if agent_type.lower() == "research":
                agent = ResearchAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=self.helix,
                    llm_client=self.lm_client,
                    research_domain=domain,
                    token_budget_manager=self.token_budget_manager,
                    max_tokens=800
                )
            elif agent_type.lower() == "analysis":
                agent = AnalysisAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=self.helix,
                    llm_client=self.lm_client,
                    analysis_type=domain,
                    token_budget_manager=self.token_budget_manager,
                    max_tokens=800
                )
            elif agent_type.lower() == "critic":
                agent = CriticAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=self.helix,
                    llm_client=self.lm_client,
                    review_focus=domain,
                    token_budget_manager=self.token_budget_manager,
                    max_tokens=800
                )
            elif agent_type.lower() == "system":
                agent = SystemAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=self.helix,
                    llm_client=self.lm_client,
                    max_tokens=1500,  # SystemAgent uses higher token budget for precise commands
                    token_budget_manager=self.token_budget_manager
                )
            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return None

            # Register agent with systems
            self.agent_manager.register_agent(agent)

            # Use spoke topology if enabled, otherwise direct central post registration
            if self.spoke_manager:
                # Create spoke connection (automatically registers with central post)
                self.spoke_manager.create_spoke(agent)
                logger.info(f"Spawned {agent_type} agent: {agent_id} (domain: {domain}) with spoke connection")
            elif self.central_post:
                # Fallback to direct registration
                self.central_post.register_agent(agent)
                logger.info(f"Spawned {agent_type} agent: {agent_id} (domain: {domain})")

            return agent

        except Exception as e:
            logger.error(f"Failed to spawn {agent_type} agent: {e}", exc_info=True)
            return None

    def send_task_to_agent(self, agent_id: str, task_description: str) -> Optional[Dict[str, Any]]:
        """
        Send a task to a specific agent.

        Args:
            agent_id: ID of the target agent
            task_description: Description of the task

        Returns:
            Task result or None if failed
        """
        if not self.running:
            logger.error("Cannot send task: Felix system not running")
            return None

        try:
            agent = self.agent_manager.get_agent(agent_id)
            if not agent:
                logger.error(f"Agent not found: {agent_id}")
                return None

            # Create task
            from src.agents.llm_agent import LLMTask
            task = LLMTask(
                task_id=f"task_{int(self._current_time * 1000)}",
                description=task_description,
                context="User-initiated task from GUI"
            )

            # Spawn agent if not already spawned
            if agent.state == AgentState.WAITING:
                agent.spawn(self._current_time, task)

            # Process task
            result = agent.process_task_with_llm(task, self._current_time)

            # Store result in knowledge base
            if self.central_post and result:
                self.central_post.store_agent_result_as_knowledge(
                    agent_id=agent_id,
                    content=result.content,
                    confidence=result.confidence,
                    domain="gui_task"
                )

            return {
                "agent_id": agent_id,
                "content": result.content if result else None,
                "confidence": result.confidence if result else 0.0
            }

        except Exception as e:
            logger.error(f"Failed to send task to agent {agent_id}: {e}", exc_info=True)
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        if not self.running:
            return {
                "running": False,
                "agents": 0,
                "messages": 0
            }

        status = {
            "running": True,
            "agents": self.agent_manager.get_agent_count(),
            "current_time": self._current_time
        }

        if self.central_post:
            metrics = self.central_post.get_performance_summary()
            status.update({
                "messages_processed": metrics.get("total_messages_processed", 0),
                "message_throughput": metrics.get("message_throughput", 0.0),
                "active_connections": metrics.get("active_connections", 0)
            })

        if self.config.enable_memory and self.central_post:
            memory_summary = self.central_post.get_memory_summary()
            status.update({
                "knowledge_entries": memory_summary.get("knowledge_entries", 0),
                "task_patterns": memory_summary.get("task_patterns", 0)
            })

        # Add spoke manager stats
        if self.spoke_manager:
            spoke_summary = self.spoke_manager.get_connection_summary()
            status.update({
                "spoke_connections": spoke_summary.get("connected_spokes", 0),
                "total_spokes": spoke_summary.get("total_spokes", 0),
                "spoke_messages": spoke_summary.get("total_messages_sent", 0)
            })

        # Add compression info if enabled
        if self.context_compressor:
            status.update({
                "compression_enabled": True,
                "compression_strategy": self.config.compression_strategy
            })

        return status

    @property
    def dynamic_spawner(self):
        """Get dynamic spawning system if available."""
        if self.agent_factory and hasattr(self.agent_factory, 'dynamic_spawner'):
            return self.agent_factory.dynamic_spawner
        return None

    def analyze_team_needs(self, processed_messages: List[Message]) -> List[Any]:
        """
        Analyze current team composition and spawn agents if needed.

        This exposes the dynamic spawning system for GUI access.

        Args:
            processed_messages: Recent messages for analysis

        Returns:
            List of newly spawned agents
        """
        if self.agent_factory:
            current_agents = self.agent_manager.get_all_agents()
            new_agents = self.agent_factory.assess_team_needs(
                processed_messages,
                self._current_time,
                current_agents
            )

            # Register new agents with the system
            for agent in new_agents:
                self.agent_manager.register_agent(agent)
                if self.spoke_manager:
                    self.spoke_manager.create_spoke(agent)
                elif self.central_post:
                    self.central_post.register_agent(agent)

            return new_agents
        return []

    def compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress context using the context compressor.

        Args:
            context: Context dictionary to compress

        Returns:
            Compressed context or original if compression fails
        """
        if self.context_compressor:
            try:
                compressed = self.context_compressor.compress_context(context)
                # CompressedContext has a 'content' attribute with the compressed data
                if hasattr(compressed, 'content'):
                    return compressed.content
                return context
            except Exception as e:
                logger.warning(f"Context compression failed: {e}")
                return context
        return context

    def advance_time(self, delta: float = 0.1) -> None:
        """Advance simulation time."""
        self._current_time += delta

        # Update all active agents
        for agent in self.agent_manager.get_all_agents():
            if agent.state == AgentState.ACTIVE:
                agent.update_position(self._current_time)

    def run_workflow(self, task_input: str, progress_callback=None, max_steps_override=None,
                    parent_workflow_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a workflow through the Felix system.

        This method properly integrates workflows with the Felix system:
        - Uses CentralPost for O(N) hub-spoke communication
        - Uses AgentFactory for intelligent agent spawning
        - Uses shared LLM client across all operations
        - Uses memory systems for persistent knowledge
        - Enables dynamic spawning based on confidence monitoring
        - Agents spawned are visible in the Agents tab
        - Supports conversation continuity via parent_workflow_id

        Args:
            task_input: Task description to process (or follow-up question if continuing)
            progress_callback: Optional callback(status, progress_percentage)
            max_steps_override: Optional override for max workflow steps (None = adaptive)
            parent_workflow_id: Optional ID of parent workflow to continue from

        Returns:
            Dictionary with workflow results
        """
        if not self.running:
            return {
                "status": "failed",
                "error": "Felix system not running"
            }

        try:
            # Use proper Felix workflow implementation
            from src.workflows.felix_workflow import run_felix_workflow

            logger.info("Running workflow through Felix framework")

            # Run workflow using Felix system components
            result = run_felix_workflow(self, task_input, progress_callback, max_steps_override, parent_workflow_id)

            logger.info(f"Workflow completed: {result.get('status')}")

            return result

        except Exception as e:
            logger.error(f"Error running workflow: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
