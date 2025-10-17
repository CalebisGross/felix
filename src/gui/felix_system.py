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
from src.llm.lm_studio_client import LMStudioClient
from src.llm.token_budget import TokenBudgetManager
from src.memory.knowledge_store import KnowledgeStore
from src.memory.task_memory import TaskMemory
from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent

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
    max_agents: int = 15
    base_token_budget: int = 2048

    # Memory
    memory_db_path: str = "felix_memory.db"
    knowledge_db_path: str = "felix_knowledge.db"

    # Features
    enable_metrics: bool = True
    enable_memory: bool = True
    enable_dynamic_spawning: bool = True


class AgentManager:
    """Manages active agents in the Felix system."""

    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_id -> agent instance
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
        self.central_post: Optional[CentralPost] = None
        self.agent_factory: Optional[AgentFactory] = None
        self.token_budget_manager: Optional[TokenBudgetManager] = None

        # Agent management
        self.agent_manager = AgentManager()

        # Memory systems
        self.knowledge_store: Optional[KnowledgeStore] = None
        self.task_memory: Optional[TaskMemory] = None

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
            self.lm_client = LMStudioClient(base_url=base_url)

            # Test LM Studio connection
            if not self.lm_client.test_connection():
                logger.error(f"Failed to connect to LM Studio at {base_url}")
                logger.error("Please ensure LM Studio is running with a model loaded")
                return False

            logger.info(f"Connected to LM Studio at {base_url}")

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

            # Initialize central post
            self.central_post = CentralPost(
                max_agents=self.config.max_agents,
                enable_metrics=self.config.enable_metrics,
                enable_memory=self.config.enable_memory,
                memory_db_path=self.config.memory_db_path
            )
            logger.info("Central post initialized")

            # Initialize agent factory
            self.agent_factory = AgentFactory(
                helix=self.helix,
                llm_client=self.lm_client,
                token_budget_manager=self.token_budget_manager,
                enable_dynamic_spawning=self.config.enable_dynamic_spawning,
                max_agents=self.config.max_agents,
                token_budget_limit=self.config.base_token_budget * self.config.max_agents
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
                if self.central_post:
                    self.central_post.deregister_agent(agent_id)

            # Shutdown central post
            if self.central_post:
                self.central_post.shutdown()

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
            elif agent_type.lower() == "synthesis":
                agent = SynthesisAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=self.helix,
                    llm_client=self.lm_client,
                    output_format=domain,
                    token_budget_manager=self.token_budget_manager,
                    max_tokens=1200
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
            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return None

            # Register agent with systems
            self.agent_manager.register_agent(agent)
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
            if agent.state == "waiting":
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

        return status

    def advance_time(self, delta: float = 0.1) -> None:
        """Advance simulation time."""
        self._current_time += delta

        # Update all active agents
        for agent in self.agent_manager.get_all_agents():
            if agent.state == "active":
                agent.update_position(self._current_time)
