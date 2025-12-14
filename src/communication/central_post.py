"""
Central coordination system for the Felix Framework.

The central post manages communication and coordination between agents,
implementing the hub of the spoke-based communication model from thefelix.md.

Mathematical Foundation:
- Spoke communication: O(N) message complexity vs O(N²) mesh topology
- Maximum communication distance: R_top (helix outer radius)
- Performance metrics for efficiency benchmarking and statistical analysis

Key Features:
- Agent registration and connection management
- FIFO message queuing with guaranteed ordering
- Performance metrics collection (throughput, latency, overhead ratios)

Implementation supports rigorous performance testing and communication efficiency measurement.
"""

import time
import uuid
import random
import logging
import threading
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Callable
from collections import deque
from queue import Queue, Empty
import asyncio

# Memory system imports
from src.memory.knowledge_store import KnowledgeStore, KnowledgeEntry, KnowledgeType, ConfidenceLevel
from src.memory.task_memory import TaskMemory, TaskPattern, TaskOutcome
from src.memory.context_compression import ContextCompressor, CompressionStrategy

# System execution imports (for system autonomy)
from src.execution import SystemExecutor, TrustManager, CommandHistory, TrustLevel, CommandResult

# Message types (extracted to avoid circular imports)
from src.communication.message_types import Message, MessageType

# Extracted component imports
from src.communication.system_command_manager import SystemCommandManager
from src.communication.web_search_coordinator import WebSearchCoordinator
from src.communication.synthesis_engine import SynthesisEngine
from src.communication.memory_facade import MemoryFacade
from src.communication.streaming_coordinator import StreamingCoordinator
from src.communication.performance_monitor import PerformanceMonitor

# Centralized .felixignore support for filtering command output
from src.core.felixignore import filter_command_output

# Dynamic spawning imports - moved to avoid circular imports

if TYPE_CHECKING:
    from agents.llm_agent import LLMAgent
    from core.helix_geometry import HelixGeometry
    from llm.lm_studio_client import LMStudioClient
    from llm.token_budget import TokenBudgetManager

# Set up logging
logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for tracking agents by their helical position and phase.

    This class enables agent awareness through position-based tracking,
    supporting the helical convergence model where agents move from
    exploration (top) to synthesis (bottom).

    Phase definitions:
    - Exploration (0.0-0.3): Wide radius, broad discovery
    - Analysis (0.3-0.7): Converging radius, pattern identification
    - Synthesis (0.7-1.0): Narrow radius, focused output
    """

    def __init__(self):
        """Initialize the agent registry with phase-based tracking."""
        # Track agents by their current phase
        self._agents_by_phase: Dict[str, Dict[str, Dict[str, Any]]] = {
            'exploration': {},  # depth_ratio 0.0-0.3
            'analysis': {},     # depth_ratio 0.3-0.7
            'synthesis': {}     # depth_ratio 0.7-1.0
        }

        # Position index: agent_id -> position info
        self._position_index: Dict[str, Dict[str, Any]] = {}

        # Capability matrix: agent_id -> capabilities
        self._capability_matrix: Dict[str, Dict[str, Any]] = {}

        # Agent metadata: agent_id -> full metadata
        self._agent_metadata: Dict[str, Dict[str, Any]] = {}

        # Performance tracking: agent_id -> performance metrics
        self._performance_metrics: Dict[str, Dict[str, Any]] = {}

        # Collaboration tracking: agent_id -> list of influenced agents
        self._collaboration_graph: Dict[str, List[str]] = {}

        # Confidence history: track confidence over time
        self._confidence_history: List[Tuple[float, str, float]] = []  # (time, agent_id, confidence)

        # Live agents database for cross-process visibility
        self._live_db_path = "felix_live_agents.db"
        self._init_live_db()

    def _init_live_db(self) -> None:
        """Initialize SQLite database for live agent positions (cross-process visibility)."""
        try:
            with sqlite3.connect(self._live_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS live_agents (
                        agent_id TEXT PRIMARY KEY,
                        agent_type TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        progress REAL DEFAULT 0.0,
                        x_position REAL DEFAULT 0.0,
                        y_position REAL DEFAULT 0.0,
                        z_position REAL DEFAULT 0.0,
                        confidence REAL DEFAULT 0.5,
                        last_update REAL NOT NULL,
                        status TEXT DEFAULT 'active',
                        created_at REAL DEFAULT 0.0
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_live_agents_last_update ON live_agents(last_update DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_live_agents_status ON live_agents(status)")
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to initialize live agents database: {e}")

    def _persist_agent_position(self, agent_id: str, position_info: Dict[str, Any]) -> None:
        """Persist agent position to SQLite for cross-process visibility."""
        try:
            metadata = self._agent_metadata.get(agent_id, {})
            metrics = self._performance_metrics.get(agent_id, {})
            depth_ratio = position_info.get('depth_ratio', 0.0)

            with sqlite3.connect(self._live_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO live_agents
                    (agent_id, agent_type, phase, progress, x_position, y_position,
                     z_position, confidence, last_update, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_id,
                    metadata.get('agent_type', 'unknown'),
                    position_info.get('phase', self._get_phase_from_depth(depth_ratio)),
                    depth_ratio,
                    position_info.get('x', 0.0),
                    position_info.get('y', 0.0),
                    position_info.get('z', 0.0),
                    metrics.get('avg_confidence', 0.5),
                    time.time(),
                    'active',
                    metadata.get('registered_at', time.time())
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist agent position: {e}")

    def mark_agent_complete(self, agent_id: str) -> None:
        """Mark an agent as completed in the database."""
        try:
            with sqlite3.connect(self._live_db_path) as conn:
                conn.execute(
                    "UPDATE live_agents SET status = 'completed', last_update = ? WHERE agent_id = ?",
                    (time.time(), agent_id)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to mark agent complete: {e}")

    def clear_old_agents(self, max_age_seconds: float = 300.0) -> int:
        """Remove agents not updated within max_age_seconds. Returns count removed."""
        try:
            cutoff = time.time() - max_age_seconds
            with sqlite3.connect(self._live_db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM live_agents WHERE last_update < ? OR status = 'completed'",
                    (cutoff,)
                )
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.warning(f"Failed to clear old agents: {e}")
            return 0

    def register_agent(self, agent_id: str, metadata: Dict[str, Any]) -> None:
        """
        Register an agent with its initial metadata.

        Args:
            agent_id: Unique identifier for the agent
            metadata: Agent metadata including type, spawn_time, capabilities
        """
        self._agent_metadata[agent_id] = metadata

        # Extract capabilities if provided
        if 'capabilities' in metadata:
            self._capability_matrix[agent_id] = metadata['capabilities']

        # Initialize performance metrics
        self._performance_metrics[agent_id] = {
            'messages_sent': 0,
            'avg_confidence': 0.0,
            'processing_time': 0.0,
            'tokens_used': 0
        }

        # Initialize collaboration tracking
        self._collaboration_graph[agent_id] = []

        # Determine initial phase based on spawn_time
        spawn_time = metadata.get('spawn_time', 0.0)
        initial_phase = self._get_phase_from_depth(spawn_time)
        self._agents_by_phase[initial_phase][agent_id] = metadata

    def update_agent_position(self, agent_id: str, position_info: Dict[str, Any]) -> None:
        """
        Update an agent's position on the helix.

        Args:
            agent_id: Agent identifier
            position_info: Dictionary containing x, y, z, depth_ratio, radius
        """
        if agent_id not in self._agent_metadata:
            return

        # Store position
        self._position_index[agent_id] = position_info

        # Update phase tracking
        depth_ratio = position_info.get('depth_ratio', 0.0)
        new_phase = self._get_phase_from_depth(depth_ratio)

        # Remove from old phase if necessary
        for phase in self._agents_by_phase:
            if agent_id in self._agents_by_phase[phase] and phase != new_phase:
                del self._agents_by_phase[phase][agent_id]

        # Add to new phase
        self._agents_by_phase[new_phase][agent_id] = self._agent_metadata[agent_id]

        # Persist to database for cross-process visibility
        self._persist_agent_position(agent_id, position_info)

    def update_agent_performance(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update agent performance metrics.

        Args:
            agent_id: Agent identifier
            metrics: Performance metrics to update
        """
        if agent_id in self._performance_metrics:
            for key, value in metrics.items():
                if key == 'confidence':
                    # Track confidence history
                    self._confidence_history.append((time.time(), agent_id, value))
                    # Update running average
                    old_avg = self._performance_metrics[agent_id].get('avg_confidence', 0.0)
                    count = self._performance_metrics[agent_id].get('messages_sent', 0)
                    new_avg = (old_avg * count + value) / (count + 1) if count > 0 else value
                    self._performance_metrics[agent_id]['avg_confidence'] = new_avg
                else:
                    self._performance_metrics[agent_id][key] = value

    def record_collaboration(self, agent_id: str, influenced_agent_id: str) -> None:
        """
        Record that one agent influenced another.

        Args:
            agent_id: Agent that provided influence
            influenced_agent_id: Agent that was influenced
        """
        if agent_id in self._collaboration_graph:
            if influenced_agent_id not in self._collaboration_graph[agent_id]:
                self._collaboration_graph[agent_id].append(influenced_agent_id)

    def get_agents_in_phase(self, phase: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all agents currently in a specific phase.

        Args:
            phase: Phase name ('exploration', 'analysis', or 'synthesis')

        Returns:
            Dictionary of agents in the specified phase
        """
        return self._agents_by_phase.get(phase, {}).copy()

    def get_nearby_agents(self, agent_id: str, radius_threshold: float = 0.1) -> List[str]:
        """
        Find agents at similar depth on the helix.

        Args:
            agent_id: Reference agent
            radius_threshold: Maximum depth difference

        Returns:
            List of agent IDs within threshold
        """
        if agent_id not in self._position_index:
            return []

        ref_depth = self._position_index[agent_id].get('depth_ratio', 0.0)
        nearby = []

        for other_id, pos_info in self._position_index.items():
            if other_id != agent_id:
                other_depth = pos_info.get('depth_ratio', 0.0)
                if abs(ref_depth - other_depth) <= radius_threshold:
                    nearby.append(other_id)

        return nearby

    def get_convergence_status(self) -> Dict[str, Any]:
        """
        Analyze team convergence toward synthesis.

        Returns:
            Dictionary with convergence metrics
        """
        # Calculate confidence trend
        recent_confidence = self._get_recent_confidence_trend()

        # Count agents by phase
        phase_distribution = {
            phase: len(agents)
            for phase, agents in self._agents_by_phase.items()
        }

        # Check for synthesis readiness
        synthesis_agents = self._agents_by_phase.get('synthesis', {})
        synthesis_ready = False

        for agent_id in synthesis_agents:
            if agent_id in self._performance_metrics:
                avg_conf = self._performance_metrics[agent_id].get('avg_confidence', 0.0)
                if avg_conf >= 0.8:
                    synthesis_ready = True
                    break

        return {
            'confidence_trend': recent_confidence,
            'phase_distribution': phase_distribution,
            'synthesis_ready': synthesis_ready,
            'total_agents': sum(phase_distribution.values()),
            'collaboration_density': self._calculate_collaboration_density()
        }

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with agent information or None if not found
        """
        if agent_id not in self._agent_metadata:
            return None

        return {
            'metadata': self._agent_metadata[agent_id],
            'position': self._position_index.get(agent_id, {}),
            'capabilities': self._capability_matrix.get(agent_id, {}),
            'performance': self._performance_metrics.get(agent_id, {}),
            'collaborations': self._collaboration_graph.get(agent_id, []),
            'phase': self._get_current_phase(agent_id)
        }

    def get_active_agents(self) -> List[Dict[str, Any]]:
        """
        Get list of all active agents with basic info.

        Returns:
            List of agent summaries
        """
        agents = []
        for agent_id in self._agent_metadata:
            agents.append({
                'agent_id': agent_id,
                'agent_type': self._agent_metadata[agent_id].get('agent_type', 'unknown'),
                'phase': self._get_current_phase(agent_id),
                'depth_ratio': self._position_index.get(agent_id, {}).get('depth_ratio', 0.0),
                'avg_confidence': self._performance_metrics[agent_id].get('avg_confidence', 0.0)
            })

        return agents

    def deregister_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the registry.

        Args:
            agent_id: Agent to remove
        """
        # Remove from all tracking structures
        if agent_id in self._agent_metadata:
            del self._agent_metadata[agent_id]

        if agent_id in self._position_index:
            del self._position_index[agent_id]

        if agent_id in self._capability_matrix:
            del self._capability_matrix[agent_id]

        if agent_id in self._performance_metrics:
            del self._performance_metrics[agent_id]

        if agent_id in self._collaboration_graph:
            del self._collaboration_graph[agent_id]

        # Remove from phase tracking
        for phase in self._agents_by_phase:
            if agent_id in self._agents_by_phase[phase]:
                del self._agents_by_phase[phase][agent_id]

    def _get_phase_from_depth(self, depth_ratio: float) -> str:
        """
        Determine phase based on depth ratio.

        Args:
            depth_ratio: Position on helix (0.0 to 1.0)

        Returns:
            Phase name
        """
        if depth_ratio < 0.3:
            return 'exploration'
        elif depth_ratio < 0.7:
            return 'analysis'
        else:
            return 'synthesis'

    def _get_current_phase(self, agent_id: str) -> str:
        """
        Get current phase of an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Current phase name
        """
        depth_ratio = self._position_index.get(agent_id, {}).get('depth_ratio', 0.0)
        return self._get_phase_from_depth(depth_ratio)

    def _get_recent_confidence_trend(self, window: int = 10) -> str:
        """
        Analyze recent confidence trend.

        Args:
            window: Number of recent entries to analyze

        Returns:
            Trend indicator: 'IMPROVING', 'DECLINING', 'STABLE', or 'UNKNOWN'
        """
        if len(self._confidence_history) < 2:
            return 'UNKNOWN'

        recent = self._confidence_history[-window:] if len(self._confidence_history) >= window else self._confidence_history

        if len(recent) < 2:
            return 'UNKNOWN'

        # Calculate trend
        first_half_avg = sum(c[2] for c in recent[:len(recent)//2]) / (len(recent)//2)
        second_half_avg = sum(c[2] for c in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        diff = second_half_avg - first_half_avg

        if diff > 0.1:
            return 'IMPROVING'
        elif diff < -0.1:
            return 'DECLINING'
        else:
            return 'STABLE'

    def _calculate_collaboration_density(self) -> float:
        """
        Calculate how interconnected agents are.

        Returns:
            Collaboration density (0.0 to 1.0)
        """
        if not self._collaboration_graph:
            return 0.0

        total_possible = len(self._collaboration_graph) * (len(self._collaboration_graph) - 1)
        if total_possible == 0:
            return 0.0

        actual_connections = sum(len(collabs) for collabs in self._collaboration_graph.values())

        return min(1.0, actual_connections / total_possible)


class CentralPost:
    """
    Central coordination system managing all agent communication.
    
    The central post acts as the hub in the spoke-based communication model,
    processing messages from agents and coordinating task assignments.
    """
    
    def __init__(self, max_agents: int = 25, enable_metrics: bool = False,
                 enable_memory: bool = True, memory_db_path: str = "felix_memory.db",
                 llm_client: Optional["LMStudioClient"] = None,
                 web_search_client: Optional["WebSearchClient"] = None,
                 web_search_confidence_threshold: float = 0.7,
                 web_search_min_samples: int = 1,
                 web_search_cooldown: float = 10.0,
                 knowledge_store: Optional["KnowledgeStore"] = None,
                 config: Optional[Any] = None,
                 gui_mode: bool = False,
                 prompt_manager: Optional["PromptManager"] = None):
        """
        Initialize central post with configuration parameters.

        Args:
            max_agents: Maximum number of concurrent agent connections
            enable_metrics: Whether to collect performance metrics
            enable_memory: Whether to enable persistent memory systems
            memory_db_path: Path to the memory database file
            llm_client: Optional LLM client for CentralPost synthesis capability
            web_search_client: Optional web search client for Research agents
            web_search_confidence_threshold: Confidence threshold for triggering web search (default: 0.7)
            web_search_min_samples: Minimum confidence scores before checking average (default: 1)
            web_search_cooldown: Seconds between web searches to prevent spam (default: 10.0)
            knowledge_store: Optional shared KnowledgeStore instance (if None, creates new one)
            config: Optional FelixConfig for system-wide settings (auto-approval, etc.)
            gui_mode: Whether running in GUI mode (prevents CLI approval prompts)
            prompt_manager: Optional prompt manager for synthesis prompts
        """
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        self.enable_memory = enable_memory
        self.llm_client = llm_client  # For CentralPost synthesis
        self.web_search_client = web_search_client  # For Research agents
        self.config = config  # Store config for passing to subsystems
        self.gui_mode = gui_mode  # Store GUI mode flag for passing to subsystems
        self.prompt_manager = prompt_manager  # For synthesis prompts

        # Project root directory for command execution
        # Commands with relative paths will execute from this directory
        from pathlib import Path
        self.project_root = Path(__file__).parent.parent.parent.resolve()
        logger.debug(f"CentralPost project_root: {self.project_root}")
        
        # Connection management
        self._registered_agents: Dict[str, str] = {}  # agent_id -> connection_id
        self._connection_times: Dict[str, float] = {}  # agent_id -> registration_time

        # Agent awareness registry
        self.agent_registry = AgentRegistry()
        
        # Message processing (sync and async)
        self._message_queue: Queue = Queue()
        self._async_message_queue: Optional[asyncio.Queue] = None  # Lazy initialization
        self._processed_messages: List[Message] = []
        self._async_processors: List[asyncio.Task] = []
        
        # Performance metrics (for efficiency benchmarking)
        self._metrics_enabled = enable_metrics
        self._start_time = time.time()
        self._total_messages_processed = 0
        self._processing_times: List[float] = []
        self._overhead_ratios: List[float] = []
        self._scaling_metrics: Dict[int, float] = {}
        
        # Memory systems (Priority 5: Memory and Context Persistence)
        self._memory_enabled = enable_memory
        if enable_memory:
            # Use shared knowledge_store if provided, otherwise create new one
            if knowledge_store is not None:
                self.knowledge_store = knowledge_store
                logger.info(f"CentralPost using shared KnowledgeStore: {knowledge_store.storage_path}")
            else:
                # Fallback: create own knowledge store (for backward compatibility)
                self.knowledge_store = KnowledgeStore("felix_knowledge.db")
                logger.info("CentralPost created own KnowledgeStore: felix_knowledge.db")

            self.task_memory = TaskMemory(memory_db_path)  # Uses felix_memory.db or felix_task_memory.db
            self.context_compressor = ContextCompressor()
        else:
            self.knowledge_store = None
            self.task_memory = None
            self.context_compressor = None
        
        # System state
        self._is_active = True

        # Streaming state (for incremental token streaming)
        self._partial_thoughts: Dict[str, str] = {}  # agent_id -> accumulated content
        self._streaming_metadata: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata
        self._streaming_callbacks: List[Callable] = []  # GUI event listeners

        # Web search and confidence monitoring (configurable)
        self._web_search_trigger_threshold: float = web_search_confidence_threshold
        self._web_search_min_samples: int = web_search_min_samples
        self._recent_confidences: deque = deque(maxlen=10)  # Rolling window for average
        self._last_search_time: float = 0.0  # Prevent search spam
        self._search_cooldown: float = web_search_cooldown
        self._current_task_description: Optional[str] = None  # Track current workflow task
        self._current_workflow_id: Optional[str] = None  # Track current workflow ID for approval scoping
        self._search_count: int = 0  # Track number of searches for current task

        # System autonomy infrastructure
        self.system_executor = SystemExecutor()
        self.trust_manager = TrustManager()
        self.command_history = CommandHistory()
        self._action_results: Dict[str, CommandResult] = {}  # action_id -> result cache
        self._action_id_counter = 0

        # Approval system for REVIEW-level commands
        from src.execution.approval_manager import ApprovalManager
        self.approval_manager = ApprovalManager()
        self._action_approvals: Dict[str, str] = {}  # action_id -> approval_id mapping
        self._approval_events: Dict[str, threading.Event] = {}  # approval_id -> Event for workflow pausing

        # Command deduplication cache (per workflow session)
        self._executed_commands: Dict[str, Dict[str, CommandResult]] = {}  # workflow_id -> {command_hash -> result}

        # Live command output buffer for Terminal tab streaming
        self._live_command_outputs: Dict[int, List[tuple]] = {}  # execution_id -> [(output_line, stream_type), ...]

        # ========================================================================
        # COMPONENT INITIALIZATION - Extracted subsystems for clean separation
        # ========================================================================

        # 1. Performance Monitor
        self.performance_monitor = PerformanceMonitor(
            metrics_enabled=enable_metrics
        )

        # 2. Memory Facade
        self.memory_facade = MemoryFacade(
            knowledge_store=self.knowledge_store,
            task_memory=self.task_memory,
            context_compressor=self.context_compressor,
            memory_enabled=enable_memory
        )

        # 3. Streaming Coordinator
        self.streaming_coordinator = StreamingCoordinator()

        # 4. System Command Manager
        self.system_command_manager = SystemCommandManager(
            system_executor=self.system_executor,
            trust_manager=self.trust_manager,
            command_history=self.command_history,
            approval_manager=self.approval_manager,
            agent_registry=self.agent_registry,
            message_queue_callback=self.queue_message,
            config=config,  # Pass config for auto-approval flag (CLI mode)
            gui_mode=self.gui_mode  # Pass GUI mode flag to prevent CLI prompts
        )

        # 5. Web Search Coordinator
        self.web_search_coordinator = WebSearchCoordinator(
            web_search_client=self.web_search_client,
            knowledge_store=self.knowledge_store,
            llm_client=llm_client,
            agent_registry=self.agent_registry,
            message_queue_callback=self.queue_message,
            confidence_threshold=web_search_confidence_threshold,
            search_cooldown=web_search_cooldown,
            min_samples=web_search_min_samples
        )

        # 6. Synthesis Engine
        self.synthesis_engine = SynthesisEngine(
            llm_client=llm_client,
            get_recent_messages_callback=self.get_recent_messages,
            prompt_manager=self.prompt_manager
        )

        logger.info("✓ CentralPost initialized with 6 extracted components")
        logger.info("  System autonomy: SystemExecutor, TrustManager, CommandHistory, ApprovalManager")
        logger.info("  Components: PerformanceMonitor, MemoryFacade, StreamingCoordinator, SystemCommandManager, WebSearchCoordinator, SynthesisEngine")

    @property
    def active_connections(self) -> int:
        """Get number of currently registered agents."""
        return len(self._registered_agents)
    
    @property
    def message_queue_size(self) -> int:
        """Get number of pending messages in queue."""
        return self._message_queue.qsize()
    
    @property
    def is_active(self) -> bool:
        """Check if central post is active and accepting connections."""
        return self._is_active
    
    @property
    def total_messages_processed(self) -> int:
        """Get total number of messages processed."""
        return self._total_messages_processed
    
    def register_agent(self, agent, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Register an agent with the central post.

        Args:
            agent: Agent instance to register
            metadata: Optional additional metadata about the agent

        Returns:
            Connection ID for the registered agent, or None if capacity reached

        Raises:
            ValueError: If agent already registered (duplicate)
        """
        if self.active_connections >= self.max_agents:
            logger.warning(f"Agent cap reached ({self.max_agents} agents). Cannot register {agent.agent_id}.")
            return None  # Gracefully handle capacity limit

        if agent.agent_id in self._registered_agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")

        # Create unique connection ID
        connection_id = str(uuid.uuid4())

        # Register agent
        self._registered_agents[agent.agent_id] = connection_id
        self._connection_times[agent.agent_id] = time.time()

        # Prepare agent metadata for registry
        agent_metadata = {
            'agent_id': agent.agent_id,
            'connection_id': connection_id,
            'registration_time': self._connection_times[agent.agent_id]
        }

        # Extract metadata from agent object if available
        if hasattr(agent, 'agent_type'):
            agent_metadata['agent_type'] = agent.agent_type
        if hasattr(agent, 'spawn_time'):
            agent_metadata['spawn_time'] = agent.spawn_time
        if hasattr(agent, 'max_tokens'):
            agent_metadata['max_tokens'] = agent.max_tokens
        if hasattr(agent, 'temperature_range'):
            agent_metadata['temperature_range'] = agent.temperature_range

        # Add any provided metadata
        if metadata:
            agent_metadata.update(metadata)

        # Extract specialized agent capabilities
        if hasattr(agent, '__class__'):
            agent_class_name = agent.__class__.__name__
            if agent_class_name == 'ResearchAgent':
                agent_metadata['capabilities'] = {
                    'type': 'research',
                    'domain': getattr(agent, 'research_domain', 'general')
                }
            elif agent_class_name == 'AnalysisAgent':
                agent_metadata['capabilities'] = {
                    'type': 'analysis',
                    'analysis_type': getattr(agent, 'analysis_type', 'general')
                }
            elif agent_class_name == 'SynthesisAgent':
                agent_metadata['capabilities'] = {
                    'type': 'synthesis',
                    'output_format': getattr(agent, 'output_format', 'general')
                }
            elif agent_class_name == 'CriticAgent':
                agent_metadata['capabilities'] = {
                    'type': 'critic',
                    'review_focus': getattr(agent, 'review_focus', 'general')
                }

        # Register in agent awareness registry
        self.agent_registry.register_agent(agent.agent_id, agent_metadata)

        return connection_id

    def register_agent_id(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register an agent by ID without requiring an agent object.

        This is a lightweight registration for cases where a full Agent object
        is not available (e.g., direct mode in FelixAgent).

        Args:
            agent_id: Unique identifier for the agent
            metadata: Optional additional metadata about the agent

        Returns:
            Connection ID for the registered agent
        """
        if agent_id in self._registered_agents:
            # Already registered, return existing connection ID
            return self._registered_agents[agent_id]

        # Create unique connection ID
        connection_id = str(uuid.uuid4())

        # Register in connection tracking
        self._registered_agents[agent_id] = connection_id
        self._connection_times[agent_id] = time.time()

        # Also register in agent awareness registry if metadata provided
        if metadata:
            self.agent_registry.register_agent(agent_id=agent_id, metadata=metadata)

        logger.debug(f"Registered agent by ID: {agent_id} -> {connection_id}")
        return connection_id

    def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the central post.

        Args:
            agent_id: ID of agent to deregister

        Returns:
            True if successfully deregistered, False if not found
        """
        if agent_id not in self._registered_agents:
            return False

        # Remove agent registration
        del self._registered_agents[agent_id]
        del self._connection_times[agent_id]

        # Remove from agent awareness registry
        self.agent_registry.deregister_agent(agent_id)

        return True
    
    def is_agent_registered(self, agent_id: str) -> bool:
        """
        Check if an agent is currently registered.
        
        Args:
            agent_id: ID of agent to check
            
        Returns:
            True if agent is registered, False otherwise
        """
        return agent_id in self._registered_agents
    
    async def _ensure_async_queue(self) -> asyncio.Queue:
        """Ensure async message queue is initialized."""
        if self._async_message_queue is None:
            self._async_message_queue = asyncio.Queue(maxsize=1000)
        return self._async_message_queue
    
    def queue_message(self, message: Message) -> str:
        """
        Queue a message for processing (sync).
        
        Args:
            message: Message to queue
            
        Returns:
            Message ID for tracking
        """
        if not self._is_active:
            raise RuntimeError("Central post is not active")
        
        # Validate sender is registered
        if message.sender_id != "central_post" and message.sender_id not in self._registered_agents:
            raise ValueError(f"Message from unregistered agent: {message.sender_id}")
        
        # Queue message
        self._message_queue.put(message)
        
        return message.message_id
    
    async def queue_message_async(self, message: Message) -> str:
        """
        Queue a message for async processing.
        
        Args:
            message: Message to queue
            
        Returns:
            Message ID for tracking
        """
        if not self._is_active:
            raise RuntimeError("Central post is not active")
        
        # Validate sender is registered
        if message.sender_id != "central_post" and message.sender_id not in self._registered_agents:
            raise ValueError(f"Message from unregistered agent: {message.sender_id}")
        
        # Queue message asynchronously
        async_queue = await self._ensure_async_queue()
        await async_queue.put(message)
        
        return message.message_id
    
    def has_pending_messages(self) -> bool:
        """
        Check if there are messages waiting to be processed.

        Returns:
            True if messages are pending, False otherwise
        """
        return not self._message_queue.empty()

    def set_current_task(self, task_description: str) -> None:
        """
        Set the current task description for web search context.

        Args:
            task_description: Description of the current workflow task
        """
        self._current_task_description = task_description
        self._search_count = 0  # Reset search counter for new task

    def set_current_workflow(self, workflow_id: Optional[str]) -> None:
        """
        Set the current workflow ID for approval rule scoping.

        Args:
            workflow_id: ID of the current workflow (e.g., "workflow_001") or None to clear
        """
        self._current_workflow_id = workflow_id
        logger.info(f"Current workflow ID set: {workflow_id}")

    def clear_current_workflow(self) -> None:
        """
        Clear current workflow and clean up workflow-scoped approval rules.
        Should be called when workflow completes.
        """
        if self._current_workflow_id:
            # Clear approval rules for this workflow
            self.approval_manager.clear_workflow_rules(self._current_workflow_id)
            logger.info(f"Cleared workflow approval rules for: {self._current_workflow_id}")
            self._current_workflow_id = None

    def process_next_message(self) -> Optional[Message]:
        """
        Process the next message in the queue (FIFO order).
        
        Returns:
            Processed message, or None if queue is empty
        """
        try:
            # Get next message
            start_time = time.time() if self._metrics_enabled else None
            message = self._message_queue.get_nowait()
            
            # Process message (placeholder - actual processing depends on message type)
            self._handle_message(message)
            
            # Record metrics
            if self._metrics_enabled and start_time:
                processing_time = time.time() - start_time
                self._processing_times.append(processing_time)
            
            # Track processed message
            self._processed_messages.append(message)
            self._total_messages_processed += 1

            # Track confidence for web search monitoring
            if 'confidence' in message.content:
                self._recent_confidences.append(message.content['confidence'])

            # Check if web search is needed based on confidence
            self._check_confidence_and_search()

            return message

        except Empty:
            return None
    
    async def process_next_message_async(self) -> Optional[Message]:
        """
        Process the next message in the async queue (FIFO order).
        
        Returns:
            Processed message, or None if queue is empty
        """
        try:
            async_queue = await self._ensure_async_queue()
            
            # Try to get message without blocking
            try:
                message = async_queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
            
            # Get next message
            start_time = time.time() if self._metrics_enabled else None
            
            # Process message asynchronously
            await self._handle_message_async(message)
            
            # Record metrics
            if self._metrics_enabled and start_time:
                processing_time = time.time() - start_time
                self._processing_times.append(processing_time)
            
            # Track processed message
            self._processed_messages.append(message)
            self._total_messages_processed += 1
            
            return message
            
        except Exception as e:
            logger.error(f"Async message processing failed: {e}")
            return None

    def receive_partial_thought(
        self,
        agent_id: str,
        partial_content: str,
        accumulated: str,
        progress: float,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Receive time-batched streaming chunk from agent.

        Accumulates content for display but doesn't synthesize until complete.
        This is the HYBRID approach: real-time display, deferred synthesis.

        Args:
            agent_id: Agent sending the partial thought
            partial_content: New content since last batch
            accumulated: Full content accumulated so far
            progress: Agent's progress along helix (0.0-1.0)
            metadata: Additional metadata (agent_type, checkpoint, etc.)
        """
        return self.streaming_coordinator.receive_partial_thought(
            agent_id, partial_content, accumulated, progress, metadata
        )

    def finalize_streaming_thought(
        self,
        agent_id: str,
        final_content: str,
        confidence: float
    ) -> None:
        """
        Finalize streaming thought when agent completes.

        Now we can synthesize with complete message (hybrid approach).

        Args:
            agent_id: Agent completing the thought
            final_content: Complete final content
            confidence: Agent's confidence score
        """
        return self.streaming_coordinator.finalize_streaming_thought(
            agent_id, final_content, confidence
        )

    def register_streaming_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register callback for streaming events (for GUI updates).

        Args:
            callback: Function to call with streaming events
        """
        return self.streaming_coordinator.register_streaming_callback(callback)

    # ============================================================================
    # CENTRALPOST WEB SEARCH (Confidence-Based Information Gathering)
    # ============================================================================

    def _check_confidence_and_search(self) -> None:
        """
        Check rolling average confidence and trigger web search if low.

        Called after each message is processed to monitor team consensus.
        If confidence drops below threshold and cooldown expired, performs web search.
        """
        return self.web_search_coordinator.check_confidence_and_search()

    def update_confidence_threshold(self, new_threshold: float, reason: str = "") -> None:
        """
        Dynamically update the confidence threshold for synthesis/web search triggering.

        Args:
            new_threshold: New confidence threshold value (0.0-1.0)
            reason: Explanation for threshold change (for logging)
        """
        return self.web_search_coordinator.update_confidence_threshold(new_threshold, reason)

    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self.web_search_coordinator.get_confidence_threshold()

    def perform_web_search(self, task_description: str) -> None:
        """
        Perform web search when consensus is low and store relevant info in knowledge base.

        Args:
            task_description: The current workflow task to guide search queries
        """
        return self.web_search_coordinator.perform_web_search(task_description)

    def _handle_web_search_request(self, message: Message) -> None:
        """
        Handle explicit web search request from an agent.

        Args:
            message: Message containing WEB_SEARCH_NEEDED request
        """
        return self.web_search_coordinator.handle_web_search_request(message)

    def _handle_system_action_detection(self, message: Message) -> None:
        """
        Detect and handle SYSTEM_ACTION_NEEDED: pattern in agent response.

        Similar to web search detection, scans agent output for system action
        requests and automatically routes them through the system autonomy
        infrastructure.

        Note on Conditional Tool Instructions:
        - Agents only receive tool patterns (like SYSTEM_ACTION_NEEDED) if task requires them
        - If agent uses this pattern, it means tool instructions were provided
        - System is self-correcting: no instructions = agent doesn't know pattern
        - No additional validation needed

        Args:
            message: Message containing SYSTEM_ACTION_NEEDED request
        """
        try:
            import re

            content = message.content.get('content', '')
            agent_id = message.sender_id

            # Extract command from SYSTEM_ACTION_NEEDED: pattern
            # Capture only the first line after SYSTEM_ACTION_NEEDED: to avoid capturing
            # agent reasoning, numbered lists, or prose that follows
            # This prevents malformed commands like "mkdir /results\n\n7. Systemic Risk: ..."
            pattern = r'SYSTEM_ACTION_NEEDED:\s*([^\n]+)'
            matches = re.findall(pattern, content, re.IGNORECASE)

            if not matches:
                logger.warning(f"Agent {agent_id} used SYSTEM_ACTION_NEEDED but no command found")
                return

            # Process each command found (agents might request multiple)
            for command in matches:
                command = command.strip()

                # CRITICAL FIX: Strip surrounding quotes if agent wrapped the entire command
                # Fixes: "SYSTEM_ACTION_NEEDED: head -n 50 file" results in command"
                # This causes shell errors: /bin/sh: Syntax error: Unterminated quoted string
                if len(command) >= 2:
                    # Handle matched quotes (both opening and closing)
                    if (command[0] == '"' and command[-1] == '"') or \
                       (command[0] == "'" and command[-1] == "'"):
                        command = command[1:-1].strip()
                        logger.debug(f"Stripped surrounding quotes from command")
                    # Handle trailing quote only (opening quote lost in regex capture)
                    elif command.endswith('"') or command.endswith("'"):
                        command = command.rstrip('"').rstrip("'").strip()
                        logger.debug(f"Stripped trailing quote from command")

                # Skip nested WEB_SEARCH_NEEDED patterns (validation gap fix)
                if command.startswith('WEB_SEARCH_NEEDED:'):
                    logger.warning(f"⚠️ Agent {agent_id} nested WEB_SEARCH_NEEDED inside SYSTEM_ACTION_NEEDED")
                    logger.warning(f"   Invalid command skipped: {command}")
                    logger.info(f"   Tip: Use WEB_SEARCH_NEEDED: on its own line, not inside SYSTEM_ACTION_NEEDED:")
                    continue  # Skip this malformed command

                # Validate extracted command (debug logging)
                logger.debug(f"Extracted command: '{command}' (length: {len(command)})")
                if len(command) > 200:
                    logger.warning(f"⚠️ Extracted command is suspiciously long ({len(command)} chars), may have captured prose")
                    logger.warning(f"   First 100 chars: {command[:100]}")

                logger.info("=" * 60)
                logger.info(f"🖥️ AGENT-REQUESTED SYSTEM ACTION")
                logger.info("=" * 60)
                logger.info(f"Requesting Agent: {agent_id}")
                logger.info(f"Command: \"{command}\"")
                logger.info("")

                # Extract context from agent's message if available
                context = f"Requested by {agent_id} to complete task"
                if self._current_task_description:
                    context += f": {self._current_task_description[:100]}"

                # PRE-EXECUTION DEDUPLICATION: Check if command already executed in this workflow
                # This prevents agents from repeatedly requesting the same command
                if self._current_workflow_id:
                    command_hash = self.system_executor.compute_command_hash(command)
                    if self._current_workflow_id in self.system_command_manager._executed_commands:
                        if command_hash in self.system_command_manager._executed_commands[self._current_workflow_id]:
                            cached_result = self.system_command_manager._executed_commands[self._current_workflow_id][command_hash]
                            logger.info("⚡ Command already executed in this workflow - skipping duplicate")
                            logger.info(f"  Command: {command}")
                            logger.info(f"  Previous result: {'SUCCESS' if cached_result.success else 'FAILED'}")
                            logger.info(f"  Exit code: {cached_result.exit_code}")
                            logger.info("=" * 60)
                            continue  # Skip this duplicate command

                # Request system action through normal flow
                # This will handle trust classification, approval workflow, execution
                # Pass project_root as working directory for command execution
                action_id = self.request_system_action(
                    agent_id=agent_id,
                    command=command,
                    context=context,
                    workflow_id=self._current_workflow_id,
                    cwd=self.project_root
                )

                logger.info(f"  Action ID: {action_id}")

                # Wait for action to complete (blocks if approval needed)
                logger.info(f"⏸️  Workflow paused - waiting for command completion...")
                result = self.wait_for_approval(action_id, timeout=300.0)

                if result:
                    if result.success:
                        logger.info(f"✓ System action completed successfully")
                        logger.info(f"   Command: {command}")
                        logger.info(f"   Output: {result.stdout[:200] if result.stdout else '(no output)'}")

                        # CRITICAL FIX: Store command output as knowledge entry
                        # This allows agents to retrieve the result via context builder
                        if result.stdout:
                            try:
                                # Filter out .felixignore paths to prevent data poisoning
                                filtered_output = filter_command_output(result.stdout)
                                if filtered_output.strip():
                                    output_content = f"Command: {command}\n\nOutput:\n{filtered_output}"
                                    knowledge_id = self.store_agent_result_as_knowledge(
                                        agent_id=agent_id,
                                        content=output_content,
                                        confidence=1.0,
                                        domain="system_action"
                                    )
                                    logger.info(f"  ✓ Stored command output as knowledge entry #{knowledge_id}")
                                    logger.info(f"    Agents can now retrieve this result via context builder")
                                else:
                                    logger.debug(f"  ℹ️ Command output filtered entirely by .felixignore")
                            except Exception as store_error:
                                logger.error(f"  ⚠️ Failed to store command output as knowledge: {store_error}")
                    else:
                        logger.warning(f"⚠️ System action failed or denied")
                        logger.warning(f"   Command: {command}")
                        logger.warning(f"   Error: {result.stderr[:200] if result.stderr else '(no error message)'}")
                else:
                    logger.error(f"❌ System action timed out or failed to complete")
                    logger.error(f"   Command: {command}")

                logger.info(f"▶️  Workflow resuming...")
                logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Agent system action request failed: {e}", exc_info=True)

    def _handle_extension_request_detection(self, message: Message) -> None:
        """
        Detect and handle NEED_MORE_PROCESSING: pattern in agent response.

        Phase 3.1: Agents can request additional processing time if they reach
        a checkpoint but confidence is still low. This enables dynamic checkpoint
        injection to allow workflows to iterate until solved.

        Args:
            message: Message potentially containing extension request
        """
        try:
            import re

            content = message.content.get('content', '')
            agent_id = message.sender_id

            # Extract reason from NEED_MORE_PROCESSING: pattern
            pattern = r'NEED_MORE_PROCESSING:\s*([^\n]+)'
            matches = re.findall(pattern, content, re.IGNORECASE)

            if not matches:
                return

            reason = matches[0].strip()

            logger.info("=" * 60)
            logger.info(f"🔄 AGENT PROCESSING EXTENSION REQUEST")
            logger.info("=" * 60)
            logger.info(f"Requesting Agent: {agent_id}")
            logger.info(f"Reason: {reason}")
            logger.info("")

            # Store extension request for workflow to handle
            if not hasattr(self, '_extension_requests'):
                self._extension_requests = []

            self._extension_requests.append({
                'agent_id': agent_id,
                'reason': reason,
                'timestamp': message.timestamp
            })

            logger.info(f"  ✓ Extension request recorded ({len(self._extension_requests)} total)")
            logger.info(f"  ℹ Workflow will evaluate if additional steps should be added")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Extension request detection failed: {e}", exc_info=True)

    def get_extension_requests(self) -> List[Dict[str, Any]]:
        """
        Get all pending extension requests from agents.

        Returns:
            List of extension request dictionaries
        """
        if not hasattr(self, '_extension_requests'):
            self._extension_requests = []
        return self._extension_requests

    def clear_extension_requests(self):
        """Clear all extension requests (called after workflow processes them)."""
        self._extension_requests = []


    # ============================================================================
    # CENTRALPOST SYNTHESIS (Felix Architecture: CentralPost is Smart)
    # ============================================================================

    def synthesize_agent_outputs(self, task_description: str, max_messages: int = 20,
                                 task_complexity: str = "COMPLEX",
                                 reasoning_evals: Optional[Dict[str, Dict[str, Any]]] = None,
                                 coverage_report: Optional[Any] = None,
                                 successful_agents: Optional[List[str]] = None,
                                 failed_agents: Optional[List[str]] = None) -> Dict[str, Any]:
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

        Returns:
            Dict containing:
                - synthesis_content: Final synthesized output text
                - confidence: Synthesis confidence score (0.0-1.0)
                - meta_confidence: Coverage-adjusted confidence (Phase 7)
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
        return self.synthesis_engine.synthesize_agent_outputs(
            task_description, max_messages, task_complexity, reasoning_evals, coverage_report,
            successful_agents, failed_agents
        )

    def broadcast_synthesis_feedback(self, synthesis_result: Dict[str, Any],
                                     task_description: str,
                                     workflow_id: Optional[str] = None,
                                     task_type: Optional[str] = None,
                                     knowledge_entry_ids: Optional[List[str]] = None) -> None:
        """
        Broadcast synthesis feedback back to agents for learning and improvement.

        This implements the Feedback Integration Protocol, sending performance
        feedback to each agent about how their contributions were used in the
        final synthesis. Enables agents to learn and adapt.

        Also records knowledge usage with usefulness scores for meta-learning boost.

        Args:
            synthesis_result: The completed synthesis result from synthesize_agent_outputs()
            task_description: Original task description for context
            workflow_id: Optional workflow ID for knowledge usage tracking
            task_type: Optional classified task type for meta-learning differentiation
            knowledge_entry_ids: Optional list of knowledge entry IDs used in this workflow
        """
        if not synthesis_result or 'synthesis_content' not in synthesis_result:
            logger.warning("Cannot broadcast feedback - invalid synthesis result")
            return

        synthesis_content = synthesis_result['synthesis_content']
        synthesis_confidence = synthesis_result.get('confidence', 0.0)
        agents_synthesized = synthesis_result.get('agents_synthesized', 0)

        # Get recent agent messages (same messages used in synthesis)
        recent_messages = self._processed_messages[-synthesis_result.get('agents_synthesized', 20):]

        logger.info(f"🔄 Broadcasting synthesis feedback to {len(recent_messages)} agents")

        # Evaluate each agent's contribution
        for message in recent_messages:
            if message.message_type != MessageType.STATUS_UPDATE:
                continue

            agent_id = message.sender_id
            agent_content = message.content.get('result', '')
            agent_confidence = message.content.get('confidence', 0.0)

            # Calculate contribution evaluation metrics
            usefulness_score = self._evaluate_contribution_usefulness(
                agent_content, synthesis_content
            )

            # Calibration feedback: how did agent confidence compare to synthesis outcome?
            confidence_calibration = synthesis_confidence - agent_confidence

            # Send SYNTHESIS_FEEDBACK message
            feedback_message = Message(
                sender_id="central_post",
                message_type=MessageType.SYNTHESIS_FEEDBACK,
                content={
                    'synthesis_confidence': synthesis_confidence,
                    'synthesis_summary': synthesis_content[:500],  # First 500 chars
                    'agents_synthesized': agents_synthesized,
                    'task_description': task_description
                },
                timestamp=time.time()
            )

            # Send CONTRIBUTION_EVALUATION message
            evaluation_message = Message(
                sender_id="central_post",
                message_type=MessageType.CONTRIBUTION_EVALUATION,
                content={
                    'usefulness_score': usefulness_score,
                    'incorporated_in_synthesis': usefulness_score > 0.3,
                    'agent_confidence': agent_confidence,
                    'synthesis_confidence': synthesis_confidence,
                    'confidence_calibration': confidence_calibration,
                    'calibration_quality': 'good' if abs(confidence_calibration) < 0.2 else 'needs_adjustment'
                },
                timestamp=time.time()
            )

            # Queue feedback messages for the agent
            self._message_queue.put(feedback_message)
            self._message_queue.put(evaluation_message)

            logger.debug(f"  → Feedback sent to {agent_id}: usefulness={usefulness_score:.2f}, calibration={confidence_calibration:+.2f}")

        # Record knowledge usage with synthesis-derived usefulness scores for meta-learning
        if knowledge_entry_ids and workflow_id and self.knowledge_store:
            # Use synthesis confidence as proxy for overall usefulness
            useful_score = synthesis_confidence

            try:
                self.knowledge_store.record_knowledge_usage(
                    workflow_id=workflow_id,
                    knowledge_ids=knowledge_entry_ids,
                    task_type=task_type or 'general_task',
                    useful_score=useful_score
                )
                logger.info(f"  ✓ Recorded meta-learning usage for {len(knowledge_entry_ids)} knowledge entries "
                           f"(useful_score={useful_score:.2f}, task_type={task_type})")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to record knowledge usage for meta-learning: {e}")

    def _evaluate_contribution_usefulness(self, agent_content: str,
                                         synthesis_content: str) -> float:
        """
        Evaluate how useful an agent's contribution was to the final synthesis.

        Uses simple heuristics to estimate if agent content appears in synthesis:
        - Shared key phrases (3+ word sequences)
        - Concept overlap

        Args:
            agent_content: Agent's output content
            synthesis_content: Final synthesis content

        Returns:
            Usefulness score from 0.0 (not used) to 1.0 (heavily used)
        """
        if not agent_content or not synthesis_content:
            return 0.0

        # Convert to lowercase for comparison
        agent_lower = agent_content.lower()
        synthesis_lower = synthesis_content.lower()

        # Extract significant phrases (3+ words) from agent content
        agent_words = agent_lower.split()
        matches = 0
        total_phrases = 0

        # Check for 3-word phrase matches
        for i in range(len(agent_words) - 2):
            phrase = ' '.join(agent_words[i:i+3])
            if len(phrase) > 15:  # Only meaningful phrases
                total_phrases += 1
                if phrase in synthesis_lower:
                    matches += 1

        if total_phrases == 0:
            # Fallback: simple word overlap
            agent_significant_words = {w for w in agent_words if len(w) > 4}
            synthesis_words = set(synthesis_lower.split())
            overlap = len(agent_significant_words & synthesis_words)
            return min(1.0, overlap / max(len(agent_significant_words), 1) * 2)

        # Return phrase match ratio
        return min(1.0, matches / total_phrases)

    def get_action_results(self) -> List['CommandResult']:
        """
        Get list of completed system action results.

        Returns:
            List of CommandResult objects for all executed actions

        Example:
            >>> results = central_post.get_action_results()
            >>> all_succeeded = all(r.success for r in results)
        """
        return list(self._action_results.values())


    def _handle_message(self, message: Message) -> None:
        """
        Handle specific message types (internal processing).

        Args:
            message: Message to handle
        """
        # DEBUG: Log message handling
        logger.info(f"📨 CentralPost handling message type={message.message_type.value} from {message.sender_id}")

        # Update agent registry with message metadata
        self._update_agent_registry_from_message(message)

        # Message type-specific handling
        if message.message_type == MessageType.TASK_REQUEST:
            self._handle_task_request(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            logger.info(f"  → Routing to _handle_status_update()")
            self._handle_status_update(message)
        elif message.message_type == MessageType.TASK_COMPLETE:
            self._handle_task_completion(message)
        elif message.message_type == MessageType.ERROR_REPORT:
            self._handle_error_report(message)
        # Phase-aware message handlers
        elif message.message_type == MessageType.PHASE_ANNOUNCE:
            self._handle_phase_announce(message)
        elif message.message_type == MessageType.CONVERGENCE_SIGNAL:
            self._handle_convergence_signal(message)
        elif message.message_type == MessageType.COLLABORATION_REQUEST:
            self._handle_collaboration_request(message)
        elif message.message_type == MessageType.SYNTHESIS_READY:
            self._handle_synthesis_ready(message)
        elif message.message_type == MessageType.AGENT_QUERY:
            self._handle_agent_query(message)
        # System action handlers
        elif message.message_type == MessageType.SYSTEM_ACTION_REQUEST:
            self._handle_system_action_request(message)
        elif message.message_type == MessageType.SYSTEM_ACTION_RESULT:
            # CRITICAL FIX: Store system action results as knowledge entries
            # This ensures agents can retrieve command outputs via context builder
            command = message.content.get('command', '')
            stdout = message.content.get('stdout', '')
            success = message.content.get('success', False)
            agent_id = message.sender_id if message.sender_id else 'system'

            logger.debug(f"System action result message processed: {message.content.get('action_id')}")

            # Store successful command outputs as retrievable knowledge
            if success and stdout:
                try:
                    # Filter out .felixignore paths to prevent data poisoning
                    filtered_stdout = filter_command_output(stdout)
                    if filtered_stdout.strip():
                        output_content = f"Command: {command}\n\nOutput:\n{filtered_stdout}"
                        knowledge_id = self.store_agent_result_as_knowledge(
                            agent_id=agent_id,
                            content=output_content,
                            confidence=1.0,
                            domain="system_action"
                        )
                        logger.debug(f"  ✓ Stored system action result as knowledge entry #{knowledge_id}")
                except Exception as store_error:
                    logger.warning(f"  ⚠️ Failed to store system action result as knowledge: {store_error}")

    async def _handle_message_async(self, message: Message) -> None:
        """
        Handle specific message types asynchronously (internal processing).

        Args:
            message: Message to handle
        """
        # Update agent registry with message metadata
        self._update_agent_registry_from_message(message)

        # Message type-specific async handling
        if message.message_type == MessageType.TASK_REQUEST:
            await self._handle_task_request_async(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self._handle_status_update_async(message)
        elif message.message_type == MessageType.TASK_COMPLETE:
            await self._handle_task_completion_async(message)
        elif message.message_type == MessageType.ERROR_REPORT:
            await self._handle_error_report_async(message)
        # Phase-aware message handlers
        elif message.message_type == MessageType.PHASE_ANNOUNCE:
            await self._handle_phase_announce_async(message)
        elif message.message_type == MessageType.CONVERGENCE_SIGNAL:
            await self._handle_convergence_signal_async(message)
        elif message.message_type == MessageType.COLLABORATION_REQUEST:
            await self._handle_collaboration_request_async(message)
        elif message.message_type == MessageType.SYNTHESIS_READY:
            await self._handle_synthesis_ready_async(message)
        elif message.message_type == MessageType.AGENT_QUERY:
            await self._handle_agent_query_async(message)
        # System action handlers
        elif message.message_type == MessageType.SYSTEM_ACTION_REQUEST:
            self._handle_system_action_request(message)

    def _update_agent_registry_from_message(self, message: Message) -> None:
        """
        Extract and update agent information from messages.

        Args:
            message: Message containing agent metadata
        """
        agent_id = message.sender_id
        if agent_id == "central_post":
            return

        # Update position if present
        if 'position_info' in message.content:
            self.agent_registry.update_agent_position(agent_id, message.content['position_info'])

        # Update performance metrics
        metrics_to_update = {}
        if 'confidence' in message.content:
            metrics_to_update['confidence'] = message.content['confidence']
            metrics_to_update['messages_sent'] = self.agent_registry._performance_metrics.get(
                agent_id, {}).get('messages_sent', 0) + 1

        if 'tokens_used' in message.content:
            metrics_to_update['tokens_used'] = message.content['tokens_used']

        if 'processing_time' in message.content:
            metrics_to_update['processing_time'] = message.content['processing_time']

        if metrics_to_update:
            self.agent_registry.update_agent_performance(agent_id, metrics_to_update)

        # Track collaborations if influenced_by is present
        if 'influenced_by' in message.content:
            for influencer_id in message.content['influenced_by']:
                self.agent_registry.record_collaboration(influencer_id, agent_id)
    
    def _handle_task_request(self, message: Message) -> None:
        """Handle task request from agent."""
        # Placeholder for task assignment logic
        pass
    
    def _validate_agent_response(self, message: Message) -> Dict[str, Any]:
        """
        Validate agent followed context awareness protocol.

        Detects violations like requesting web search when data is already available
        or requesting tools that were already provided.

        Args:
            message: Message from agent

        Returns:
            Dictionary with validation results and any violations
        """
        content = message.content.get('content', '')
        validation_result = {
            'followed_protocol': False,
            'violations': [],
            'warnings': []
        }

        # Check 1: Did agent acknowledge context? (optional check, just log)
        if 'CONTEXT_USED:' in content:
            validation_result['followed_protocol'] = True
            logger.debug(f"✓ {message.sender_id} acknowledged context usage")
        else:
            # This is just a warning, not a violation
            validation_result['warnings'].append("Agent did not explicitly acknowledge context (missing 'CONTEXT_USED:')")

        # Check 2: Did agent request redundant web search?
        if 'WEB_SEARCH_NEEDED:' in content:
            # Check if web search data was already available to this agent
            # We can approximate this by checking recent knowledge entries
            try:
                from src.memory.knowledge_store import KnowledgeQuery, ConfidenceLevel
                import time as time_module

                # Check if web search results exist in recent knowledge
                recent_web_search = self.memory_facade.retrieve_knowledge_with_query(
                    KnowledgeQuery(
                        domains=["web_search"],
                        min_confidence=ConfidenceLevel.MEDIUM,
                        time_range=(time_module.time() - 3600, time_module.time()),  # Last hour
                        limit=5
                    )
                )

                if recent_web_search and len(recent_web_search) > 0:
                    validation_result['violations'].append(
                        f"Agent requested web search despite {len(recent_web_search)} recent web search result(s) being available"
                    )
                    logger.warning(f"⚠️ PROTOCOL VIOLATION: {message.sender_id} requested redundant web search")
                    logger.warning(f"   {len(recent_web_search)} web search results were available in knowledge store")
            except Exception as e:
                logger.debug(f"Could not validate web search redundancy: {e}")

        return validation_result

    def _handle_status_update(self, message: Message) -> None:
        """Handle status update from agent and detect web search + system action requests."""
        content = message.content.get('content', '')

        # NEW: Validate agent response for protocol compliance
        validation = self._validate_agent_response(message)
        if validation['violations']:
            for violation in validation['violations']:
                logger.warning(f"  ⚠️ VIOLATION: {violation}")
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.debug(f"  ⚠️ WARNING: {warning}")

        # DEBUG: Log to trace pattern detection
        logger.debug(f"_handle_status_update called for agent {message.sender_id}")
        logger.debug(f"  Content type: {type(content)}")
        logger.debug(f"  Content length: {len(content) if isinstance(content, str) else 'N/A'}")
        logger.debug(f"  Content preview: {content[:100] if isinstance(content, str) else str(content)[:100]}")

        # Check if agent is requesting a web search
        if isinstance(content, str) and 'WEB_SEARCH_NEEDED:' in content:
            logger.info(f"🔍 Detected WEB_SEARCH_NEEDED pattern from {message.sender_id}")
            self._handle_web_search_request(message)

        # Check if agent is requesting a system action
        if isinstance(content, str) and 'SYSTEM_ACTION_NEEDED:' in content:
            logger.info(f"🖥️ Detected SYSTEM_ACTION_NEEDED pattern from {message.sender_id}")
            self._handle_system_action_detection(message)

        # Phase 3.1: Check if agent is requesting processing extension
        if isinstance(content, str) and 'NEED_MORE_PROCESSING:' in content:
            logger.info(f"🔄 Detected NEED_MORE_PROCESSING pattern from {message.sender_id}")
            self._handle_extension_request_detection(message)

        if isinstance(content, str) and 'SYSTEM_ACTION_NEEDED:' not in content and 'NEED_MORE_PROCESSING:' not in content:
            logger.debug(f"  No special patterns found in content")

        # Continue with normal status tracking
        pass
    
    def _handle_task_completion(self, message: Message) -> None:
        """Handle task completion notification."""
        # Placeholder for completion processing logic
        pass
    
    def _handle_error_report(self, message: Message) -> None:
        """Handle error report from agent."""
        # Placeholder for error handling logic
        pass

    # Phase-aware message handlers

    def _handle_phase_announce(self, message: Message) -> None:
        """
        Handle phase transition announcement from agent.

        Expected content:
        - 'old_phase': Previous phase name
        - 'new_phase': New phase name
        - 'depth_ratio': Current depth on helix
        """
        agent_id = message.sender_id
        new_phase = message.content.get('new_phase')
        depth_ratio = message.content.get('depth_ratio', 0.0)

        if new_phase:
            logger.info(f"Agent {agent_id} entering {new_phase} phase at depth {depth_ratio:.2f}")

            # Update position in registry
            position_info = {
                'depth_ratio': depth_ratio,
                'phase': new_phase
            }
            self.agent_registry.update_agent_position(agent_id, position_info)

            # Broadcast phase change to other agents in same phase
            phase_peers = self.agent_registry.get_agents_in_phase(new_phase)
            if len(phase_peers) > 1:  # More than just this agent
                notification = Message(
                    sender_id="central_post",
                    message_type=MessageType.AGENT_DISCOVERY,
                    content={
                        'event': 'phase_peer_joined',
                        'agent_id': agent_id,
                        'phase': new_phase,
                        'phase_peers': list(phase_peers.keys())
                    },
                    timestamp=time.time()
                )
                self._processed_messages.append(notification)

    def _handle_convergence_signal(self, message: Message) -> None:
        """
        Handle convergence readiness signal from agent.

        Expected content:
        - 'confidence': Agent's confidence level
        - 'ready_for_synthesis': Boolean
        """
        agent_id = message.sender_id
        confidence = message.content.get('confidence', 0.0)
        ready = message.content.get('ready_for_synthesis', False)

        # Update performance metrics
        self.agent_registry.update_agent_performance(agent_id, {'confidence': confidence})

        # Check if synthesis criteria are met
        if ready:
            synthesis_status = self._check_synthesis_criteria()
            if synthesis_status['synthesis_ready']:
                # Notify all agents that synthesis can begin
                synthesis_notification = Message(
                    sender_id="central_post",
                    message_type=MessageType.SYNTHESIS_READY,
                    content={
                        'ready_agents': synthesis_status['ready_agents'],
                        'convergence_status': synthesis_status['convergence_status']
                    },
                    timestamp=time.time()
                )
                self._processed_messages.append(synthesis_notification)

    def _handle_collaboration_request(self, message: Message) -> None:
        """
        Handle collaboration request from agent seeking peers.

        Expected content:
        - 'collaboration_type': Type of collaboration needed
        - 'phase': Current phase of requesting agent
        """
        agent_id = message.sender_id
        collab_type = message.content.get('collaboration_type', 'general')
        phase = message.content.get('phase')

        # Find suitable collaborators
        if phase:
            phase_peers = self.agent_registry.get_agents_in_phase(phase)
            # Remove requesting agent from list
            potential_collaborators = {
                peer_id: peer_data
                for peer_id, peer_data in phase_peers.items()
                if peer_id != agent_id
            }
        else:
            # Find nearby agents based on position
            nearby = self.agent_registry.get_nearby_agents(agent_id)
            potential_collaborators = {
                peer_id: self.agent_registry.get_agent_info(peer_id)
                for peer_id in nearby
            }

        # Send collaboration response
        response = Message(
            sender_id="central_post",
            message_type=MessageType.AGENT_DISCOVERY,
            content={
                'collaboration_response': True,
                'potential_collaborators': list(potential_collaborators.keys()),
                'collaboration_type': collab_type
            },
            timestamp=time.time()
        )
        self._processed_messages.append(response)

    def _handle_synthesis_ready(self, message: Message) -> None:
        """
        Handle synthesis readiness signal.

        This is typically sent by the system when criteria are met.
        """
        # Log synthesis readiness
        logger.info("Synthesis criteria met - system ready for final output")

        # Update all synthesis agents
        synthesis_agents = self.agent_registry.get_agents_in_phase('synthesis')
        for agent_id in synthesis_agents:
            self.agent_registry.update_agent_performance(
                agent_id,
                {'synthesis_authorized': True}
            )

    def _handle_agent_query(self, message: Message) -> None:
        """
        Handle agent awareness query request.

        Expected content:
        - 'query_type': Type of awareness query
        - 'target_agent_id': Optional specific agent to query about
        """
        agent_id = message.sender_id
        query_type = message.content.get('query_type', 'team_composition')
        target_id = message.content.get('target_agent_id')

        # Execute query
        query_result = self.query_team_awareness(query_type, target_id)

        # Send response
        response = Message(
            sender_id="central_post",
            message_type=MessageType.AGENT_DISCOVERY,
            content={
                'query_response': True,
                'query_type': query_type,
                'result': query_result,
                'requesting_agent': agent_id
            },
            timestamp=time.time()
        )
        self._processed_messages.append(response)
    
    # Async message handlers
    async def _handle_task_request_async(self, message: Message) -> None:
        """Handle task request from agent asynchronously."""
        # Async task assignment logic
        pass
    
    async def _handle_status_update_async(self, message: Message) -> None:
        """Handle status update from agent asynchronously."""
        # Async status tracking logic
        pass
    
    async def _handle_task_completion_async(self, message: Message) -> None:
        """Handle task completion notification asynchronously."""
        # Async completion processing logic
        pass
    
    async def _handle_error_report_async(self, message: Message) -> None:
        """Handle error report from agent asynchronously."""
        # Async error handling logic
        pass

    # Async phase-aware message handlers

    async def _handle_phase_announce_async(self, message: Message) -> None:
        """Handle phase transition announcement asynchronously."""
        # Delegate to sync handler for now
        self._handle_phase_announce(message)

    async def _handle_convergence_signal_async(self, message: Message) -> None:
        """Handle convergence signal asynchronously."""
        # Delegate to sync handler for now
        self._handle_convergence_signal(message)

    async def _handle_collaboration_request_async(self, message: Message) -> None:
        """Handle collaboration request asynchronously."""
        # Delegate to sync handler for now
        self._handle_collaboration_request(message)

    async def _handle_synthesis_ready_async(self, message: Message) -> None:
        """Handle synthesis ready signal asynchronously."""
        # Delegate to sync handler for now
        self._handle_synthesis_ready(message)

    async def _handle_agent_query_async(self, message: Message) -> None:
        """Handle agent query asynchronously."""
        # Delegate to sync handler for now
        self._handle_agent_query(message)

    def get_recent_messages(self,
                           limit: int = 10,
                           since_time: Optional[float] = None,
                           message_types: Optional[List[MessageType]] = None,
                           exclude_sender: Optional[str] = None) -> List[Message]:
        """
        Retrieve recent messages from central post for agent collaboration.

        This enables agents to read accumulated messages from previous agents,
        supporting the O(N) hub-spoke communication architecture where agents
        retrieve context through the central hub rather than peer-to-peer.

        Args:
            limit: Maximum number of messages to retrieve (default: 10)
            since_time: Only return messages after this timestamp (Unix time)
            message_types: Filter by specific message types (e.g., [MessageType.TASK_COMPLETE])
            exclude_sender: Exclude messages from specific agent_id

        Returns:
            List of messages (newest first), limited by 'limit' parameter

        Example:
            >>> # Get last 5 task completion messages
            >>> messages = central_post.get_recent_messages(
            ...     limit=5,
            ...     message_types=[MessageType.TASK_COMPLETE]
            ... )
            >>> for msg in messages:
            ...     print(f"{msg.sender_id}: {msg.content.get('content')}")
        """
        filtered_messages = self._processed_messages.copy()

        # Filter by timestamp if specified
        if since_time is not None:
            filtered_messages = [msg for msg in filtered_messages if msg.timestamp >= since_time]

        # Filter by message types if specified
        if message_types is not None:
            filtered_messages = [msg for msg in filtered_messages if msg.message_type in message_types]

        # Exclude specific sender if specified
        if exclude_sender is not None:
            filtered_messages = [msg for msg in filtered_messages if msg.sender_id != exclude_sender]

        # Sort by timestamp (newest first) and limit
        filtered_messages.sort(key=lambda m: m.timestamp, reverse=True)

        return filtered_messages[:limit]

    # Agent Awareness Query Methods

    def query_team_awareness(self, query_type: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Centralized awareness queries maintaining O(N) complexity.

        Args:
            query_type: Type of awareness query to perform
            agent_id: Optional agent ID for agent-specific queries

        Returns:
            Query results as dictionary

        Query types:
            - 'team_composition': Current team makeup
            - 'phase_distribution': Agents by phase
            - 'confidence_landscape': Confidence levels by depth
            - 'convergence_readiness': Synthesis criteria status
            - 'collaboration_graph': Agent collaboration patterns
            - 'domain_coverage': Explored domains and gaps
        """
        queries = {
            'team_composition': self._get_team_composition,
            'phase_distribution': self._get_phase_distribution,
            'confidence_landscape': self._get_confidence_by_depth,
            'convergence_readiness': self._check_synthesis_criteria,
            'collaboration_graph': self._get_collaboration_patterns,
            'domain_coverage': self._get_explored_domains
        }

        if query_type not in queries:
            return {'error': f'Unknown query type: {query_type}'}

        return queries[query_type](agent_id)

    def _get_team_composition(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current team composition and statistics."""
        active_agents = self.agent_registry.get_active_agents()

        composition = {
            'total_agents': len(active_agents),
            'agents_by_type': {},
            'agents': active_agents
        }

        for agent in active_agents:
            agent_type = agent['agent_type']
            if agent_type not in composition['agents_by_type']:
                composition['agents_by_type'][agent_type] = 0
            composition['agents_by_type'][agent_type] += 1

        return composition

    def _get_phase_distribution(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get distribution of agents across phases."""
        distribution = {
            'exploration': self.agent_registry.get_agents_in_phase('exploration'),
            'analysis': self.agent_registry.get_agents_in_phase('analysis'),
            'synthesis': self.agent_registry.get_agents_in_phase('synthesis')
        }

        return {
            'phases': {
                phase: len(agents) for phase, agents in distribution.items()
            },
            'details': distribution
        }

    def _get_confidence_by_depth(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get confidence levels mapped to helical depth."""
        agents = self.agent_registry.get_active_agents()

        confidence_map = []
        for agent in agents:
            confidence_map.append({
                'agent_id': agent['agent_id'],
                'depth_ratio': agent['depth_ratio'],
                'confidence': agent['avg_confidence'],
                'phase': agent['phase']
            })

        # Sort by depth
        confidence_map.sort(key=lambda x: x['depth_ratio'])

        return {
            'confidence_map': confidence_map,
            'avg_confidence_by_phase': self._calculate_phase_confidence()
        }

    def _check_synthesis_criteria(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if natural selection criteria for synthesis are met."""
        convergence_status = self.agent_registry.get_convergence_status()

        # Check for synthesis-ready agents
        synthesis_agents = self.agent_registry.get_agents_in_phase('synthesis')
        ready_agents = []

        for s_agent_id, agent_data in synthesis_agents.items():
            agent_info = self.agent_registry.get_agent_info(s_agent_id)
            if agent_info:
                avg_conf = agent_info['performance'].get('avg_confidence', 0.0)
                depth_ratio = agent_info['position'].get('depth_ratio', 0.0)

                if depth_ratio >= 0.7 and avg_conf >= 0.8:
                    ready_agents.append({
                        'agent_id': s_agent_id,
                        'confidence': avg_conf,
                        'depth_ratio': depth_ratio
                    })

        return {
            'synthesis_ready': len(ready_agents) > 0,
            'ready_agents': ready_agents,
            'convergence_status': convergence_status,
            'blocking_factors': self._identify_synthesis_blockers()
        }

    def _get_collaboration_patterns(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent collaboration patterns."""
        if agent_id:
            # Get specific agent's collaborations
            agent_info = self.agent_registry.get_agent_info(agent_id)
            if not agent_info:
                return {'error': f'Agent {agent_id} not found'}

            return {
                'agent_id': agent_id,
                'influenced': agent_info['collaborations'],
                'nearby_agents': self.agent_registry.get_nearby_agents(agent_id)
            }
        else:
            # Get overall collaboration graph
            return {
                'collaboration_density': self.agent_registry._calculate_collaboration_density(),
                'collaboration_graph': dict(self.agent_registry._collaboration_graph)
            }

    def _get_explored_domains(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get domains that have been explored and identify gaps."""
        explored_domains = set()
        domain_agents = {}

        for a_id, metadata in self.agent_registry._agent_metadata.items():
            capabilities = self.agent_registry._capability_matrix.get(a_id, {})
            domain = capabilities.get('domain')
            if domain:
                explored_domains.add(domain)
                if domain not in domain_agents:
                    domain_agents[domain] = []
                domain_agents[domain].append(a_id)

        # Identify potential gaps
        standard_domains = {'general', 'technical', 'creative', 'analytical', 'critical'}
        missing_domains = standard_domains - explored_domains

        return {
            'explored': list(explored_domains),
            'missing': list(missing_domains),
            'domain_agents': domain_agents,
            'coverage_ratio': len(explored_domains) / len(standard_domains) if standard_domains else 0
        }

    def _calculate_phase_confidence(self) -> Dict[str, float]:
        """Calculate average confidence by phase."""
        phase_confidences = {'exploration': [], 'analysis': [], 'synthesis': []}

        for phase in phase_confidences:
            agents = self.agent_registry.get_agents_in_phase(phase)
            for agent_id in agents:
                agent_info = self.agent_registry.get_agent_info(agent_id)
                if agent_info:
                    conf = agent_info['performance'].get('avg_confidence', 0.0)
                    phase_confidences[phase].append(conf)

        return {
            phase: sum(confs) / len(confs) if confs else 0.0
            for phase, confs in phase_confidences.items()
        }

    def _identify_synthesis_blockers(self) -> List[str]:
        """Identify factors blocking synthesis readiness."""
        blockers = []

        # Check confidence trend
        convergence = self.agent_registry.get_convergence_status()
        if convergence['confidence_trend'] == 'DECLINING':
            blockers.append('Confidence trend is declining')

        # Check synthesis agent presence
        if convergence['phase_distribution']['synthesis'] == 0:
            blockers.append('No synthesis agents active')

        # Check collaboration density
        if convergence['collaboration_density'] < 0.2:
            blockers.append('Low collaboration between agents')

        # Check exploration coverage
        domains = self._get_explored_domains()
        if domains['coverage_ratio'] < 0.5:
            blockers.append(f"Limited domain coverage ({domains['coverage_ratio']:.1%})")

        return blockers

    def get_agent_awareness_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive awareness information for a specific agent.

        Args:
            agent_id: Agent to query

        Returns:
            Dictionary with agent's awareness context
        """
        agent_info = self.agent_registry.get_agent_info(agent_id)
        if not agent_info:
            return None

        # Get phase-appropriate context
        phase = agent_info['phase']
        nearby_agents = self.agent_registry.get_nearby_agents(agent_id)

        awareness_context = {
            'self': agent_info,
            'nearby_agents': nearby_agents,
            'phase_peers': list(self.agent_registry.get_agents_in_phase(phase).keys()),
            'convergence_status': self.agent_registry.get_convergence_status()
        }

        # Add phase-specific awareness
        if phase == 'exploration':
            awareness_context['unexplored_domains'] = self._get_explored_domains()['missing']
        elif phase == 'analysis':
            convergence = self.agent_registry.get_convergence_status()
            awareness_context['confidence_trend'] = convergence['confidence_trend']
        elif phase == 'synthesis':
            awareness_context['synthesis_ready'] = self._check_synthesis_criteria()['synthesis_ready']

        return awareness_context

    # Performance metrics methods (for efficiency benchmarking)
    
    def get_current_time(self) -> float:
        """Get current timestamp for performance measurements."""
        return self.performance_monitor.get_current_time()

    def get_message_throughput(self) -> float:
        """
        Calculate message processing throughput.

        Returns:
            Messages processed per second
        """
        return self.performance_monitor.get_message_throughput()

    def measure_communication_overhead(self, num_messages: int, processing_time: float) -> float:
        """
        Measure communication overhead vs processing time.

        Args:
            num_messages: Number of messages in the measurement
            processing_time: Actual processing time for comparison

        Returns:
            Communication overhead time
        """
        return self.performance_monitor.measure_communication_overhead(num_messages, processing_time)

    def record_overhead_ratio(self, overhead_ratio: float) -> None:
        """
        Record overhead ratio for performance benchmarking.

        Args:
            overhead_ratio: Communication overhead / processing time ratio
        """
        return self.performance_monitor.record_overhead_ratio(overhead_ratio)

    def get_average_overhead_ratio(self) -> float:
        """
        Get average overhead ratio across all measurements.

        Returns:
            Average overhead ratio
        """
        return self.performance_monitor.get_average_overhead_ratio()

    def record_scaling_metric(self, agent_count: int, processing_time: float) -> None:
        """
        Record scaling performance metric.

        Args:
            agent_count: Number of agents in the test
            processing_time: Time to process messages from all agents
        """
        return self.performance_monitor.record_scaling_metric(agent_count, processing_time)

    def get_scaling_metrics(self) -> Dict[int, float]:
        """
        Get scaling performance metrics.

        Returns:
            Dictionary mapping agent count to processing time
        """
        return self.performance_monitor.get_scaling_metrics()
    
    async def start_async_processing(self, max_concurrent_processors: int = 3) -> None:
        """Start async message processors."""
        for i in range(max_concurrent_processors):
            processor = asyncio.create_task(self._async_message_processor(f"processor_{i}"))
            self._async_processors.append(processor)
    
    async def _async_message_processor(self, processor_id: str) -> None:
        """Individual async message processor."""
        while self._is_active:
            try:
                message = await self.process_next_message_async()
                if message is None:
                    # No messages to process, wait briefly
                    await asyncio.sleep(0.01)
                    continue
                    
                logger.debug(f"Processor {processor_id} handled message {message.message_id}")
                
            except Exception as e:
                logger.error(f"Async processor {processor_id} error: {e}")
                await asyncio.sleep(0.1)  # Brief recovery delay
    
    def shutdown(self) -> None:
        """Shutdown the central post and disconnect all agents."""
        self._is_active = False
        
        # Clear all connections
        self._registered_agents.clear()
        self._connection_times.clear()
        
        # Clear message queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except Empty:
                break
    
    async def shutdown_async(self) -> None:
        """Shutdown async components."""
        self._is_active = False
        
        # Cancel async processors
        for processor in self._async_processors:
            processor.cancel()
        
        # Wait for processors to finish
        if self._async_processors:
            await asyncio.gather(*self._async_processors, return_exceptions=True)
        
        self._async_processors.clear()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for analysis.

        Returns:
            Dictionary containing all performance metrics
        """
        return self.performance_monitor.get_performance_summary(
            active_connections=self.active_connections,
            async_processors=len(self._async_processors),
            async_queue_size=self._async_message_queue.qsize() if self._async_message_queue else 0
        )
    
    def accept_high_confidence_result(self, message: Message, min_confidence: float = 0.8) -> bool:
        """
        Accept agent results that meet minimum confidence threshold.
        
        This implements the natural selection aspect of the helix model -
        only high-quality results from synthesis agents deep in the helix
        are accepted as final output from the central coordination system.
        
        Args:
            message: Message containing agent result
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            
        Returns:
            True if result was accepted, False if rejected
        """
        if message.message_type != MessageType.STATUS_UPDATE:
            return False
        
        content = message.content
        confidence = content.get("confidence", 0.0)
        depth_ratio = content.get("position_info", {}).get("depth_ratio", 0.0)
        agent_type = content.get("agent_type", "")
        
        # Only synthesis agents can produce final output
        if agent_type != "synthesis":
            return False
        
        # Synthesis agents should be deep in the helix (>0.7) with high confidence
        if depth_ratio >= 0.7 and confidence >= min_confidence:
            # Accept the result - add to processed messages
            self._processed_messages.append(message)
            self._total_messages_processed += 1
            return True
        else:
            # Reject the result
            return False

    # Memory Integration Methods (Priority 5: Memory and Context Persistence)
    
    def store_agent_result_as_knowledge(self, agent_id: str, content: str,
                                      confidence: float, domain: str = "general",
                                      tags: Optional[List[str]] = None) -> bool:
        """
        Store agent result as knowledge in the persistent knowledge base.

        Args:
            agent_id: ID of the agent producing the result
            content: Content of the result to store
            confidence: Confidence level of the result (0.0 to 1.0)
            domain: Domain/category for the knowledge
            tags: Optional tags for the knowledge entry

        Returns:
            True if knowledge was stored successfully, False otherwise
        """
        return self.memory_facade.store_agent_result_as_knowledge(
            agent_id, content, confidence, domain, tags
        )
    
    def retrieve_relevant_knowledge(self, domain: Optional[str] = None,
                                  knowledge_type: Optional[KnowledgeType] = None,
                                  keywords: Optional[List[str]] = None,
                                  min_confidence: Optional[ConfidenceLevel] = None,
                                  limit: int = 10) -> List[KnowledgeEntry]:
        """
        Retrieve relevant knowledge from the knowledge base.

        Args:
            domain: Filter by domain
            knowledge_type: Filter by knowledge type
            keywords: Keywords to search for
            min_confidence: Minimum confidence level
            limit: Maximum number of entries to return

        Returns:
            List of relevant knowledge entries
        """
        return self.memory_facade.retrieve_relevant_knowledge(
            domain, knowledge_type, keywords, min_confidence, limit
        )
    
    def get_task_strategy_recommendations(self, task_description: str,
                                        task_type: str = "general",
                                        complexity: str = "MODERATE") -> Dict[str, Any]:
        """
        Get strategy recommendations based on task memory.

        Args:
            task_description: Description of the task
            task_type: Type of task (e.g., "research", "analysis", "synthesis")
            complexity: Task complexity level ("SIMPLE", "MODERATE", "COMPLEX", "VERY_COMPLEX")

        Returns:
            Dictionary containing strategy recommendations
        """
        return self.memory_facade.get_task_strategy_recommendations(
            task_description, task_type, complexity
        )
    
    def compress_large_context(self, context: str,
                             strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE_SUMMARY,
                             target_size: Optional[int] = None):
        """
        Compress large context using the context compression system.

        Args:
            context: Content to compress
            strategy: Compression strategy to use
            target_size: Optional target size for compression

        Returns:
            CompressedContext object or None if compression failed
        """
        return self.memory_facade.compress_large_context(context, strategy, target_size)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory system status and contents.

        Returns:
            Dictionary with memory system summary
        """
        return self.memory_facade.get_memory_summary()
    # SYSTEM AUTONOMY METHODS (System Action Execution)
    # ===========================================================================

    def request_system_action(self, agent_id: str, command: str,
                             context: str = "", workflow_id: Optional[str] = None,
                             cwd: Optional["Path"] = None) -> str:
        """
        Agent requests system action (command execution).

        Args:
            agent_id: ID of requesting agent
            command: Command to execute
            context: Context/reason for command
            workflow_id: Associated workflow ID
            cwd: Working directory for command execution (defaults to project root)

        Returns:
            action_id for tracking the request
        """
        from pathlib import Path
        # Use provided cwd or fall back to project root
        working_dir = cwd if cwd is not None else self.project_root

        return self.system_command_manager.request_system_action(
            agent_id, command, context, workflow_id, cwd=working_dir
        )

    def get_action_result(self, action_id: str) -> Optional[CommandResult]:
        """
        Get result of a system action.

        Args:
            action_id: Action ID to query

        Returns:
            CommandResult if available, None otherwise
        """
        return self.system_command_manager.get_action_result(action_id)

    def wait_for_approval(self, action_id: str, timeout: float = 300.0) -> Optional[CommandResult]:
        """
        Wait for approval to be processed and return result.

        This method blocks until the approval is processed (approved or denied)
        or the timeout is reached. Used by workflows to pause execution while
        waiting for user approval.

        Args:
            action_id: Action ID to wait for
            timeout: Maximum seconds to wait (default 300 = 5 minutes)

        Returns:
            CommandResult if approval processed, None if timeout or not found
        """
        return self.system_command_manager.wait_for_approval(action_id, timeout)

    def approve_system_action(self, approval_id: str, decision: 'ApprovalDecision',
                             decided_by: str = "user") -> bool:
        """
        Approve a pending system action with specific decision type.

        This method integrates with ApprovalManager to handle approval decisions
        including "always approve" rules that apply for the current workflow session.

        Args:
            approval_id: Approval request ID
            decision: Type of approval decision (ApprovalDecision enum)
            decided_by: Who made the decision (default: "user")

        Returns:
            True if approved and executed successfully, False otherwise
        """
        return self.system_command_manager.approve_system_action(
            approval_id, decision, decided_by
        )

    def approve_action(self, approval_id: str, approver: str = "user") -> bool:
        """
        Approve a pending system action.

        Args:
            approval_id: Approval request ID
            approver: Who approved (default "user")

        Returns:
            True if approved and executed successfully
        """
        return self.system_command_manager.approve_action(approval_id, approver)

    def get_pending_actions(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of pending action approvals.

        Args:
            workflow_id: Optional workflow ID to filter approvals

        Returns:
            List of pending approval dictionaries
        """
        return self.system_command_manager.get_pending_actions(workflow_id)

    def _handle_system_action_request(self, message: Message) -> None:
        """
        Handle system action request from agent.

        Args:
            message: SYSTEM_ACTION_REQUEST message
        """
        self.system_command_manager.handle_system_action_request(message)

    def _broadcast_action_result(self, action_id: str, agent_id: str,
                                 command: str, result: CommandResult) -> None:
        """
        Broadcast action result back to requesting agent.

        Args:
            action_id: Action ID
            agent_id: Requesting agent ID
            command: Command that was executed
            result: Command execution result
        """
        result_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_RESULT,
            content={
                'action_id': action_id,
                'target_agent': agent_id,
                'command': command,
                'success': result.success,
                'exit_code': result.exit_code,
                'stdout': result.stdout[:500],  # Preview
                'stderr': result.stderr[:500],
                'duration': result.duration,
                'error_category': result.error_category.value if result.error_category else None
            },
            timestamp=time.time()
        )

        self.queue_message(result_message)

    def _broadcast_approval_needed(self, action_id: str, approval_id: str,
                                   agent_id: str, command: str, context: str) -> None:
        """
        Broadcast that a command needs approval.

        Args:
            action_id: Action ID
            approval_id: Approval request ID
            agent_id: Requesting agent ID
            command: Command awaiting approval
            context: Context for command
        """
        approval_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_APPROVAL_NEEDED,
            content={
                'action_id': action_id,
                'approval_id': approval_id,
                'agent_id': agent_id,
                'command': command,
                'context': context
            },
            timestamp=time.time()
        )

        self.queue_message(approval_message)

    def _broadcast_action_denial(self, action_id: str, agent_id: str,
                                command: str, reason: str) -> None:
        """
        Broadcast that a command was denied.

        Args:
            action_id: Action ID
            agent_id: Requesting agent ID
            command: Command that was denied
            reason: Denial reason
        """
        denial_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_DENIED,
            content={
                'action_id': action_id,
                'target_agent': agent_id,
                'command': command,
                'reason': reason
            },
            timestamp=time.time()
        )

        self.queue_message(denial_message)

    def _broadcast_command_start(self, action_id: str, execution_id: int,
                                 command: str, agent_id: str, context: str = "") -> None:
        """
        Broadcast that command execution has started.

        Used by Terminal tab to display active commands in real-time.

        Args:
            action_id: Action ID
            execution_id: Database execution ID from CommandHistory
            command: Command being executed
            agent_id: Requesting agent ID
            context: Command context/reason
        """
        # Initialize live output buffer immediately so Terminal can poll it
        # Even if command fails before producing output, buffer will exist
        self._live_command_outputs[execution_id] = []

        start_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_START,
            content={
                'action_id': action_id,
                'execution_id': execution_id,
                'command': command,
                'agent_id': agent_id,
                'context': context,
                'status': 'running'
            },
            timestamp=time.time()
        )

        self.queue_message(start_message)
        logger.debug(f"📡 Broadcast: Command started - {action_id}")

    def _broadcast_command_output(self, action_id: str, execution_id: int,
                                  output_line: str, stream_type: str) -> None:
        """
        Broadcast real-time command output line.

        Used by Terminal tab to stream stdout/stderr in real-time.

        Args:
            action_id: Action ID
            execution_id: Database execution ID
            output_line: Single line of output
            stream_type: 'stdout' or 'stderr'
        """
        # Store in live output buffer for Terminal tab polling
        if execution_id not in self._live_command_outputs:
            self._live_command_outputs[execution_id] = []
        self._live_command_outputs[execution_id].append((output_line, stream_type))

        output_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_OUTPUT,
            content={
                'action_id': action_id,
                'execution_id': execution_id,
                'output_line': output_line,
                'stream_type': stream_type
            },
            timestamp=time.time()
        )

        self.queue_message(output_message)
        # Don't log every line (too verbose)

    def _broadcast_command_complete(self, action_id: str, execution_id: int,
                                    result: CommandResult) -> None:
        """
        Broadcast that command execution has completed.

        Used by Terminal tab to update command status from 'running' to 'completed'/'failed'.

        Args:
            action_id: Action ID
            execution_id: Database execution ID
            result: Final CommandResult
        """
        # Clear live output buffer for this command (keep for 5 seconds for Terminal tab to retrieve)
        # Terminal tab will clear it from active_outputs after displaying
        if execution_id in self._live_command_outputs:
            # Schedule cleanup after delay to allow Terminal tab final poll
            threading.Timer(5.0, lambda: self._live_command_outputs.pop(execution_id, None)).start()

        complete_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_COMPLETE,
            content={
                'action_id': action_id,
                'execution_id': execution_id,
                'success': result.success,
                'exit_code': result.exit_code,
                'duration': result.duration,
                'status': 'completed' if result.success else 'failed',
                'error_category': result.error_category.value if result.error_category else None
            },
            timestamp=time.time()
        )

        self.queue_message(complete_message)
        logger.debug(f"📡 Broadcast: Command completed - {action_id} (success={result.success})")

    def get_live_command_output(self, execution_id: int) -> List[tuple]:
        """
        Get accumulated live output for a command execution.

        Used by Terminal tab to poll for real-time command output during execution.

        Args:
            execution_id: Database execution ID

        Returns:
            List of (output_line, stream_type) tuples, or empty list if none available
        """
        return self.system_command_manager.get_live_command_output(execution_id)


class AgentFactory:
    """
    Factory for creating agents dynamically based on task needs.
    
    The AgentFactory allows the central post to spawn new agents
    as needed during the helix processing, enabling emergent behavior
    and adaptive team composition.
    """
    
    def __init__(self, helix: "HelixGeometry", llm_client: "LMStudioClient",
                 token_budget_manager: Optional["TokenBudgetManager"] = None,
                 random_seed: Optional[int] = None, enable_dynamic_spawning: bool = True,
                 max_agents: int = 25, token_budget_limit: int = 45000,
                 web_search_client: Optional["WebSearchClient"] = None,
                 max_web_queries: int = 3,
                 agent_registry: Optional["AgentPluginRegistry"] = None,
                 plugin_directories: Optional[List[str]] = None,
                 prompt_manager: Optional["PromptManager"] = None,
                 prompt_optimizer: Optional["PromptOptimizer"] = None):
        """
        Initialize the agent factory.

        Args:
            helix: Helix geometry for new agents
            llm_client: LM Studio client for new agents
            token_budget_manager: Optional token budget manager
            random_seed: Seed for random spawn time generation
            enable_dynamic_spawning: Enable intelligent agent spawning
            max_agents: Maximum number of agents for dynamic spawning
            token_budget_limit: Token budget limit for dynamic spawning
            web_search_client: Optional web search client for Research agents
            max_web_queries: Maximum web queries per research agent (default: 3)
            agent_registry: Optional AgentPluginRegistry (creates default if None)
            plugin_directories: Optional list of external plugin directories to load
            prompt_manager: Optional prompt manager for custom prompt templates
            prompt_optimizer: Optional prompt optimizer for learning and optimization
        """
        self.helix = helix
        self.llm_client = llm_client
        self.token_budget_manager = token_budget_manager
        self.random_seed = random_seed
        self._agent_counter = 0
        self.enable_dynamic_spawning = enable_dynamic_spawning
        self.web_search_client = web_search_client
        self.max_web_queries = max_web_queries
        self.prompt_manager = prompt_manager
        self.prompt_optimizer = prompt_optimizer

        # Initialize agent plugin registry
        if agent_registry is not None:
            self.agent_registry = agent_registry
        else:
            # Create and initialize default registry
            from src.agents.agent_plugin_registry import get_global_registry
            self.agent_registry = get_global_registry()

        # Load external plugin directories if provided
        if plugin_directories:
            for directory in plugin_directories:
                try:
                    count = self.agent_registry.add_plugin_directory(directory)
                    logger.info(f"Loaded {count} plugins from {directory}")
                except Exception as e:
                    logger.error(f"Failed to load plugins from {directory}: {e}")

        # Initialize dynamic spawning system if enabled
        if enable_dynamic_spawning:
            # Import here to avoid circular imports
            from src.agents.dynamic_spawning import DynamicSpawning
            self.dynamic_spawner = DynamicSpawning(
                agent_factory=self,
                confidence_threshold=0.8,
                max_agents=max_agents,
                token_budget_limit=token_budget_limit
            )
        else:
            self.dynamic_spawner = None

        if random_seed is not None:
            random.seed(random_seed)
    
    def create_research_agent(self, domain: str = "general",
                            spawn_time_range: Tuple[float, float] = (0.0, 0.3)) -> "LLMAgent":
        """Create a research agent with random spawn time in specified range."""
        from src.agents.specialized_agents import ResearchAgent

        spawn_time = random.uniform(*spawn_time_range)
        agent_id = f"dynamic_research_{self._agent_counter:03d}"
        self._agent_counter += 1

        return ResearchAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=self.helix,
            llm_client=self.llm_client,
            research_domain=domain,
            token_budget_manager=self.token_budget_manager,
            max_tokens=16000,
            web_search_client=self.web_search_client,
            max_web_queries=self.max_web_queries,
            prompt_manager=self.prompt_manager,
            prompt_optimizer=self.prompt_optimizer
        )
    
    def create_analysis_agent(self, analysis_type: str = "general",
                            spawn_time_range: Tuple[float, float] = (0.2, 0.7)) -> "LLMAgent":
        """Create an analysis agent with random spawn time in specified range."""
        from src.agents.specialized_agents import AnalysisAgent
        
        spawn_time = random.uniform(*spawn_time_range)
        agent_id = f"dynamic_analysis_{self._agent_counter:03d}"
        self._agent_counter += 1
        
        return AnalysisAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=self.helix,
            llm_client=self.llm_client,
            analysis_type=analysis_type,
            token_budget_manager=self.token_budget_manager,
            max_tokens=16000,
            prompt_manager=self.prompt_manager,
            prompt_optimizer=self.prompt_optimizer
        )

    def create_critic_agent(self, review_focus: str = "general",
                          spawn_time_range: Tuple[float, float] = (0.5, 0.8)) -> "LLMAgent":
        """Create a critic agent with random spawn time in specified range."""
        from src.agents.specialized_agents import CriticAgent

        spawn_time = random.uniform(*spawn_time_range)
        agent_id = f"dynamic_critic_{self._agent_counter:03d}"
        self._agent_counter += 1

        return CriticAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=self.helix,
            llm_client=self.llm_client,
            review_focus=review_focus,
            token_budget_manager=self.token_budget_manager,
            max_tokens=16000,
            prompt_manager=self.prompt_manager,
            prompt_optimizer=self.prompt_optimizer
        )

    def create_agent_by_type(self,
                            agent_type: str,
                            spawn_time_range: Optional[Tuple[float, float]] = None,
                            complexity: str = "medium",
                            **kwargs) -> "LLMAgent":
        """
        Create an agent of any registered type using the plugin registry.

        This method enables spawning both built-in and custom agent types.
        It automatically handles spawn time generation based on agent metadata
        and task complexity.

        Args:
            agent_type: Type of agent to create (e.g., "research", "analysis", "code_review")
            spawn_time_range: Optional spawn time range (uses plugin default if None)
            complexity: Task complexity for spawn range lookup ("simple", "medium", "complex")
            **kwargs: Additional parameters passed to the agent plugin

        Returns:
            Instance of the requested agent type

        Raises:
            AgentPluginError: If agent_type is not registered

        Example:
            ```python
            # Create builtin research agent
            research = factory.create_agent_by_type("research", domain="technical")

            # Create custom code review agent
            reviewer = factory.create_agent_by_type(
                "code_review",
                complexity="complex",
                review_style="security-focused"
            )
            ```
        """
        # Get spawn range
        if spawn_time_range is None:
            spawn_time_range = self.agent_registry.get_spawn_range(agent_type, complexity)

        # Generate random spawn time in range
        spawn_time = random.uniform(*spawn_time_range)

        # Generate unique agent ID
        agent_id = f"dynamic_{agent_type}_{self._agent_counter:03d}"
        self._agent_counter += 1

        # Create agent using registry
        # CRITICAL: Pass prompt_manager and prompt_optimizer to ensure agents get proper prompts
        return self.agent_registry.create_agent(
            agent_type=agent_type,
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=self.helix,
            llm_client=self.llm_client,
            token_budget_manager=self.token_budget_manager,
            prompt_manager=self.prompt_manager,
            prompt_optimizer=self.prompt_optimizer,
            **kwargs
        )

    def list_available_agent_types(self) -> List[str]:
        """
        Get list of all available agent types (builtin + custom plugins).

        Returns:
            List of agent type identifiers

        Example:
            ```python
            types = factory.list_available_agent_types()
            # ['research', 'analysis', 'critic', 'code_review', ...]
            ```
        """
        return self.agent_registry.list_agent_types()

    def get_agent_metadata(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific agent type.

        Args:
            agent_type: Type identifier

        Returns:
            Dictionary with agent metadata (capabilities, spawn_range, etc.)
        """
        from dataclasses import asdict
        metadata = self.agent_registry.get_metadata(agent_type)
        return asdict(metadata) if metadata else None

    def get_suitable_agents_for_task(self,
                                     task_description: str,
                                     task_complexity: str = "medium") -> List[str]:
        """
        Get list of agent types suitable for a given task.

        This method filters agents based on task characteristics and returns
        agent types sorted by priority.

        Args:
            task_description: Human-readable task description
            task_complexity: Task complexity ("simple", "medium", "complex")

        Returns:
            List of agent types sorted by suitability (highest priority first)

        Example:
            ```python
            agents = factory.get_suitable_agents_for_task(
                "Review Python code for security vulnerabilities",
                "complex"
            )
            # Returns: ["code_review", "security_auditor", "critic"]
            ```
        """
        return self.agent_registry.get_agents_for_task(
            task_description=task_description,
            task_complexity=task_complexity
        )
    
    def assess_team_needs(self, processed_messages: List[Message],
                         current_time: float, current_agents: Optional[List["LLMAgent"]] = None,
                         task_description: Optional[str] = None) -> List["LLMAgent"]:
        """
        Assess current team composition and suggest new agents if needed.

        Enhanced with DynamicSpawning system that provides:
        - Confidence monitoring with trend analysis
        - Content analysis for contradictions and gaps
        - Team size optimization based on task complexity
        - Resource-aware spawning decisions
        - NEW: Plugin-aware spawning based on task description

        Falls back to basic heuristics if dynamic spawning is disabled.

        Args:
            processed_messages: Messages processed so far
            current_time: Current simulation time
            current_agents: List of currently active agents
            task_description: Optional task description for plugin-aware spawning

        Returns:
            List of recommended new agents to spawn
        """
        # Use dynamic spawning if enabled and available
        if self.enable_dynamic_spawning and self.dynamic_spawner:
            # NEW: Set task description for plugin-aware spawning
            if task_description:
                self.dynamic_spawner.set_task_description(task_description)

            return self.dynamic_spawner.analyze_and_spawn(
                processed_messages, current_agents or [], current_time
            )

        # Fallback to basic heuristics for backward compatibility
        return self._assess_team_needs_basic(processed_messages, current_time)
    
    def _assess_team_needs_basic(self, processed_messages: List[Message], 
                                current_time: float) -> List["LLMAgent"]:
        """
        Basic team assessment for backward compatibility.
        
        This implements simple heuristics when dynamic spawning is disabled.
        """
        recommended_agents = []
        
        if not processed_messages:
            return recommended_agents
        
        # Analyze recent messages for patterns
        recent_messages = [msg for msg in processed_messages 
                          if msg.timestamp > current_time - 0.2]  # Last 0.2 time units
        
        if not recent_messages:
            return recommended_agents
        
        # Check for consistent low confidence
        low_confidence_count = sum(1 for msg in recent_messages
                                 if msg.content.get("confidence", 1.0) < 0.6)
        
        if low_confidence_count >= 2:
            # Spawn critic agent to improve quality
            critic = self.create_critic_agent(
                review_focus="quality_improvement",
                spawn_time_range=(current_time + 0.1, current_time + 0.3)
            )
            recommended_agents.append(critic)
        
        # Check for gaps in research domains
        research_domains = set()
        for msg in recent_messages:
            if "research_domain" in msg.content:
                research_domains.add(msg.content["research_domain"])
        
        # If only general research, add technical research
        if len(research_domains) == 1 and "general" in research_domains:
            technical_research = self.create_research_agent(
                domain="technical",
                spawn_time_range=(current_time + 0.05, current_time + 0.2)
            )
            recommended_agents.append(technical_research)
        
        # Note: Synthesis is now handled directly by CentralPost, not by a specialized agent
        
        return recommended_agents
    
    def get_spawning_summary(self) -> Dict[str, Any]:
        """
        Get summary of dynamic spawning activity.
        
        Returns:
            Dictionary with spawning statistics and activity
        """
        if self.enable_dynamic_spawning and self.dynamic_spawner:
            return self.dynamic_spawner.get_spawning_summary()
        else:
            return {
                "dynamic_spawning_enabled": False,
                "total_spawns": 0,
                "spawns_by_type": {},
                "average_priority": 0.0,
                "spawning_reasons": []
            }

    # ===========================================================================
