"""
Central coordination system for the Felix Framework.

The central post manages communication and coordination between agents,
implementing the hub of the spoke-based communication model from thefelix.md.

Mathematical Foundation:
- Spoke communication: O(N) message complexity vs O(N²) mesh topology
- Maximum communication distance: R_top (helix outer radius)
- Performance metrics for Hypothesis H2 validation and statistical analysis

Key Features:
- Agent registration and connection management
- FIFO message queuing with guaranteed ordering
- Performance metrics collection (throughput, latency, overhead ratios)
- Scalability up to 133 agents (matching OpenSCAD model parameters)

Implementation supports rigorous testing of Hypothesis H2 communication efficiency claims.
"""

import time
import uuid
import random
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from collections import deque
from queue import Queue, Empty
import asyncio

# Memory system imports
from src.memory.knowledge_store import KnowledgeStore, KnowledgeEntry, KnowledgeType, ConfidenceLevel
from src.memory.task_memory import TaskMemory, TaskPattern, TaskOutcome
from src.memory.context_compression import ContextCompressor, CompressionStrategy

# Dynamic spawning imports - moved to avoid circular imports

if TYPE_CHECKING:
    from agents.llm_agent import LLMAgent
    from core.helix_geometry import HelixGeometry
    from llm.lm_studio_client import LMStudioClient
    from llm.token_budget import TokenBudgetManager

# Set up logging
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the communication system."""
    TASK_REQUEST = "task_request"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    TASK_COMPLETE = "task_complete"
    ERROR_REPORT = "error_report"
    # Phase-aware message types for agent awareness
    PHASE_ANNOUNCE = "phase_announce"  # Agent announces entering new phase
    CONVERGENCE_SIGNAL = "convergence_signal"  # Agent signals convergence readiness
    COLLABORATION_REQUEST = "collaboration_request"  # Agent seeks peers in same phase
    SYNTHESIS_READY = "synthesis_ready"  # Signal that synthesis criteria met
    AGENT_QUERY = "agent_query"  # Agent queries for awareness information
    AGENT_DISCOVERY = "agent_discovery"  # Response with agent information


@dataclass
class Message:
    """Message structure for communication between agents and central post."""
    sender_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


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
                 llm_client: Optional["LMStudioClient"] = None):
        """
        Initialize central post with configuration parameters.

        Args:
            max_agents: Maximum number of concurrent agent connections
            enable_metrics: Whether to collect performance metrics
            enable_memory: Whether to enable persistent memory systems
            memory_db_path: Path to the memory database file
            llm_client: Optional LLM client for CentralPost synthesis capability
        """
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        self.enable_memory = enable_memory
        self.llm_client = llm_client  # For CentralPost synthesis
        
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
        
        # Performance metrics (for Hypothesis H2)
        self._metrics_enabled = enable_metrics
        self._start_time = time.time()
        self._total_messages_processed = 0
        self._processing_times: List[float] = []
        self._overhead_ratios: List[float] = []
        self._scaling_metrics: Dict[int, float] = {}
        
        # Memory systems (Priority 5: Memory and Context Persistence)
        self._memory_enabled = enable_memory
        if enable_memory:
            # Use separate database files to match GUI expectations
            self.knowledge_store = KnowledgeStore("felix_knowledge.db")  # GUI reads from this
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
        # Update accumulated state
        self._partial_thoughts[agent_id] = accumulated
        self._streaming_metadata[agent_id] = metadata

        # Emit event for GUI (real-time display)
        self._emit_streaming_event({
            "type": "partial_thought",
            "agent_id": agent_id,
            "partial": partial_content,
            "accumulated": accumulated,
            "progress": progress,
            "agent_type": metadata.get("agent_type"),
            "checkpoint": metadata.get("checkpoint"),
            "tokens_so_far": metadata.get("tokens_so_far", 0),
            "timestamp": time.time()
        })

        # Note: Do NOT synthesize here (wait for completion per hybrid approach)

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
        # Clean up streaming state
        metadata = self._streaming_metadata.pop(agent_id, {})
        self._partial_thoughts.pop(agent_id, None)

        # Log completion
        logger.info(f"✓ Streaming thought complete: {agent_id} (confidence: {confidence:.2f})")

        # Emit completion event
        self._emit_streaming_event({
            "type": "thought_complete",
            "agent_id": agent_id,
            "final_content": final_content,
            "confidence": confidence,
            "agent_type": metadata.get("agent_type"),
            "timestamp": time.time()
        })

        # Now consider synthesis with complete message (hybrid approach)
        # Note: Actual synthesis logic can be added here or handled by workflow

    def _emit_streaming_event(self, event: Dict[str, Any]) -> None:
        """
        Emit streaming event to registered callbacks (GUI listeners).

        Args:
            event: Event data to emit
        """
        for callback in self._streaming_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Streaming event callback failed: {e}")

    def register_streaming_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register callback for streaming events (for GUI updates).

        Args:
            callback: Function to call with streaming events
        """
        self._streaming_callbacks.append(callback)
        logger.info(f"Registered streaming callback (total: {len(self._streaming_callbacks)})")

    # ============================================================================
    # CENTRALPOST SYNTHESIS (Felix Architecture: CentralPost is Smart)
    # ============================================================================

    def synthesize_agent_outputs(self, task_description: str, max_messages: int = 20) -> Dict[str, Any]:
        """
        Synthesize final output from all agent communications.

        This is the core synthesis capability of CentralPost, replacing the need for
        synthesis agents. CentralPost represents the central axis of the helix where
        all agent trajectories converge.

        Args:
            task_description: Original task description
            max_messages: Maximum number of agent messages to include in synthesis

        Returns:
            Dict containing:
                - synthesis_content: Final synthesized output text
                - confidence: Synthesis confidence score (0.0-1.0)
                - temperature: Temperature used for synthesis
                - tokens_used: Number of tokens used
                - max_tokens: Token budget allocated
                - agents_synthesized: Number of agent outputs included
                - timestamp: Synthesis timestamp

        Raises:
            RuntimeError: If no LLM client available for synthesis
        """
        if not self.llm_client:
            raise RuntimeError("CentralPost synthesis requires LLM client. Pass llm_client to CentralPost.__init__()")

        logger.info("=" * 60)
        logger.info("CENTRALPOST SYNTHESIS STARTING")
        logger.info("=" * 60)

        # Gather recent agent messages
        messages = self.get_recent_messages(
            limit=max_messages,
            message_types=[MessageType.STATUS_UPDATE]
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

        # Calculate average confidence from agent outputs
        confidences = []
        for msg in messages:
            if msg.message_type == MessageType.STATUS_UPDATE:
                conf = msg.content.get('confidence', 0.0)
                if conf > 0:
                    confidences.append(conf)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Calculate adaptive synthesis parameters
        temperature = self._calculate_synthesis_temperature(avg_confidence)
        max_tokens = self._calculate_synthesis_tokens(len(messages))

        logger.info(f"Synthesis Parameters:")
        logger.info(f"  Agent messages: {len(messages)}")
        logger.info(f"  Average confidence: {avg_confidence:.2f}")
        logger.info(f"  Adaptive temperature: {temperature}")
        logger.info(f"  Adaptive token budget: {max_tokens}")

        # Build synthesis prompt
        user_prompt = self._build_synthesis_prompt(task_description, messages)

        # Helical-aware system prompt
        system_prompt = """You are the Central Post of the Felix helical multi-agent system.

Felix agents operate along a helical geometry:
- Top of helix: Broad exploration (research agents)
- Middle spiral: Focused analysis (analysis agents)
- Bottom convergence: Critical validation (critic agents)

You represent the central axis of this helix - the single point of truth that all agents spiral around.

Your synthesis represents the convergence point where all helical agent paths meet.

Synthesize the agent outputs below into a final answer that:
1. Captures insights from the exploration phase (research)
2. Integrates findings from the analysis phase (analysis)
3. Addresses concerns raised by the validation phase (critics)
4. Represents the natural convergence of all agent trajectories

This is the emergent output of the entire helical system."""

        # Call LLM for synthesis
        start_time = time.time()
        try:
            llm_response = self.llm_client.complete(
                agent_id="central_post_synthesizer",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            synthesis_time = time.time() - start_time

            logger.info(f"✓ Synthesis complete in {synthesis_time:.2f}s")
            logger.info(f"  Tokens used: {llm_response.tokens_used} / {max_tokens}")
            logger.info(f"  Content length: {len(llm_response.content)} chars")
            logger.info("=" * 60)

            return {
                "synthesis_content": llm_response.content,
                "confidence": 0.95,  # High confidence for CentralPost synthesis
                "temperature": temperature,
                "tokens_used": llm_response.tokens_used,
                "max_tokens": max_tokens,
                "agents_synthesized": len(messages),
                "avg_agent_confidence": avg_confidence,
                "synthesis_time": synthesis_time,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"✗ CentralPost synthesis failed: {e}")
            raise

    def _calculate_synthesis_temperature(self, avg_confidence: float) -> float:
        """
        Calculate adaptive temperature for synthesis based on agent confidence consensus.

        High confidence → focused synthesis (0.2)
        Medium confidence → balanced synthesis (0.3)
        Low confidence → creative integration (0.4)

        Args:
            avg_confidence: Average confidence from agent outputs (0.0-1.0)

        Returns:
            Temperature value (0.2-0.4)
        """
        if avg_confidence >= 0.9:
            return 0.2  # High confidence → very focused
        elif avg_confidence >= 0.75:
            return 0.3  # Medium confidence → balanced
        else:
            return 0.4  # Lower confidence → more creative integration

    def _calculate_synthesis_tokens(self, agent_count: int) -> int:
        """
        Calculate adaptive token budget for synthesis based on number of agents.

        More agents → more content to synthesize → larger budget

        Args:
            agent_count: Number of agent outputs to synthesize

        Returns:
            Token budget (1500-3000)
        """
        if agent_count >= 10:
            return 3000  # Many agents → comprehensive synthesis
        elif agent_count >= 5:
            return 2000  # Medium team → balanced synthesis
        else:
            return 1500  # Small team → focused synthesis

    def _build_synthesis_prompt(self, task_description: str, messages: List[Message]) -> str:
        """
        Build synthesis prompt from task description and agent messages.

        Args:
            task_description: Original task description
            messages: List of agent messages to synthesize

        Returns:
            Formatted synthesis prompt
        """
        prompt_parts = [
            f"Original Task: {task_description}",
            "",
            "Agent Communications to Synthesize:",
            ""
        ]

        # Add each agent output with metadata
        for i, msg in enumerate(messages, 1):
            if msg.message_type == MessageType.STATUS_UPDATE:
                agent_type = msg.content.get('agent_type', 'unknown')
                content = msg.content.get('content', '')
                confidence = msg.content.get('confidence', 0.0)

                prompt_parts.append(
                    f"{i}. {agent_type.upper()} Agent (confidence: {confidence:.2f}):"
                )
                prompt_parts.append(content)
                prompt_parts.append("")

        prompt_parts.append("---")
        prompt_parts.append("")
        prompt_parts.append("Create a comprehensive final synthesis that integrates all agent findings above.")

        return "\n".join(prompt_parts)

    def _handle_message(self, message: Message) -> None:
        """
        Handle specific message types (internal processing).

        Args:
            message: Message to handle
        """
        # Update agent registry with message metadata
        self._update_agent_registry_from_message(message)

        # Message type-specific handling
        if message.message_type == MessageType.TASK_REQUEST:
            self._handle_task_request(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
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
    
    def _handle_status_update(self, message: Message) -> None:
        """Handle status update from agent."""
        # Placeholder for status tracking logic
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

    # Performance metrics methods (for Hypothesis H2)
    
    def get_current_time(self) -> float:
        """Get current timestamp for performance measurements."""
        return time.time()
    
    def get_message_throughput(self) -> float:
        """
        Calculate message processing throughput.
        
        Returns:
            Messages processed per second
        """
        if not self._metrics_enabled or self._total_messages_processed == 0:
            return 0.0
        
        elapsed_time = time.time() - self._start_time
        if elapsed_time == 0:
            return 0.0
        
        return self._total_messages_processed / elapsed_time
    
    def measure_communication_overhead(self, num_messages: int, processing_time: float) -> float:
        """
        Measure communication overhead vs processing time.
        
        Args:
            num_messages: Number of messages in the measurement
            processing_time: Actual processing time for comparison
            
        Returns:
            Communication overhead time
        """
        if not self._metrics_enabled:
            return 0.0
        
        # Simulate communication overhead calculation
        if self._processing_times:
            avg_msg_time = sum(self._processing_times) / len(self._processing_times)
            communication_overhead = avg_msg_time * num_messages
            return communication_overhead
        
        return 0.0
    
    def record_overhead_ratio(self, overhead_ratio: float) -> None:
        """
        Record overhead ratio for hypothesis validation.
        
        Args:
            overhead_ratio: Communication overhead / processing time ratio
        """
        if self._metrics_enabled:
            self._overhead_ratios.append(overhead_ratio)
    
    def get_average_overhead_ratio(self) -> float:
        """
        Get average overhead ratio across all measurements.
        
        Returns:
            Average overhead ratio
        """
        if not self._overhead_ratios:
            return 0.0
        
        return sum(self._overhead_ratios) / len(self._overhead_ratios)
    
    def record_scaling_metric(self, agent_count: int, processing_time: float) -> None:
        """
        Record scaling performance metric.
        
        Args:
            agent_count: Number of agents in the test
            processing_time: Time to process messages from all agents
        """
        if self._metrics_enabled:
            self._scaling_metrics[agent_count] = processing_time
    
    def get_scaling_metrics(self) -> Dict[int, float]:
        """
        Get scaling performance metrics.
        
        Returns:
            Dictionary mapping agent count to processing time
        """
        return self._scaling_metrics.copy()
    
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
        if not self._metrics_enabled:
            return {"metrics_enabled": False}
        
        return {
            "metrics_enabled": True,
            "total_messages_processed": self._total_messages_processed,
            "message_throughput": self.get_message_throughput(),
            "average_overhead_ratio": self.get_average_overhead_ratio(),
            "scaling_metrics": self.get_scaling_metrics(),
            "active_connections": self.active_connections,
            "uptime": time.time() - self._start_time,
            "async_processors": len(self._async_processors),
            "async_queue_size": self._async_message_queue.qsize() if self._async_message_queue else 0
        }
    
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
        if not self._memory_enabled or not self.knowledge_store:
            return False
        
        try:
            # Convert confidence to ConfidenceLevel enum
            if confidence >= 0.8:
                confidence_level = ConfidenceLevel.HIGH
            elif confidence >= 0.6:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW
            
            # Store in knowledge base using correct method signature
            entry_id = self.knowledge_store.store_knowledge(
                knowledge_type=KnowledgeType.TASK_RESULT,
                content={"result": content, "confidence": confidence},
                confidence_level=confidence_level,
                source_agent=agent_id,
                domain=domain,
                tags=tags
            )
            return entry_id is not None
            
        except Exception as e:
            logger.error(f"Failed to store knowledge from agent {agent_id}: {e}")
            return False
    
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
        if not self._memory_enabled or not self.knowledge_store:
            return []
        
        try:
            from src.memory.knowledge_store import KnowledgeQuery
            query = KnowledgeQuery(
                knowledge_types=[knowledge_type] if knowledge_type else None,
                domains=[domain] if domain else None,
                content_keywords=keywords,
                min_confidence=min_confidence,
                limit=limit
            )
            return self.knowledge_store.retrieve_knowledge(query)
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []
    
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
        if not self._memory_enabled or not self.task_memory:
            return {}

        try:
            from src.memory.task_memory import TaskComplexity
            # Convert string complexity to enum
            complexity_enum = TaskComplexity.MODERATE
            if complexity.upper() == "SIMPLE":
                complexity_enum = TaskComplexity.SIMPLE
            elif complexity.upper() == "COMPLEX":
                complexity_enum = TaskComplexity.COMPLEX
            elif complexity.upper() == "VERY_COMPLEX":
                complexity_enum = TaskComplexity.VERY_COMPLEX
                
            return self.task_memory.recommend_strategy(
                task_description=task_description,
                task_type=task_type,
                complexity=complexity_enum
            )
        except Exception as e:
            logger.error(f"Failed to get strategy recommendations: {e}")
            return {}
    
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
        if not self._memory_enabled or not self.context_compressor:
            return None
        
        try:
            # Convert string context to dict format expected by compressor
            context_dict = {"main_content": context}
            return self.context_compressor.compress_context(
                context=context_dict,
                target_size=target_size,
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Failed to compress context: {e}")
            return None
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory system status and contents.

        Returns:
            Dictionary with memory system summary
        """
        if not self._memory_enabled:
            return {
                "knowledge_entries": 0,
                "task_patterns": 0,
                "memory_enabled": False
            }

        try:
            summary: Dict[str, Any] = {"memory_enabled": True}
            
            if self.knowledge_store:
                # Get knowledge entry count using proper query
                from src.memory.knowledge_store import KnowledgeQuery
                query = KnowledgeQuery(limit=1000)
                all_knowledge = self.knowledge_store.retrieve_knowledge(query)
                summary["knowledge_entries"] = len(all_knowledge)
                
                # Get domain breakdown
                domains: Dict[str, int] = {}
                for entry in all_knowledge:
                    domains[entry.domain] = domains.get(entry.domain, 0) + 1
                summary["knowledge_by_domain"] = domains
            else:
                summary["knowledge_entries"] = 0
                summary["knowledge_by_domain"] = {}
            
            if self.task_memory:
                # Get task pattern count and summary
                memory_summary = self.task_memory.get_memory_summary()
                summary["task_patterns"] = memory_summary.get("total_patterns", 0)
                summary["task_executions"] = memory_summary.get("total_executions", 0)
                
                # Handle success rate calculation from outcome distribution
                outcome_dist = memory_summary.get("outcome_distribution", {})
                total_executions = sum(outcome_dist.values()) if outcome_dist else 0
                if total_executions > 0:
                    successful_outcomes = outcome_dist.get("success", 0) + outcome_dist.get("partial_success", 0)
                    summary["success_rate"] = successful_outcomes / total_executions
                else:
                    summary["success_rate"] = 0.0
                
                # Get top task types
                summary["top_task_types"] = memory_summary.get("top_task_types", {})
                summary["success_by_complexity"] = memory_summary.get("success_by_complexity", {})
            else:
                summary["task_patterns"] = 0
                summary["task_executions"] = 0
                summary["success_rate"] = 0.0
                summary["top_task_types"] = {}
                summary["success_by_complexity"] = {}
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get memory summary: {e}")
            return {
                "knowledge_entries": 0,
                "task_patterns": 0,
                "memory_enabled": True,
                "error": str(e)
            }


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
                 max_agents: int = 25, token_budget_limit: int = 10000):
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
        """
        self.helix = helix
        self.llm_client = llm_client
        self.token_budget_manager = token_budget_manager
        self.random_seed = random_seed
        self._agent_counter = 0
        self.enable_dynamic_spawning = enable_dynamic_spawning
        
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
            max_tokens=1200
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
            max_tokens=1200
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
            max_tokens=1200
        )
    
    def assess_team_needs(self, processed_messages: List[Message], 
                         current_time: float, current_agents: Optional[List["LLMAgent"]] = None) -> List["LLMAgent"]:
        """
        Assess current team composition and suggest new agents if needed.
        
        Enhanced with DynamicSpawning system that provides:
        - Confidence monitoring with trend analysis
        - Content analysis for contradictions and gaps
        - Team size optimization based on task complexity
        - Resource-aware spawning decisions
        
        Falls back to basic heuristics if dynamic spawning is disabled.
        
        Args:
            processed_messages: Messages processed so far
            current_time: Current simulation time
            current_agents: List of currently active agents
            
        Returns:
            List of recommended new agents to spawn
        """
        # Use dynamic spawning if enabled and available
        if self.enable_dynamic_spawning and self.dynamic_spawner:
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
        
        # Check for need for alternative synthesis
        synthesis_count = sum(1 for msg in recent_messages
                            if msg.content.get("agent_type") == "synthesis")
        
        if synthesis_count == 0 and current_time > 0.6:
            # Late in process but no synthesis yet
            synthesis = self.create_synthesis_agent(
                output_format="comprehensive",
                spawn_time_range=(current_time + 0.1, current_time + 0.25)
            )
            recommended_agents.append(synthesis)
        
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
