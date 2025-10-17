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

Mathematical references:
- docs/architecture/core/mathematical_model.md, Section 5: Spoke geometry and communication complexity
- docs/architecture/core/hypothesis_mathematics.md, Section H2: Communication overhead analysis and proofs
- Theoretical proof of O(N) vs O(N²) scaling advantage in hypothesis documentation

Implementation supports rigorous testing of Hypothesis H2 communication efficiency claims.
"""

import time
import uuid
import random
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
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


@dataclass
class Message:
    """Message structure for communication between agents and central post."""
    sender_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CentralPost:
    """
    Central coordination system managing all agent communication.
    
    The central post acts as the hub in the spoke-based communication model,
    processing messages from agents and coordinating task assignments.
    """
    
    def __init__(self, max_agents: int = 133, enable_metrics: bool = False,
                 enable_memory: bool = True, memory_db_path: str = "felix_memory.db"):
        """
        Initialize central post with configuration parameters.

        Args:
            max_agents: Maximum number of concurrent agent connections
            enable_metrics: Whether to collect performance metrics
            enable_memory: Whether to enable persistent memory systems
            memory_db_path: Path to the memory database file
        """
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        self.enable_memory = enable_memory
        
        # Connection management
        self._registered_agents: Dict[str, str] = {}  # agent_id -> connection_id
        self._connection_times: Dict[str, float] = {}  # agent_id -> registration_time
        
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
            self.knowledge_store = KnowledgeStore(memory_db_path)
            self.task_memory = TaskMemory(memory_db_path)
            self.context_compressor = ContextCompressor()
        else:
            self.knowledge_store = None
            self.task_memory = None
            self.context_compressor = None
        
        # System state
        self._is_active = True
    
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
    
    def register_agent(self, agent) -> str:
        """
        Register an agent with the central post.
        
        Args:
            agent: Agent instance to register
            
        Returns:
            Connection ID for the registered agent
            
        Raises:
            ValueError: If maximum connections exceeded or agent already registered
        """
        if self.active_connections >= self.max_agents:
            raise ValueError("Maximum agent connections exceeded")
        
        if agent.agent_id in self._registered_agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        
        # Create unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Register agent
        self._registered_agents[agent.agent_id] = connection_id
        self._connection_times[agent.agent_id] = time.time()
        
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
    
    def _handle_message(self, message: Message) -> None:
        """
        Handle specific message types (internal processing).
        
        Args:
            message: Message to handle
        """
        # Message type-specific handling
        if message.message_type == MessageType.TASK_REQUEST:
            self._handle_task_request(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            self._handle_status_update(message)
        elif message.message_type == MessageType.TASK_COMPLETE:
            self._handle_task_completion(message)
        elif message.message_type == MessageType.ERROR_REPORT:
            self._handle_error_report(message)
        # Add more handlers as needed
    
    async def _handle_message_async(self, message: Message) -> None:
        """
        Handle specific message types asynchronously (internal processing).
        
        Args:
            message: Message to handle
        """
        # Message type-specific async handling
        if message.message_type == MessageType.TASK_REQUEST:
            await self._handle_task_request_async(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self._handle_status_update_async(message)
        elif message.message_type == MessageType.TASK_COMPLETE:
            await self._handle_task_completion_async(message)
        elif message.message_type == MessageType.ERROR_REPORT:
            await self._handle_error_report_async(message)
        # Add more handlers as needed
    
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
            from memory.knowledge_store import KnowledgeQuery
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
            from memory.task_memory import TaskComplexity
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
                from memory.knowledge_store import KnowledgeQuery
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
                 max_agents: int = 15, token_budget_limit: int = 10000):
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
            from agents.dynamic_spawning import DynamicSpawning
            self.dynamic_spawner = DynamicSpawning(
                agent_factory=self,
                confidence_threshold=0.7,
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
        from agents.specialized_agents import ResearchAgent
        
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
            max_tokens=800
        )
    
    def create_analysis_agent(self, analysis_type: str = "general",
                            spawn_time_range: Tuple[float, float] = (0.2, 0.7)) -> "LLMAgent":
        """Create an analysis agent with random spawn time in specified range."""
        from agents.specialized_agents import AnalysisAgent
        
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
            max_tokens=800
        )
    
    def create_critic_agent(self, review_focus: str = "general",
                          spawn_time_range: Tuple[float, float] = (0.5, 0.8)) -> "LLMAgent":
        """Create a critic agent with random spawn time in specified range."""
        from agents.specialized_agents import CriticAgent
        
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
            max_tokens=800
        )
    
    def create_synthesis_agent(self, output_format: str = "general",
                             spawn_time_range: Tuple[float, float] = (0.7, 0.95)) -> "LLMAgent":
        """Create a synthesis agent with random spawn time in specified range."""
        from agents.specialized_agents import SynthesisAgent
        
        spawn_time = random.uniform(*spawn_time_range)
        agent_id = f"dynamic_synthesis_{self._agent_counter:03d}"
        self._agent_counter += 1
        
        return SynthesisAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=self.helix,
            llm_client=self.llm_client,
            output_format=output_format,
            token_budget_manager=self.token_budget_manager,
            max_tokens=1200  # Increased for comprehensive blog posts
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
