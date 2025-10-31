"""
Pydantic models for Felix REST API request/response schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================================================
# System Models
# ============================================================================

class SystemStatus(BaseModel):
    """System status response."""
    status: str = Field(..., description="System status: running, stopped, error")
    felix_version: str = Field(..., description="Felix version")
    api_version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    active_workflows: int = Field(..., description="Number of active workflows")
    active_agents: int = Field(..., description="Number of active agents")
    llm_provider: str = Field(..., description="Current LLM provider")
    knowledge_brain_enabled: bool = Field(..., description="Whether knowledge brain is enabled")


class SystemConfig(BaseModel):
    """System configuration."""
    lm_host: str = "127.0.0.1"
    lm_port: int = 1234
    helix_top_radius: float = 3.0
    helix_bottom_radius: float = 0.5
    helix_height: float = 8.0
    helix_turns: int = 2
    max_agents: int = 10
    base_token_budget: int = 2500
    enable_knowledge_brain: bool = False
    verbose_llm_logging: bool = False


# ============================================================================
# Workflow Models
# ============================================================================

class WorkflowRequest(BaseModel):
    """Request to create a new workflow."""
    task: str = Field(..., description="Task description", min_length=1, max_length=10000)
    max_steps: Optional[int] = Field(None, description="Maximum workflow steps", ge=1, le=50)
    parent_workflow_id: Optional[str] = Field(None, description="Parent workflow ID for conversation continuity")
    enable_web_search: bool = Field(True, description="Enable web search for this workflow")

    class Config:
        json_schema_extra = {
            "example": {
                "task": "Explain quantum computing in simple terms",
                "max_steps": 10,
                "enable_web_search": True
            }
        }


class WorkflowStatus(str, Enum):
    """Workflow status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentInfo(BaseModel):
    """Information about an agent in a workflow."""
    agent_id: str
    agent_type: str
    spawn_time: float
    confidence: Optional[float] = None


class SynthesisResult(BaseModel):
    """Workflow synthesis result."""
    content: str
    confidence: float
    agents_synthesized: int
    token_count: Optional[int] = None


class WorkflowResponse(BaseModel):
    """Workflow execution response."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    status: WorkflowStatus = Field(..., description="Current workflow status")
    task: str = Field(..., description="Original task")
    created_at: datetime = Field(..., description="Workflow creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Workflow completion timestamp")
    agents_spawned: List[AgentInfo] = Field(default_factory=list, description="Agents spawned for this workflow")
    synthesis: Optional[SynthesisResult] = Field(None, description="Final synthesis result")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_12345abc",
                "status": "completed",
                "task": "Explain quantum computing",
                "created_at": "2025-10-30T10:00:00Z",
                "completed_at": "2025-10-30T10:02:30Z",
                "agents_spawned": [
                    {"agent_id": "research_001", "agent_type": "research", "spawn_time": 0.1, "confidence": 0.85}
                ],
                "synthesis": {
                    "content": "Quantum computing is...",
                    "confidence": 0.87,
                    "agents_synthesized": 3
                }
            }
        }


class WorkflowListResponse(BaseModel):
    """List of workflows response."""
    workflows: List[WorkflowResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Agent Models
# ============================================================================

class AgentCreateRequest(BaseModel):
    """Request to spawn a new agent."""
    agent_type: str = Field(..., description="Type of agent to spawn (research, analysis, critic, etc.)")
    spawn_time: Optional[float] = Field(None, description="Spawn time (0.0-1.0), auto-calculated if None", ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_type": "research",
                "parameters": {
                    "research_domain": "technical"
                }
            }
        }


class AgentResponse(BaseModel):
    """Agent information response."""
    agent_id: str
    agent_type: str
    spawn_time: float
    status: str
    created_at: datetime
    current_task: Optional[str] = None
    confidence: Optional[float] = None
    messages_processed: int = 0
    tokens_used: int = 0


class AgentListResponse(BaseModel):
    """List of agents response."""
    agents: List[AgentResponse]
    total: int


class AgentPluginMetadata(BaseModel):
    """Agent plugin metadata."""
    agent_type: str
    display_name: str
    description: str
    spawn_range: tuple[float, float]
    capabilities: List[str]
    tags: List[str]
    default_tokens: int
    version: str
    author: Optional[str] = None
    priority: int


class AgentPluginListResponse(BaseModel):
    """List of agent plugins."""
    plugins: List[AgentPluginMetadata]
    total: int
    builtin_count: int
    external_count: int


# ============================================================================
# Knowledge Brain Models
# ============================================================================

class DocumentUploadRequest(BaseModel):
    """Request to upload a document for ingestion."""
    file_path: str = Field(..., description="Path to document file")
    process_immediately: bool = Field(True, description="Process document immediately vs batch")


class DocumentResponse(BaseModel):
    """Document information response."""
    document_id: int
    file_path: str
    file_name: str
    status: str  # pending, processing, completed, failed
    chunks_created: int
    concepts_extracted: int
    created_at: datetime
    processed_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentResponse]
    total: int


class KnowledgeSearchRequest(BaseModel):
    """Request to search knowledge base."""
    query: str = Field(..., description="Search query", min_length=1)
    limit: int = Field(10, description="Maximum results to return", ge=1, le=100)
    task_complexity: str = Field("medium", description="Task complexity: simple, medium, complex")
    include_graph: bool = Field(False, description="Include related concepts from graph")


class KnowledgeEntry(BaseModel):
    """Knowledge entry."""
    entry_id: int
    content: str
    domain: str
    confidence: float
    source_agent: str
    created_at: datetime


class KnowledgeSearchResponse(BaseModel):
    """Knowledge search results."""
    results: List[KnowledgeEntry]
    total: int
    query: str


class ConceptResponse(BaseModel):
    """Concept information."""
    concept: str
    occurrences: int
    domain: str
    related_concepts: List[str] = Field(default_factory=list)


class ConceptListResponse(BaseModel):
    """List of concepts."""
    concepts: List[ConceptResponse]
    total: int


class DaemonStatusResponse(BaseModel):
    """Knowledge daemon status."""
    running: bool
    mode: str  # batch, watch, refine
    documents_processed: int
    documents_pending: int
    last_run: Optional[datetime] = None
    next_refinement: Optional[datetime] = None


# ============================================================================
# Memory & History Models
# ============================================================================

# Task Memory Models

class TaskComplexity(str, Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class TaskOutcome(str, Enum):
    """Task execution outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class TaskPatternModel(BaseModel):
    """Task pattern information."""
    pattern_id: str
    task_type: str
    complexity: TaskComplexity
    keywords: List[str]
    typical_duration: Optional[float] = None
    success_rate: float
    failure_modes: List[str] = []
    optimal_strategies: List[str] = []
    required_agents: List[str] = []
    context_requirements: Dict[str, Any] = {}
    usage_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "pattern_id": "research_complex_abc123",
                "task_type": "research",
                "complexity": "complex",
                "keywords": ["quantum", "computing", "algorithms"],
                "typical_duration": 45.5,
                "success_rate": 0.87,
                "optimal_strategies": ["multi-agent", "web-search"],
                "required_agents": ["research", "analysis", "critic"],
                "usage_count": 15
            }
        }


class TaskExecutionModel(BaseModel):
    """Task execution record."""
    execution_id: str
    task_description: str
    task_type: str
    complexity: TaskComplexity
    outcome: TaskOutcome
    duration: float
    agents_used: List[str]
    strategies_used: List[str]
    context_size: int
    error_messages: List[str] = []
    success_metrics: Dict[str, Any] = {}
    patterns_matched: List[str] = []
    created_at: datetime


class TaskPatternQueryRequest(BaseModel):
    """Query request for task patterns."""
    task_types: Optional[List[str]] = None
    complexity_levels: Optional[List[TaskComplexity]] = None
    keywords: Optional[List[str]] = None
    min_success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_duration: Optional[float] = None
    limit: int = Field(50, ge=1, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "task_types": ["research", "analysis"],
                "complexity_levels": ["complex"],
                "min_success_rate": 0.8,
                "limit": 20
            }
        }


class TaskPatternListResponse(BaseModel):
    """List of task patterns."""
    patterns: List[TaskPatternModel]
    total: int


class TaskExecutionQueryRequest(BaseModel):
    """Query request for task executions."""
    task_types: Optional[List[str]] = None
    complexity_levels: Optional[List[TaskComplexity]] = None
    outcomes: Optional[List[TaskOutcome]] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    limit: int = Field(50, ge=1, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "task_types": ["research"],
                "outcomes": ["success", "partial_success"],
                "limit": 50
            }
        }


class TaskExecutionListResponse(BaseModel):
    """List of task executions."""
    executions: List[TaskExecutionModel]
    total: int


class TaskExecutionRequest(BaseModel):
    """Record new task execution."""
    task_description: str = Field(..., min_length=1, max_length=5000)
    task_type: str = Field(..., min_length=1)
    complexity: TaskComplexity
    outcome: TaskOutcome
    duration: float = Field(..., gt=0)
    agents_used: List[str] = []
    strategies_used: List[str] = []
    context_size: int = Field(0, ge=0)
    error_messages: List[str] = []
    success_metrics: Dict[str, Any] = {}

    class Config:
        json_schema_extra = {
            "example": {
                "task_description": "Research quantum computing fundamentals",
                "task_type": "research",
                "complexity": "complex",
                "outcome": "success",
                "duration": 42.5,
                "agents_used": ["research_001", "analysis_002"],
                "strategies_used": ["web-search", "multi-agent"],
                "context_size": 3500
            }
        }


class TaskExecutionResponse(BaseModel):
    """Task execution response."""
    execution_id: str
    patterns_matched: List[str]
    patterns_updated: List[str]
    message: str


class StrategyRecommendationRequest(BaseModel):
    """Request for strategy recommendation."""
    task_description: str = Field(..., min_length=1, max_length=5000)
    task_type: str = Field(..., min_length=1)
    complexity: TaskComplexity

    class Config:
        json_schema_extra = {
            "example": {
                "task_description": "Analyze market trends for renewable energy",
                "task_type": "analysis",
                "complexity": "complex"
            }
        }


class StrategyRecommendationResponse(BaseModel):
    """Strategy recommendation response."""
    recommended_strategies: List[str]
    recommended_agents: List[str]
    estimated_duration: Optional[float] = None
    success_probability: float
    similar_patterns: List[TaskPatternModel]
    confidence: float

    class Config:
        json_schema_extra = {
            "example": {
                "recommended_strategies": ["web-search", "multi-agent", "knowledge-augment"],
                "recommended_agents": ["research", "analysis", "critic"],
                "estimated_duration": 45.0,
                "success_probability": 0.87,
                "confidence": 0.92
            }
        }


class TaskMemorySummaryResponse(BaseModel):
    """Task memory statistics."""
    total_patterns: int
    total_executions: int
    average_success_rate: float
    most_common_task_types: Dict[str, int]
    complexity_distribution: Dict[TaskComplexity, int]
    outcome_distribution: Dict[TaskOutcome, int]


# Workflow History Models

class WorkflowHistoryModel(BaseModel):
    """Workflow history record."""
    workflow_id: int
    task_input: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    final_synthesis: Optional[str] = None
    confidence: Optional[float] = None
    agents_count: int
    tokens_used: int
    max_tokens: int
    processing_time: float
    temperature: float
    metadata: Dict[str, Any] = {}
    parent_workflow_id: Optional[int] = None
    conversation_thread_id: Optional[str] = None


class WorkflowHistoryQueryRequest(BaseModel):
    """Query parameters for workflow history."""
    status: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    search_query: Optional[str] = Field(None, description="Search in task_input and synthesis")
    parent_workflow_id: Optional[int] = None
    conversation_thread_id: Optional[str] = None
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "completed",
                "from_date": "2025-10-01T00:00:00Z",
                "search_query": "quantum computing",
                "limit": 50
            }
        }


class WorkflowHistoryListResponse(BaseModel):
    """List of workflow history records."""
    workflows: List[WorkflowHistoryModel]
    total: int
    offset: int
    limit: int


class WorkflowThreadResponse(BaseModel):
    """Conversation thread response."""
    thread_id: str
    root_workflow: WorkflowHistoryModel
    child_workflows: List[WorkflowHistoryModel]
    total_workflows: int
    thread_depth: int


class WorkflowAnalyticsResponse(BaseModel):
    """Workflow analytics and metrics."""
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_confidence: float
    average_agents_count: float
    average_processing_time: float
    average_tokens_used: float
    status_distribution: Dict[str, int]
    workflows_by_date: Dict[str, int]

    class Config:
        json_schema_extra = {
            "example": {
                "total_workflows": 150,
                "completed_workflows": 142,
                "failed_workflows": 8,
                "average_confidence": 0.85,
                "average_agents_count": 4.2,
                "average_processing_time": 38.5,
                "status_distribution": {"completed": 142, "failed": 8}
            }
        }


# Knowledge Memory Models (extends knowledge store for memory API)

class KnowledgeType(str, Enum):
    """Knowledge entry types."""
    TASK_RESULT = "task_result"
    AGENT_INSIGHT = "agent_insight"
    PATTERN_RECOGNITION = "pattern_recognition"
    FAILURE_ANALYSIS = "failure_analysis"
    OPTIMIZATION_DATA = "optimization_data"
    DOMAIN_EXPERTISE = "domain_expertise"


class ConfidenceLevel(str, Enum):
    """Confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class KnowledgeEntryModel(BaseModel):
    """Knowledge entry from knowledge store."""
    knowledge_id: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    confidence_level: ConfidenceLevel
    source_agent: str
    domain: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    access_count: int
    success_rate: float
    validation_score: float
    validation_status: str


class KnowledgeQueryRequest(BaseModel):
    """Query for knowledge entries."""
    knowledge_types: Optional[List[KnowledgeType]] = None
    domains: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_confidence: Optional[ConfidenceLevel] = None
    min_success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    content_keywords: Optional[List[str]] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    task_type: Optional[str] = Field(None, description="For meta-learning boost")
    task_complexity: Optional[str] = None
    limit: int = Field(50, ge=1, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "knowledge_types": ["domain_expertise", "agent_insight"],
                "domains": ["computer_science", "physics"],
                "min_confidence": "high",
                "min_success_rate": 0.8,
                "limit": 20
            }
        }


class KnowledgeStoreRequest(BaseModel):
    """Store new knowledge entry."""
    knowledge_type: KnowledgeType
    content: Dict[str, Any] = Field(..., min_items=1)
    confidence_level: ConfidenceLevel
    source_agent: str = Field(..., min_length=1)
    domain: str = Field(..., min_length=1)
    tags: List[str] = []

    class Config:
        json_schema_extra = {
            "example": {
                "knowledge_type": "domain_expertise",
                "content": {"concept": "quantum entanglement", "definition": "..."},
                "confidence_level": "high",
                "source_agent": "research_001",
                "domain": "physics",
                "tags": ["quantum", "physics", "advanced"]
            }
        }


class KnowledgeStoreResponse(BaseModel):
    """Knowledge store response."""
    knowledge_id: str
    stored: bool
    updated: bool
    message: str


class KnowledgeListResponse(BaseModel):
    """List of knowledge entries."""
    entries: List[KnowledgeEntryModel]
    total: int
    offset: int = 0
    limit: int = 50


class KnowledgeUsageRequest(BaseModel):
    """Record knowledge usage."""
    knowledge_id: str
    workflow_id: str
    task_type: str
    task_complexity: str
    useful_score: float = Field(..., ge=0.0, le=1.0, description="How useful was this knowledge")
    retrieval_method: str = Field(..., description="sql, semantic, or hybrid")

    class Config:
        json_schema_extra = {
            "example": {
                "knowledge_id": "kb_abc123",
                "workflow_id": "wf_xyz789",
                "task_type": "research",
                "task_complexity": "complex",
                "useful_score": 0.9,
                "retrieval_method": "semantic"
            }
        }


class KnowledgeUsageResponse(BaseModel):
    """Knowledge usage response."""
    recorded: bool
    message: str


class KnowledgeSuccessRateRequest(BaseModel):
    """Update knowledge success rate."""
    new_success_rate: float = Field(..., ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "new_success_rate": 0.92
            }
        }


class KnowledgeRelationshipItem(BaseModel):
    """Knowledge relationship."""
    target_knowledge_id: str
    target_content_preview: str
    relationship_strength: float
    relationship_type: str


class KnowledgeRelationshipsResponse(BaseModel):
    """Knowledge relationships."""
    knowledge_id: str
    relationships: List[KnowledgeRelationshipItem]
    total: int


class KnowledgeMemorySummaryResponse(BaseModel):
    """Knowledge memory statistics."""
    total_entries: int
    entries_by_type: Dict[KnowledgeType, int]
    entries_by_domain: Dict[str, int]
    entries_by_confidence: Dict[ConfidenceLevel, int]
    average_success_rate: float
    total_access_count: int


# Context Compression Models

class CompressionStrategy(str, Enum):
    """Compression strategies."""
    EXTRACTIVE_SUMMARY = "extractive_summary"
    ABSTRACTIVE_SUMMARY = "abstractive_summary"
    KEYWORD_EXTRACTION = "keyword_extraction"
    HIERARCHICAL_SUMMARY = "hierarchical_summary"
    RELEVANCE_FILTERING = "relevance_filtering"
    PROGRESSIVE_REFINEMENT = "progressive_refinement"


class CompressionLevel(str, Enum):
    """Compression levels."""
    LIGHT = "light"      # 80%
    MODERATE = "moderate"  # 60%
    HEAVY = "heavy"      # 40%
    EXTREME = "extreme"    # 20%


class CompressionRequest(BaseModel):
    """Context compression request."""
    context: Dict[str, Any] = Field(..., description="Context to compress")
    strategy: CompressionStrategy = CompressionStrategy.PROGRESSIVE_REFINEMENT
    level: CompressionLevel = CompressionLevel.MODERATE
    preserve_keywords: List[str] = []
    preserve_structure: bool = True
    topic_keywords: Optional[List[str]] = Field(None, description="For relevance filtering")

    class Config:
        json_schema_extra = {
            "example": {
                "context": {"agent_outputs": ["...", "..."], "synthesis": "..."},
                "strategy": "progressive_refinement",
                "level": "moderate",
                "preserve_keywords": ["quantum", "entanglement"]
            }
        }


class CompressionResponse(BaseModel):
    """Compression result."""
    context_id: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    level_used: CompressionLevel
    compressed_content: Dict[str, Any]
    relevance_scores: Dict[str, float]
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "context_id": "ctx_abc123",
                "original_size": 5000,
                "compressed_size": 3000,
                "compression_ratio": 0.6,
                "strategy_used": "progressive_refinement",
                "level_used": "moderate"
            }
        }


class CompressionStatsResponse(BaseModel):
    """Compression system statistics."""
    max_context_size: int
    default_strategy: CompressionStrategy
    default_level: CompressionLevel
    available_strategies: List[CompressionStrategy]
    available_levels: List[CompressionLevel]


class CompressionConfigRequest(BaseModel):
    """Update compression configuration."""
    max_context_size: Optional[int] = Field(None, ge=1000)
    strategy: Optional[CompressionStrategy] = None
    level: Optional[CompressionLevel] = None
    preserve_keywords: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "max_context_size": 5000,
                "strategy": "hierarchical_summary",
                "level": "moderate"
            }
        }


# ============================================================================
# Command Execution Models
# ============================================================================

class CommandExecuteRequest(BaseModel):
    """Request to execute a system command."""
    command: str = Field(..., description="Command to execute", min_length=1)
    timeout: Optional[int] = Field(300, description="Timeout in seconds", ge=1, le=600)
    agent_id: Optional[str] = Field(None, description="Agent requesting the command")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")


class CommandResult(BaseModel):
    """Command execution result."""
    execution_id: str
    command: str
    status: str  # pending_approval, running, completed, failed, cancelled
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class ApprovalRequest(BaseModel):
    """Approval request information."""
    approval_id: str
    command: str
    agent_id: str
    context: Dict[str, Any]
    trust_level: str  # SAFE, REVIEW, BLOCKED
    requested_at: datetime


class ApprovalDecision(BaseModel):
    """Approval decision."""
    decision: str = Field(..., description="approve, reject, or modify")
    modified_command: Optional[str] = Field(None, description="Modified command if decision is 'modify'")
    reason: Optional[str] = Field(None, description="Reason for decision")


class ApprovalResponse(BaseModel):
    """Approval response."""
    approval_id: str
    decision: str
    decided_by: str
    decided_at: datetime
    execution_id: Optional[str] = None


# ============================================================================
# WebSocket Event Models
# ============================================================================

class WebSocketEvent(BaseModel):
    """Base WebSocket event."""
    type: str
    timestamp: datetime
    data: Dict[str, Any]


class WorkflowProgressEvent(BaseModel):
    """Workflow progress event."""
    type: str = "progress"
    workflow_id: str
    status: str
    progress: float  # 0.0-1.0
    message: str


class AgentSpawnedEvent(BaseModel):
    """Agent spawned event."""
    type: str = "agent_spawned"
    workflow_id: str
    agent_id: str
    agent_type: str
    spawn_time: float


class AgentOutputEvent(BaseModel):
    """Agent output event."""
    type: str = "agent_output"
    workflow_id: str
    agent_id: str
    content: str
    confidence: float


class WorkflowCompleteEvent(BaseModel):
    """Workflow complete event."""
    type: str = "workflow_complete"
    workflow_id: str
    status: str
    synthesis: Optional[SynthesisResult] = None


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Task description is required",
                "details": {"field": "task"},
                "timestamp": "2025-10-30T10:00:00Z"
            }
        }


# ============================================================================
# Knowledge Brain Models
# ============================================================================

# Document Models

class DocumentMetadataModel(BaseModel):
    """Document metadata."""
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    file_hash: str
    page_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class DocumentIngestRequest(BaseModel):
    """Document ingest request."""
    file_path: str = Field(..., description="Path to document file", min_length=1)
    process_immediately: bool = Field(True, description="Process document immediately")

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/document.pdf",
                "process_immediately": True
            }
        }


class DocumentIngestResponse(BaseModel):
    """Document ingest response."""
    document_id: str
    file_name: str
    status: str  # processing, complete, failed
    chunks_count: int
    metadata: DocumentMetadataModel
    message: str


class DocumentBatchRequest(BaseModel):
    """Batch document processing request."""
    directory_path: str = Field(..., description="Directory containing documents")
    recursive: bool = Field(True, description="Process subdirectories")
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to match (e.g. ['*.pdf', '*.txt'])")

    class Config:
        json_schema_extra = {
            "example": {
                "directory_path": "/path/to/documents",
                "recursive": True,
                "file_patterns": ["*.pdf", "*.txt"]
            }
        }


class DocumentBatchResponse(BaseModel):
    """Batch processing response."""
    total_files: int
    processed: int
    failed: int
    documents: List[DocumentIngestResponse]
    processing_time_seconds: float


class DocumentStatus(str, Enum):
    """Document processing status."""
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class DocumentListItem(BaseModel):
    """Document list item."""
    document_id: str
    file_name: str
    file_type: str
    file_size: int
    status: DocumentStatus
    chunks_count: Optional[int] = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentListItem]
    total: int
    filtered_by_status: Optional[DocumentStatus] = None


class DocumentDetailResponse(BaseModel):
    """Detailed document information."""
    document_id: str
    metadata: DocumentMetadataModel
    status: DocumentStatus
    chunks_count: int
    concepts_extracted: int
    created_at: datetime
    updated_at: datetime


# Search Models

class SearchRequest(BaseModel):
    """Knowledge search request."""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    task_type: Optional[str] = Field(None, description="Task type for meta-learning boost")
    task_complexity: Optional[str] = Field(None, description="Task complexity (simple/medium/complex)")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    domains: Optional[List[str]] = Field(None, description="Filter by domains")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "quantum computing fundamentals",
                "task_type": "research",
                "top_k": 10,
                "min_confidence": 0.7,
                "domains": ["physics", "computer_science"]
            }
        }


class SearchResultItem(BaseModel):
    """Single search result."""
    knowledge_id: str
    content: str
    relevance_score: float
    confidence: float
    domain: Optional[str] = None
    source_document_id: Optional[str] = None
    tags: List[str] = []


class SearchResponse(BaseModel):
    """Search results."""
    query: str
    results: List[SearchResultItem]
    total_results: int
    retrieval_method: str  # embedding, tfidf, fts5
    processing_time_ms: float


class AugmentRequest(BaseModel):
    """Context augmentation request."""
    task_description: str = Field(..., min_length=1, max_length=2000)
    task_type: Optional[str] = None
    max_concepts: int = Field(10, ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "task_description": "Write a report on renewable energy",
                "task_type": "analysis",
                "max_concepts": 10
            }
        }


class AugmentResponse(BaseModel):
    """Augmented context response."""
    task_description: str
    augmented_context: str
    concepts_used: int
    retrieval_method: str


# Knowledge Graph Models

class GraphBuildRequest(BaseModel):
    """Knowledge graph build request."""
    document_id: Optional[str] = Field(None, description="Build graph for specific document, or None for global")
    max_documents: Optional[int] = Field(None, description="Max documents for global graph")
    similarity_threshold: float = Field(0.75, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": None,
                "max_documents": 100,
                "similarity_threshold": 0.75
            }
        }


class GraphBuildResponse(BaseModel):
    """Graph build response."""
    relationships_created: int
    concepts_processed: int
    documents_processed: Optional[int] = None
    entities_linked: Optional[int] = None
    concepts_merged: Optional[int] = None
    processing_time_seconds: float


class RelationshipItem(BaseModel):
    """Knowledge relationship."""
    source_id: str
    source_content: str
    target_id: str
    target_content: str
    relationship_type: str  # related_to, similar_to, cooccurs_with, etc.
    strength: float
    basis: str  # explicit_mention, embedding_similarity, cooccurrence


class GraphRelationshipsRequest(BaseModel):
    """Get relationships for a concept."""
    concept_id: str = Field(..., description="Knowledge ID")
    max_depth: int = Field(1, ge=1, le=3, description="Traversal depth")
    min_strength: float = Field(0.5, ge=0.0, le=1.0)


class GraphRelationshipsResponse(BaseModel):
    """Concept relationships."""
    concept_id: str
    concept_content: str
    relationships: List[RelationshipItem]
    total_relationships: int


class GraphStatisticsResponse(BaseModel):
    """Knowledge graph statistics."""
    total_nodes: int
    total_relationships: int
    nodes_with_relationships: int
    average_degree: float
    documents_covered: int
    relationship_types: Dict[str, int]


# Daemon Models

class DaemonStatusResponse(BaseModel):
    """Knowledge daemon status."""
    running: bool
    batch_processor_active: bool
    refiner_active: bool
    file_watcher_active: bool
    documents_processed: int
    documents_pending: int
    documents_failed: int
    current_activity: Optional[str] = None
    uptime_seconds: float
    watch_directories: List[str]


class WatchDirectoriesRequest(BaseModel):
    """Update watched directories."""
    directories: List[str] = Field(..., min_items=1, description="List of directories to watch")

    class Config:
        json_schema_extra = {
            "example": {
                "directories": ["/path/to/docs1", "/path/to/docs2"]
            }
        }


# Concept Models

class ConceptListRequest(BaseModel):
    """List concepts request."""
    domain: Optional[str] = Field(None, description="Filter by domain")
    search_query: Optional[str] = Field(None, description="Search in concept content")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class ConceptItem(BaseModel):
    """Concept list item."""
    knowledge_id: str
    concept_name: str
    definition: str
    confidence: float
    domain: Optional[str] = None
    tags: List[str] = []
    source_document_id: Optional[str] = None


class ConceptListResponse(BaseModel):
    """List of concepts."""
    concepts: List[ConceptItem]
    total: int
    offset: int
    limit: int


class ConceptDetailResponse(BaseModel):
    """Detailed concept information."""
    knowledge_id: str
    concept_name: str
    definition: str
    confidence: float
    domain: Optional[str] = None
    tags: List[str] = []
    examples: List[str] = []
    source_document_id: Optional[str] = None
    related_concept_ids: List[str] = []
    access_count: int
    created_at: datetime


class RelatedConceptItem(BaseModel):
    """Related concept."""
    knowledge_id: str
    concept_name: str
    definition: str
    relationship_type: str
    strength: float


class RelatedConceptsResponse(BaseModel):
    """Related concepts."""
    concept_id: str
    concept_name: str
    related_concepts: List[RelatedConceptItem]
    total: int
