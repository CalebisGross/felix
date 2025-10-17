"""
Dynamic Agent Spawning System for Felix Framework

Implements Priority 2 of the enhancement plan:
- ConfidenceMonitor for team-wide confidence tracking
- ContentAnalyzer for detecting contradictions, gaps, complexity
- TeamSizeOptimizer for adaptive team sizing with resource constraints
- Enhanced spawning logic building on existing AgentFactory

This system transforms the basic assess_team_needs() into a comprehensive
adaptive agent spawning architecture.
"""

import time
import statistics
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

# Import Message and MessageType only when needed to avoid circular imports


class ConfidenceTrend(Enum):
    """Trends in team confidence over time."""
    IMPROVING = "improving"
    DECLINING = "declining" 
    STABLE = "stable"
    VOLATILE = "volatile"


class ContentIssue(Enum):
    """Types of content issues that trigger spawning."""
    CONTRADICTION = "contradiction"
    KNOWLEDGE_GAP = "knowledge_gap"
    HIGH_COMPLEXITY = "high_complexity"
    LOW_QUALITY = "low_quality"
    MISSING_DOMAIN = "missing_domain"
    INSUFFICIENT_ANALYSIS = "insufficient_analysis"


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for team performance."""
    current_average: float
    trend: ConfidenceTrend
    volatility: float
    time_window_minutes: float
    agent_type_breakdown: Dict[str, float] = field(default_factory=dict)
    position_breakdown: Dict[str, float] = field(default_factory=dict)  # helix depth ranges
    recent_samples: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, confidence)


@dataclass
class ContentAnalysis:
    """Analysis of content issues requiring new agent spawning."""
    detected_issues: Set[ContentIssue]
    complexity_score: float
    contradiction_count: int
    gap_domains: Set[str]
    quality_score: float
    analysis_depth_score: float
    suggested_agent_types: List[str]


@dataclass
class SpawningDecision:
    """Decision result for agent spawning."""
    should_spawn: bool
    agent_type: str
    spawn_parameters: Dict[str, Any]
    priority_score: float
    reasoning: str


class ConfidenceMonitor:
    """
    Monitors team-wide confidence metrics and detects when additional
    agents should be spawned to improve performance.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, 
                 volatility_threshold: float = 0.15,
                 time_window_minutes: float = 5.0):
        """
        Initialize confidence monitor.
        
        Args:
            confidence_threshold: Below this, spawn critic agents
            volatility_threshold: Above this, spawn stabilizing agents
            time_window_minutes: Time window for trend analysis
        """
        self.confidence_threshold = confidence_threshold
        self.volatility_threshold = volatility_threshold
        self.time_window_minutes = time_window_minutes
        
        # Confidence tracking
        self._confidence_history = deque(maxlen=100)  # (timestamp, confidence, agent_type, depth)
        self._agent_type_confidence: Dict[str, List[float]] = defaultdict(list)
        self._position_confidence: Dict[str, List[float]] = defaultdict(list)
        
        # Trend analysis
        self._last_trend_calculation = 0.0
        self._cached_metrics: Optional[ConfidenceMetrics] = None
    
    def record_confidence(self, message: Any) -> None:
        """
        Record confidence from agent message.
        
        Args:
            message: Message containing confidence data
        """
        content = message.content
        confidence = content.get("confidence", 0.0)
        agent_type = content.get("agent_type", "unknown")
        position_info = content.get("position_info", {})
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Record in history
        timestamp = message.timestamp
        self._confidence_history.append((timestamp, confidence, agent_type, depth_ratio))
        
        # Track by agent type
        self._agent_type_confidence[agent_type].append(confidence)
        
        # Track by position (discretize depth into ranges)
        depth_category = self._categorize_depth(depth_ratio)
        self._position_confidence[depth_category].append(confidence)
        
        # Invalidate cached metrics
        self._cached_metrics = None
    
    def _categorize_depth(self, depth_ratio: float) -> str:
        """Categorize helix depth into ranges."""
        if depth_ratio <= 0.3:
            return "shallow"
        elif depth_ratio <= 0.7:
            return "middle"
        else:
            return "deep"
    
    def get_current_metrics(self) -> ConfidenceMetrics:
        """
        Get current comprehensive confidence metrics.
        
        Returns:
            Current confidence metrics with trend analysis
        """
        current_time = time.time()
        
        # Use cached metrics if recent
        if (self._cached_metrics and 
            current_time - self._last_trend_calculation < 30.0):  # 30 second cache
            return self._cached_metrics
        
        # Calculate fresh metrics
        recent_data = self._get_recent_confidence_data(current_time)
        
        if not recent_data:
            return ConfidenceMetrics(
                current_average=0.0,
                trend=ConfidenceTrend.STABLE,
                volatility=0.0,
                time_window_minutes=self.time_window_minutes
            )
        
        # Calculate average confidence
        confidences = [conf for _, conf, _, _ in recent_data]
        current_average = statistics.mean(confidences)
        
        # Calculate volatility (standard deviation)
        volatility = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        # Calculate trend
        trend = self._calculate_trend(recent_data)
        
        # Agent type breakdown
        agent_type_breakdown = {}
        for agent_type, conf_list in self._agent_type_confidence.items():
            if conf_list:
                agent_type_breakdown[agent_type] = statistics.mean(conf_list[-10:])  # Last 10 samples
        
        # Position breakdown  
        position_breakdown = {}
        for position, conf_list in self._position_confidence.items():
            if conf_list:
                position_breakdown[position] = statistics.mean(conf_list[-10:])
        
        # Recent samples for detailed analysis
        recent_samples = [(timestamp, conf) for timestamp, conf, _, _ in recent_data]
        
        metrics = ConfidenceMetrics(
            current_average=current_average,
            trend=trend,
            volatility=volatility,
            time_window_minutes=self.time_window_minutes,
            agent_type_breakdown=agent_type_breakdown,
            position_breakdown=position_breakdown,
            recent_samples=recent_samples
        )
        
        # Cache results
        self._cached_metrics = metrics
        self._last_trend_calculation = current_time
        
        return metrics
    
    def _get_recent_confidence_data(self, current_time: float) -> List[Tuple[float, float, str, float]]:
        """Get confidence data within the time window."""
        time_cutoff = current_time - (self.time_window_minutes * 60)
        return [data for data in self._confidence_history if data[0] >= time_cutoff]
    
    def _calculate_trend(self, recent_data: List[Tuple[float, float, str, float]]) -> ConfidenceTrend:
        """Calculate confidence trend from recent data."""
        if len(recent_data) < 3:
            return ConfidenceTrend.STABLE
        
        # Extract timestamps and confidences
        timestamps = [data[0] for data in recent_data]
        confidences = [data[1] for data in recent_data]
        
        # Calculate simple linear trend
        n = len(confidences)
        sum_x = sum(timestamps)
        sum_y = sum(confidences)
        sum_xy = sum(t * c for t, c in zip(timestamps, confidences))
        sum_x2 = sum(t * t for t in timestamps)
        
        # Linear regression slope
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return ConfidenceTrend.STABLE
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Classify trend based on slope and volatility
        if abs(slope) < 0.001:  # Very small slope
            return ConfidenceTrend.STABLE
        elif slope > 0.001:
            return ConfidenceTrend.IMPROVING
        else:  # slope < -0.001
            return ConfidenceTrend.DECLINING
    
    def should_spawn_for_confidence(self) -> bool:
        """
        Determine if agents should be spawned based on confidence metrics.
        
        Returns:
            True if spawning recommended based on confidence
        """
        metrics = self.get_current_metrics()
        
        # Spawn if confidence is below threshold
        if metrics.current_average < self.confidence_threshold:
            return True
        
        # Spawn if trend is declining and confidence is not high
        if (metrics.trend == ConfidenceTrend.DECLINING and 
            metrics.current_average < 0.8):
            return True
        
        # Spawn if volatility is too high
        if metrics.volatility > self.volatility_threshold:
            return True
        
        return False
    
    def get_recommended_agent_type(self) -> str:
        """
        Get recommended agent type based on confidence analysis.
        
        Returns:
            Recommended agent type to spawn
        """
        metrics = self.get_current_metrics()
        
        # If overall confidence is low, spawn critic
        if metrics.current_average < self.confidence_threshold:
            return "critic"
        
        # If declining trend, spawn based on weakest area
        if metrics.trend == ConfidenceTrend.DECLINING:
            # Find weakest agent type
            if metrics.agent_type_breakdown:
                weakest_type = min(metrics.agent_type_breakdown.items(), key=lambda x: x[1])
                if weakest_type[1] < 0.7:
                    # Spawn complementary agent
                    if weakest_type[0] == "research":
                        return "analysis"
                    elif weakest_type[0] == "analysis":
                        return "synthesis"
                    else:
                        return "critic"
        
        # If high volatility, spawn stabilizing critic
        if metrics.volatility > self.volatility_threshold:
            return "critic"
        
        # Default fallback
        return "critic"


class ContentAnalyzer:
    """
    Analyzes message content to detect issues requiring specialized agent spawning.
    """
    
    def __init__(self):
        """Initialize content analyzer with detection patterns."""
        # Patterns for detecting different content issues
        self.contradiction_patterns = [
            r"however|but|although|despite|conversely|on the contrary",
            r"disagree|contradict|conflict|inconsistent",
            r"not accurate|incorrect|wrong|false"
        ]
        
        self.complexity_indicators = [
            r"complex|complicated|intricate|sophisticated|multifaceted",
            r"requires? (further|additional|more) (analysis|research|investigation)",
            r"unclear|ambiguous|uncertain|confusing",
            r"multiple (factors|aspects|dimensions|considerations)"
        ]
        
        self.gap_indicators = [
            r"(need|require|lack) more (information|data|research|details)",
            r"insufficient (data|information|evidence|analysis)",
            r"gaps? in|missing (information|data|analysis|coverage)",
            r"(unknown|unclear|unspecified|undefined)"
        ]
        
        self.quality_indicators = [
            r"(preliminary|draft|initial|rough|basic) (analysis|research|findings)",
            r"needs? (improvement|refinement|enhancement|development)",
            r"(low|poor|insufficient) quality",
            r"(incomplete|partial|limited) (analysis|coverage|scope)"
        ]
        
        # Domain keywords for gap detection
        self.domain_keywords = {
            "technical": ["algorithm", "implementation", "code", "system", "architecture"],
            "business": ["market", "revenue", "cost", "strategy", "competition"],
            "scientific": ["research", "study", "experiment", "hypothesis", "methodology"],
            "creative": ["design", "aesthetic", "artistic", "creative", "visual"],
            "analytical": ["analysis", "statistics", "metrics", "data", "measurement"]
        }
    
    def analyze_content(self, messages: List[Any]) -> ContentAnalysis:
        """
        Analyze messages for content issues requiring agent spawning.
        
        Args:
            messages: List of recent messages to analyze
            
        Returns:
            Comprehensive content analysis
        """
        if not messages:
            return ContentAnalysis(
                detected_issues=set(),
                complexity_score=0.0,
                contradiction_count=0,
                gap_domains=set(),
                quality_score=1.0,
                analysis_depth_score=0.0,
                suggested_agent_types=[]
            )
        
        # Combine all message content for analysis
        combined_content = ""
        for msg in messages:
            content = msg.content.get("result", "")
            if isinstance(content, str):
                combined_content += content + " "
        
        combined_content = combined_content.lower()
        
        # Detect issues
        detected_issues = set()
        
        # Check for contradictions
        contradiction_count = 0
        for pattern in self.contradiction_patterns:
            contradiction_count += len(re.findall(pattern, combined_content, re.IGNORECASE))
        
        if contradiction_count > 0:
            detected_issues.add(ContentIssue.CONTRADICTION)
        
        # Check for complexity indicators
        complexity_matches = 0
        for pattern in self.complexity_indicators:
            complexity_matches += len(re.findall(pattern, combined_content, re.IGNORECASE))
        
        complexity_score = min(1.0, complexity_matches / 10.0)  # Normalize to 0-1
        if complexity_score > 0.3:
            detected_issues.add(ContentIssue.HIGH_COMPLEXITY)
        
        # Check for knowledge gaps
        gap_matches = 0
        for pattern in self.gap_indicators:
            gap_matches += len(re.findall(pattern, combined_content, re.IGNORECASE))
        
        if gap_matches > 0:
            detected_issues.add(ContentIssue.KNOWLEDGE_GAP)
        
        # Check for quality issues
        quality_matches = 0
        for pattern in self.quality_indicators:
            quality_matches += len(re.findall(pattern, combined_content, re.IGNORECASE))
        
        quality_score = max(0.0, 1.0 - (quality_matches / 5.0))  # Invert and normalize
        if quality_score < 0.7:
            detected_issues.add(ContentIssue.LOW_QUALITY)
        
        # Detect missing domains
        covered_domains = set()
        gap_domains = set()
        
        for domain, keywords in self.domain_keywords.items():
            domain_coverage = sum(1 for keyword in keywords 
                                if keyword in combined_content)
            if domain_coverage > 0:
                covered_domains.add(domain)
        
        # If only one domain covered, others are gaps
        if len(covered_domains) == 1:
            gap_domains = set(self.domain_keywords.keys()) - covered_domains
            detected_issues.add(ContentIssue.MISSING_DOMAIN)
        
        # Calculate analysis depth score
        analysis_depth_indicators = [
            "because", "therefore", "analysis shows", "data indicates",
            "research suggests", "evidence supports", "conclusion",
            "findings", "methodology", "approach", "framework"
        ]
        
        depth_matches = sum(1 for indicator in analysis_depth_indicators
                           if indicator in combined_content)
        analysis_depth_score = min(1.0, depth_matches / 8.0)
        
        if analysis_depth_score < 0.3:
            detected_issues.add(ContentIssue.INSUFFICIENT_ANALYSIS)
        
        # Generate agent type suggestions
        suggested_agent_types = self._suggest_agent_types(detected_issues, gap_domains)
        
        return ContentAnalysis(
            detected_issues=detected_issues,
            complexity_score=complexity_score,
            contradiction_count=contradiction_count,
            gap_domains=gap_domains,
            quality_score=quality_score,
            analysis_depth_score=analysis_depth_score,
            suggested_agent_types=suggested_agent_types
        )
    
    def _suggest_agent_types(self, issues: Set[ContentIssue], gap_domains: Set[str]) -> List[str]:
        """Suggest agent types based on detected issues."""
        suggestions = []
        
        if ContentIssue.CONTRADICTION in issues:
            suggestions.append("critic")
        
        if ContentIssue.KNOWLEDGE_GAP in issues or ContentIssue.MISSING_DOMAIN in issues:
            suggestions.append("research")
        
        if ContentIssue.HIGH_COMPLEXITY in issues or ContentIssue.INSUFFICIENT_ANALYSIS in issues:
            suggestions.append("analysis")
        
        if ContentIssue.LOW_QUALITY in issues:
            suggestions.append("critic")
            suggestions.append("synthesis")  # For quality improvement
        
        # Domain-specific suggestions
        if gap_domains:
            if "technical" in gap_domains or "scientific" in gap_domains:
                suggestions.append("research")
            if "analytical" in gap_domains:
                suggestions.append("analysis")
        
        return list(set(suggestions))  # Remove duplicates


class TeamSizeOptimizer:
    """
    Optimizes team size based on task complexity, resource constraints,
    and performance feedback.
    """
    
    def __init__(self, max_agents: int = 15, token_budget_limit: int = 10000,
                 performance_weight: float = 0.4, efficiency_weight: float = 0.6):
        """
        Initialize team size optimizer.
        
        Args:
            max_agents: Maximum allowed agents
            token_budget_limit: Total token budget limit
            performance_weight: Weight for performance in optimization
            efficiency_weight: Weight for efficiency in optimization  
        """
        self.max_agents = max_agents
        self.token_budget_limit = token_budget_limit
        self.performance_weight = performance_weight
        self.efficiency_weight = efficiency_weight
        
        # Historical performance tracking
        self._team_size_performance: Dict[int, List[float]] = defaultdict(list)
        self._team_size_efficiency: Dict[int, List[float]] = defaultdict(list)
        self._current_team_size = 0
        self._current_token_usage = 0
    
    def update_current_state(self, team_size: int, token_usage: int) -> None:
        """Update current team state for optimization calculations."""
        self._current_team_size = team_size
        self._current_token_usage = token_usage
    
    def record_performance(self, team_size: int, performance_score: float, 
                          efficiency_score: float) -> None:
        """Record team performance for the given size."""
        self._team_size_performance[team_size].append(performance_score)
        self._team_size_efficiency[team_size].append(efficiency_score)
    
    def get_optimal_team_size(self, task_complexity: float, 
                            current_confidence: float) -> int:
        """
        Calculate optimal team size based on multiple factors.
        
        Args:
            task_complexity: Complexity score (0.0 to 1.0)
            current_confidence: Current team confidence (0.0 to 1.0)
            
        Returns:
            Recommended optimal team size
        """
        # Base team size from complexity (3-10 agents)
        base_size = max(3, min(10, int(3 + task_complexity * 7)))
        
        # Adjust for confidence (low confidence = more agents)
        confidence_adjustment = max(-2, min(3, int((0.7 - current_confidence) * 5)))
        adjusted_size = base_size + confidence_adjustment
        
        # Consider resource constraints
        estimated_tokens_per_agent = 800  # Average from existing agents
        max_affordable_agents = self.token_budget_limit // estimated_tokens_per_agent
        resource_constrained_size = min(adjusted_size, max_affordable_agents)
        
        # Apply hard limit
        optimal_size = min(resource_constrained_size, self.max_agents)
        
        # Historical optimization
        if self._team_size_performance:
            historical_optimal = self._get_historical_optimal()
            # Blend current calculation with historical data
            optimal_size = int(0.7 * optimal_size + 0.3 * historical_optimal)
        
        return max(1, optimal_size)  # Ensure at least 1 agent
    
    def _get_historical_optimal(self) -> int:
        """Get optimal size based on historical performance."""
        best_size = 3
        best_score = 0.0
        
        for size, perf_scores in self._team_size_performance.items():
            if not perf_scores or size not in self._team_size_efficiency:
                continue
                
            avg_performance = statistics.mean(perf_scores)
            avg_efficiency = statistics.mean(self._team_size_efficiency[size])
            
            # Weighted score
            weighted_score = (self.performance_weight * avg_performance + 
                            self.efficiency_weight * avg_efficiency)
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_size = size
        
        return best_size
    
    def should_expand_team(self, current_size: int, task_complexity: float,
                          confidence_metrics: ConfidenceMetrics) -> bool:
        """
        Determine if team should be expanded.
        
        Args:
            current_size: Current team size
            task_complexity: Task complexity score
            confidence_metrics: Current confidence metrics
            
        Returns:
            True if team expansion recommended
        """
        optimal_size = self.get_optimal_team_size(task_complexity, 
                                                confidence_metrics.current_average)
        
        # Don't expand if at or above optimal
        if current_size >= optimal_size:
            return False
        
        # Don't expand if resource constrained
        estimated_new_tokens = 800  # Per new agent
        if self._current_token_usage + estimated_new_tokens > self.token_budget_limit:
            return False
        
        # Expand if confidence is declining and we're under optimal
        if (confidence_metrics.trend == ConfidenceTrend.DECLINING and
            current_size < optimal_size):
            return True
        
        # Expand if confidence is low and volatile
        if (confidence_metrics.current_average < 0.6 and
            confidence_metrics.volatility > 0.2):
            return True
        
        return False
    
    def get_resource_budget_for_new_agent(self, agent_type: str) -> int:
        """Get token budget allocation for new agent based on type and constraints."""
        # Base budgets by agent type
        base_budgets = {
            "research": 1000,
            "analysis": 800, 
            "synthesis": 1200,
            "critic": 600
        }
        
        base_budget = base_budgets.get(agent_type, 800)
        
        # Scale down if resource constrained
        remaining_budget = self.token_budget_limit - self._current_token_usage
        if remaining_budget < base_budget:
            return max(200, int(remaining_budget * 0.8))  # Leave some buffer
        
        return base_budget


class DynamicSpawning:
    """
    Main coordinator for dynamic agent spawning combining all monitoring systems.
    
    Integrates ConfidenceMonitor, ContentAnalyzer, and TeamSizeOptimizer
    to make intelligent spawning decisions.
    """
    
    def __init__(self, agent_factory, confidence_threshold: float = 0.7,
                 max_agents: int = 15, token_budget_limit: int = 10000):
        """
        Initialize dynamic spawning system.
        
        Args:
            agent_factory: AgentFactory instance for creating agents
            confidence_threshold: Confidence threshold for spawning
            max_agents: Maximum allowed agents
            token_budget_limit: Total token budget limit
        """
        self.agent_factory = agent_factory
        
        # Initialize monitoring systems
        self.confidence_monitor = ConfidenceMonitor(confidence_threshold=confidence_threshold)
        self.content_analyzer = ContentAnalyzer()
        self.team_optimizer = TeamSizeOptimizer(max_agents=max_agents, 
                                              token_budget_limit=token_budget_limit)
        
        # State tracking
        self._last_analysis_time = 0.0
        self._spawning_history: List[SpawningDecision] = []
    
    def analyze_and_spawn(self, processed_messages: List[Any], 
                         current_agents: List[Any], current_time: float) -> List[Any]:
        """
        Main method to analyze team needs and spawn agents if necessary.
        
        Args:
            processed_messages: Recent processed messages
            current_agents: List of current active agents
            current_time: Current simulation time
            
        Returns:
            List of newly spawned agents
        """
        # Update monitors with recent data
        for msg in processed_messages:
            if msg.timestamp > self._last_analysis_time:
                self.confidence_monitor.record_confidence(msg)
        
        # Get current metrics
        confidence_metrics = self.confidence_monitor.get_current_metrics()
        content_analysis = self.content_analyzer.analyze_content(processed_messages[-10:])  # Last 10 messages
        
        # Update team optimizer state
        current_token_usage = sum(getattr(agent, 'max_tokens', 800) for agent in current_agents)
        self.team_optimizer.update_current_state(len(current_agents), current_token_usage)
        
        # Make spawning decisions
        spawning_decisions = self._make_spawning_decisions(
            confidence_metrics, content_analysis, current_agents, current_time
        )
        
        # Execute spawning decisions
        new_agents = []
        for decision in spawning_decisions:
            if decision.should_spawn:
                try:
                    new_agent = self._spawn_agent(decision)
                    if new_agent:
                        new_agents.append(new_agent)
                        self._spawning_history.append(decision)
                except Exception as e:
                    # Log error but continue with other spawns
                    print(f"Failed to spawn {decision.agent_type} agent: {e}")
        
        self._last_analysis_time = current_time
        return new_agents
    
    def _make_spawning_decisions(self, confidence_metrics: ConfidenceMetrics,
                                content_analysis: ContentAnalysis,
                                current_agents: List[Any], current_time: float) -> List[SpawningDecision]:
        """Make intelligent spawning decisions based on all available data."""
        decisions = []
        
        # Check if team expansion is warranted
        task_complexity = content_analysis.complexity_score
        should_expand = self.team_optimizer.should_expand_team(
            len(current_agents), task_complexity, confidence_metrics
        )
        
        if not should_expand:
            return decisions  # No spawning needed
        
        # Priority 1: Confidence-based spawning
        if confidence_metrics.current_average < 0.7:
            agent_type = self.confidence_monitor.get_recommended_agent_type()
            priority_score = 1.0 - confidence_metrics.current_average  # Higher priority for lower confidence
            
            decisions.append(SpawningDecision(
                should_spawn=True,
                agent_type=agent_type,
                spawn_parameters={
                    "spawn_time_range": (current_time + 0.05, current_time + 0.2),
                    "max_tokens": self.team_optimizer.get_resource_budget_for_new_agent(agent_type)
                },
                priority_score=priority_score,
                reasoning=f"Low confidence ({confidence_metrics.current_average:.2f}) triggered {agent_type} spawn"
            ))
        
        # Priority 2: Content-based spawning
        for suggested_type in content_analysis.suggested_agent_types:
            if len(decisions) >= 2:  # Limit concurrent spawns
                break
                
            # Calculate priority based on issue severity
            priority_score = 0.5  # Base priority
            if ContentIssue.CONTRADICTION in content_analysis.detected_issues:
                priority_score += 0.3
            if ContentIssue.LOW_QUALITY in content_analysis.detected_issues:
                priority_score += 0.2
            if content_analysis.complexity_score > 0.7:
                priority_score += 0.2
            
            decisions.append(SpawningDecision(
                should_spawn=True,
                agent_type=suggested_type,
                spawn_parameters={
                    "spawn_time_range": (current_time + 0.1, current_time + 0.3),
                    "max_tokens": self.team_optimizer.get_resource_budget_for_new_agent(suggested_type),
                    "specialized_focus": self._get_specialized_focus(content_analysis, suggested_type)
                },
                priority_score=priority_score,
                reasoning=f"Content analysis detected {len(content_analysis.detected_issues)} issues requiring {suggested_type} agent"
            ))
        
        # Sort by priority and return top decisions
        decisions.sort(key=lambda d: d.priority_score, reverse=True)
        return decisions[:2]  # Maximum 2 spawns per analysis cycle
    
    def _get_specialized_focus(self, content_analysis: ContentAnalysis, agent_type: str) -> str:
        """Get specialized focus for agent based on content analysis."""
        if agent_type == "critic" and ContentIssue.CONTRADICTION in content_analysis.detected_issues:
            return "contradiction_resolution"
        elif agent_type == "research" and content_analysis.gap_domains:
            return list(content_analysis.gap_domains)[0]  # Focus on first gap domain
        elif agent_type == "analysis" and content_analysis.complexity_score > 0.7:
            return "complexity_reduction"
        elif agent_type == "synthesis" and ContentIssue.LOW_QUALITY in content_analysis.detected_issues:
            return "quality_improvement"
        return "general"
    
    def _spawn_agent(self, decision: SpawningDecision):
        """Spawn agent based on decision parameters."""
        spawn_params = decision.spawn_parameters
        agent_type = decision.agent_type
        
        # Extract spawn parameters
        spawn_time_range = spawn_params.get("spawn_time_range", (0.1, 0.3))
        max_tokens = spawn_params.get("max_tokens", 800)
        specialized_focus = spawn_params.get("specialized_focus", "general")
        
        # Create agent based on type
        if agent_type == "research":
            return self.agent_factory.create_research_agent(
                domain=specialized_focus,
                spawn_time_range=spawn_time_range
            )
        elif agent_type == "analysis":
            return self.agent_factory.create_analysis_agent(
                analysis_type=specialized_focus,
                spawn_time_range=spawn_time_range
            )
        elif agent_type == "critic":
            return self.agent_factory.create_critic_agent(
                review_focus=specialized_focus,
                spawn_time_range=spawn_time_range
            )
        elif agent_type == "synthesis":
            return self.agent_factory.create_synthesis_agent(
                output_format=specialized_focus,
                spawn_time_range=spawn_time_range
            )
        
        return None
    
    def get_spawning_summary(self) -> Dict[str, Any]:
        """Get summary of spawning activity for analysis."""
        return {
            "total_spawns": len(self._spawning_history),
            "spawns_by_type": {
                agent_type: sum(1 for d in self._spawning_history if d.agent_type == agent_type)
                for agent_type in ["research", "analysis", "critic", "synthesis"]
            },
            "average_priority": statistics.mean([d.priority_score for d in self._spawning_history]) if self._spawning_history else 0.0,
            "spawning_reasons": [d.reasoning for d in self._spawning_history[-5:]]  # Last 5 reasons
        }
