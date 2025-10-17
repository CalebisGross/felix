"""
Prompt Optimization Pipeline for Felix Framework

This module implements Priority 3 of the enhancement plan, providing:
- PromptMetricsTracker: Performance tracking across different contexts
- PromptTester: A/B testing framework for systematic prompt improvement
- FailureAnalyzer: Learning from truncated/low-quality outputs  
- PromptOptimizer: Coordinator integrating all optimization systems

The system uses statistical analysis and context-aware optimization to
continuously improve prompt effectiveness across the Felix Framework.
"""

import time
import json
import statistics
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
import uuid


class PromptContext(Enum):
    """Context types for prompt optimization."""
    RESEARCH_EARLY = "research_early"      # Beginning of helix, high creativity
    RESEARCH_MID = "research_mid"          # Mid-helix research, balanced
    ANALYSIS_MID = "analysis_mid"          # Analysis phase, structured
    ANALYSIS_LATE = "analysis_late"        # Deep analysis, focused
    SYNTHESIS_LATE = "synthesis_late"      # Synthesis phase, precise
    SYNTHESIS_FINAL = "synthesis_final"    # Final synthesis, quality
    CRITIC_ANY = "critic_any"              # Critical review, any position
    GENERAL = "general"                    # General purpose prompts


@dataclass
class PromptMetrics:
    """Metrics for a single prompt execution."""
    output_quality: float              # 0.0-1.0 quality score
    confidence: float                  # Agent's confidence in output
    completion_time: float             # Time to generate response
    token_efficiency: float            # Quality per token used
    truncation_occurred: bool          # Was output truncated?
    context: PromptContext             # Context where prompt was used
    timestamp: float = field(default_factory=time.time)
    
    def overall_score(self) -> float:
        """Calculate overall prompt performance score."""
        # Weight different metrics based on importance
        quality_weight = 0.4
        confidence_weight = 0.3
        efficiency_weight = 0.2
        speed_weight = 0.1
        
        # Normalize completion time (assume good time is < 10s)
        time_score = max(0, 1 - (self.completion_time / 10.0))
        
        # Penalty for truncation
        truncation_penalty = 0.2 if self.truncation_occurred else 0.0
        
        score = (
            self.output_quality * quality_weight +
            self.confidence * confidence_weight +
            self.token_efficiency * efficiency_weight +
            time_score * speed_weight
        ) - truncation_penalty
        
        return max(0.0, min(1.0, score))


class PromptMetricsTracker:
    """
    Tracks prompt performance across different contexts and variations.
    
    Maintains historical performance data and provides statistical analysis
    of prompt effectiveness across different agent types and helix positions.
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.context_performance: Dict[PromptContext, List[float]] = defaultdict(list)
        
    def record_metrics(self, prompt_id: str, metrics: PromptMetrics):
        """Record metrics for a specific prompt."""
        self.metrics_history[prompt_id].append(metrics)
        self.context_performance[metrics.context].append(metrics.overall_score())
        
    def get_prompt_performance(self, prompt_id: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for a specific prompt."""
        if prompt_id not in self.metrics_history:
            return None
            
        metrics_list = list(self.metrics_history[prompt_id])
        if not metrics_list:
            return None
            
        scores = [m.overall_score() for m in metrics_list]
        quality_scores = [m.output_quality for m in metrics_list]
        confidence_scores = [m.confidence for m in metrics_list]
        
        return {
            'mean_score': statistics.mean(scores),
            'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'mean_quality': statistics.mean(quality_scores),
            'mean_confidence': statistics.mean(confidence_scores),
            'sample_size': len(scores),
            'truncation_rate': sum(1 for m in metrics_list if m.truncation_occurred) / len(metrics_list)
        }
        
    def get_context_performance(self, context: PromptContext) -> Optional[Dict[str, float]]:
        """Get performance statistics for a specific context."""
        scores = self.context_performance[context]
        if not scores:
            return None
            
        return {
            'mean_score': statistics.mean(scores),
            'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'sample_size': len(scores)
        }
        
    def get_best_prompts(self, context: Optional[PromptContext] = None, limit: int = 5) -> List[Tuple[str, float]]:
        """Get the best performing prompts, optionally filtered by context."""
        prompt_scores = []
        
        for prompt_id, metrics_deque in self.metrics_history.items():
            metrics_list = list(metrics_deque)
            if not metrics_list:
                continue
                
            # Filter by context if specified
            if context:
                metrics_list = [m for m in metrics_list if m.context == context]
                if not metrics_list:
                    continue
                    
            mean_score = statistics.mean([m.overall_score() for m in metrics_list])
            prompt_scores.append((prompt_id, mean_score))
            
        # Sort by score descending and return top results
        prompt_scores.sort(key=lambda x: x[1], reverse=True)
        return prompt_scores[:limit]


@dataclass
class PromptVariation:
    """A variation of a prompt for A/B testing."""
    prompt_text: str
    variation_id: str
    parent_prompt_id: Optional[str] = None
    generation_method: str = "manual"  # manual, systematic, failure_learned
    test_results: List[PromptMetrics] = field(default_factory=list)
    
    def add_result(self, metrics: PromptMetrics):
        """Add test result for this variation."""
        self.test_results.append(metrics)
        
    def get_performance(self) -> Optional[float]:
        """Get mean performance score for this variation."""
        if not self.test_results:
            return None
        return statistics.mean([m.overall_score() for m in self.test_results])


class PromptTester:
    """
    A/B testing framework for systematic prompt improvement.
    
    Manages prompt variations, statistical testing, and early stopping
    decisions based on statistical significance.
    """
    
    def __init__(self, min_samples_per_variant: int = 10, confidence_level: float = 0.95):
        self.min_samples_per_variant = min_samples_per_variant
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level  # For significance testing
        
        self.active_tests: Dict[str, List[PromptVariation]] = {}
        self.completed_tests: Dict[str, Dict[str, Any]] = {}
        
    def create_test(self, test_id: str, base_prompt: str, variations: List[str]) -> str:
        """Create a new A/B test with prompt variations."""
        prompt_variations = []
        
        # Add base prompt as control
        base_variation = PromptVariation(
            prompt_text=base_prompt,
            variation_id=f"{test_id}_control",
            generation_method="control"
        )
        prompt_variations.append(base_variation)
        
        # Add test variations
        for i, variation_text in enumerate(variations):
            variation = PromptVariation(
                prompt_text=variation_text,
                variation_id=f"{test_id}_var_{i}",
                parent_prompt_id=f"{test_id}_control",
                generation_method="systematic"
            )
            prompt_variations.append(variation)
            
        self.active_tests[test_id] = prompt_variations
        return test_id
        
    def add_test_result(self, test_id: str, variation_id: str, metrics: PromptMetrics) -> bool:
        """Add a test result and return True if test should continue."""
        if test_id not in self.active_tests:
            return False
            
        # Find variation and add result
        for variation in self.active_tests[test_id]:
            if variation.variation_id == variation_id:
                variation.add_result(metrics)
                break
        else:
            return False
            
        # Check if we should stop early
        return self._should_continue_test(test_id)
        
    def _should_continue_test(self, test_id: str) -> bool:
        """Determine if test should continue based on statistical significance."""
        variations = self.active_tests[test_id]
        
        # Need minimum samples for all variations
        min_samples = min(len(v.test_results) for v in variations)
        if min_samples < self.min_samples_per_variant:
            return True
            
        # Simple t-test approximation for early stopping
        # Compare best variation against control
        control_performance = [v for v in variations if "control" in v.variation_id][0].get_performance()
        if control_performance is None:
            return True
            
        best_variant = max(variations, key=lambda v: v.get_performance() or 0)
        best_performance = best_variant.get_performance()
        
        if best_performance is None or best_performance <= control_performance:
            return min_samples < 50  # Continue until more samples
            
        # Simplified significance test
        control_scores = [m.overall_score() for m in variations[0].test_results]
        best_scores = [m.overall_score() for m in best_variant.test_results]
        
        if len(control_scores) > 1 and len(best_scores) > 1:
            # Simple effect size calculation
            effect_size = abs(statistics.mean(best_scores) - statistics.mean(control_scores))
            pooled_std = (statistics.stdev(control_scores) + statistics.stdev(best_scores)) / 2
            
            if pooled_std > 0:
                cohen_d = effect_size / pooled_std
                # Stop if large effect size (Cohen's d > 0.8) and sufficient samples
                if cohen_d > 0.8 and min_samples >= self.min_samples_per_variant:
                    self._complete_test(test_id)
                    return False
                    
        # Continue testing up to reasonable limit
        return min_samples < 100
        
    def _complete_test(self, test_id: str):
        """Complete a test and store results."""
        variations = self.active_tests[test_id]
        
        results = {
            'test_id': test_id,
            'completion_time': time.time(),
            'variations': []
        }
        
        for variation in variations:
            perf = variation.get_performance()
            results['variations'].append({
                'variation_id': variation.variation_id,
                'prompt_text': variation.prompt_text,
                'sample_size': len(variation.test_results),
                'mean_performance': perf,
                'generation_method': variation.generation_method
            })
            
        # Find winner
        best_variation = max(variations, key=lambda v: v.get_performance() or 0)
        results['winner'] = {
            'variation_id': best_variation.variation_id,
            'performance': best_variation.get_performance(),
            'improvement': (best_variation.get_performance() or 0) - (variations[0].get_performance() or 0)
        }
        
        self.completed_tests[test_id] = results
        del self.active_tests[test_id]
        
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an active test."""
        if test_id in self.completed_tests:
            return {'status': 'completed', 'results': self.completed_tests[test_id]}
            
        if test_id not in self.active_tests:
            return None
            
        variations = self.active_tests[test_id]
        status = {
            'status': 'active',
            'variations': []
        }
        
        for variation in variations:
            perf = variation.get_performance()
            status['variations'].append({
                'variation_id': variation.variation_id,
                'sample_size': len(variation.test_results),
                'mean_performance': perf
            })
            
        return status


@dataclass
class FailurePattern:
    """Pattern detected from failed prompt executions."""
    pattern_type: str  # 'truncation', 'low_confidence', 'low_quality'
    context: PromptContext
    frequency: int = 0
    avg_failure_score: float = 0.0
    suggested_fix: str = ""
    examples: List[str] = field(default_factory=list)


class FailureAnalyzer:
    """
    Analyzes failed prompt executions to learn improvement strategies.
    
    Identifies patterns in truncated outputs, low confidence responses,
    and poor quality results to suggest prompt improvements.
    """
    
    def __init__(self, failure_threshold: float = 0.5):
        self.failure_threshold = failure_threshold
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.failure_history: List[Tuple[str, PromptMetrics]] = []
        
    def analyze_failure(self, prompt_id: str, metrics: PromptMetrics):
        """Analyze a failed prompt execution."""
        if metrics.overall_score() >= self.failure_threshold:
            return  # Not a failure
            
        self.failure_history.append((prompt_id, metrics))
        
        # Identify failure patterns
        if metrics.truncation_occurred:
            self._record_pattern("truncation", metrics.context, prompt_id)
            
        if metrics.confidence < 0.3:
            self._record_pattern("low_confidence", metrics.context, prompt_id)
            
        if metrics.output_quality < 0.3:
            self._record_pattern("low_quality", metrics.context, prompt_id)
            
    def _record_pattern(self, pattern_type: str, context: PromptContext, prompt_id: str):
        """Record a failure pattern."""
        pattern_key = f"{pattern_type}_{context.value}"
        
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = FailurePattern(
                pattern_type=pattern_type,
                context=context,
                suggested_fix=self._generate_suggested_fix(pattern_type, context)
            )
            
        pattern = self.failure_patterns[pattern_key]
        pattern.frequency += 1
        pattern.examples.append(prompt_id)
        
        # Keep only recent examples
        if len(pattern.examples) > 10:
            pattern.examples = pattern.examples[-10:]
            
    def _generate_suggested_fix(self, pattern_type: str, context: PromptContext) -> str:
        """Generate suggested fix for a failure pattern."""
        fixes = {
            'truncation': {
                PromptContext.RESEARCH_EARLY: "Use more focused questions, request structured responses",
                PromptContext.ANALYSIS_MID: "Break complex analysis into smaller chunks",
                PromptContext.SYNTHESIS_LATE: "Request executive summary first, then details",
                'default': "Reduce prompt complexity, request shorter responses"
            },
            'low_confidence': {
                PromptContext.RESEARCH_EARLY: "Provide more context, reduce scope",
                PromptContext.ANALYSIS_MID: "Include examples, clarify expectations",
                PromptContext.SYNTHESIS_LATE: "Reference previous work, provide templates",
                'default': "Add examples, clarify requirements"
            },
            'low_quality': {
                PromptContext.RESEARCH_EARLY: "Add quality criteria, request citations",
                PromptContext.ANALYSIS_MID: "Specify analysis framework, add evaluation criteria",
                PromptContext.SYNTHESIS_LATE: "Request peer review, add quality checklist",
                'default': "Add quality requirements, provide evaluation criteria"
            }
        }
        
        pattern_fixes = fixes.get(pattern_type, {})
        return pattern_fixes.get(context, pattern_fixes.get('default', "No specific recommendation"))
        
    def get_improvement_suggestions(self, context: PromptContext) -> List[Dict[str, Any]]:
        """Get improvement suggestions for a specific context."""
        suggestions = []
        
        for pattern_key, pattern in self.failure_patterns.items():
            if pattern.context == context and pattern.frequency >= 3:
                suggestions.append({
                    'pattern_type': pattern.pattern_type,
                    'frequency': pattern.frequency,
                    'suggested_fix': pattern.suggested_fix,
                    'severity': 'high' if pattern.frequency > 10 else 'medium'
                })
                
        # Sort by frequency (most common first)
        suggestions.sort(key=lambda x: x['frequency'], reverse=True)
        return suggestions
        
    def generate_improved_prompts(self, base_prompt: str, context: PromptContext) -> List[str]:
        """Generate improved prompt variations based on failure analysis."""
        suggestions = self.get_improvement_suggestions(context)
        if not suggestions:
            return []
            
        improved_prompts = []
        
        for suggestion in suggestions[:3]:  # Top 3 suggestions
            if suggestion['pattern_type'] == 'truncation':
                # Make prompts more concise
                improved = base_prompt + "\n\nPlease provide a concise, structured response focusing on key points."
                improved_prompts.append(improved)
                
            elif suggestion['pattern_type'] == 'low_confidence':
                # Add confidence boosting elements
                improved = base_prompt + "\n\nTake your time and provide your best analysis. Include confidence level in your response."
                improved_prompts.append(improved)
                
            elif suggestion['pattern_type'] == 'low_quality':
                # Add quality requirements
                improved = base_prompt + "\n\nEnsure high quality by: 1) Double-checking facts, 2) Using clear structure, 3) Providing specific examples."
                improved_prompts.append(improved)
                
        return improved_prompts


class PromptOptimizer:
    """
    Main coordinator for the prompt optimization system.
    
    Integrates metrics tracking, A/B testing, and failure analysis to
    continuously improve prompt effectiveness across the Felix Framework.
    """
    
    def __init__(self):
        self.metrics_tracker = PromptMetricsTracker()
        self.prompt_tester = PromptTester()
        self.failure_analyzer = FailureAnalyzer()
        
        # Track active optimizations
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        
    def record_prompt_execution(self, prompt_id: str, prompt_text: str, 
                              metrics: PromptMetrics) -> Dict[str, Any]:
        """Record a prompt execution and trigger optimization if needed."""
        # Record metrics
        self.metrics_tracker.record_metrics(prompt_id, metrics)
        
        # Analyze failures
        self.failure_analyzer.analyze_failure(prompt_id, metrics)
        
        # Check if this is part of an active test
        for test_id, variations in self.prompt_tester.active_tests.items():
            for variation in variations:
                if variation.variation_id == prompt_id:
                    should_continue = self.prompt_tester.add_test_result(test_id, prompt_id, metrics)
                    if not should_continue:
                        return {'status': 'test_completed', 'test_id': test_id}
                    break
                    
        # Check if we should start optimization
        performance = self.metrics_tracker.get_prompt_performance(prompt_id)
        if performance and performance['sample_size'] >= 5:
            if performance['mean_score'] < 0.6 or performance['truncation_rate'] > 0.3:
                self._trigger_optimization(prompt_id, prompt_text, metrics.context)
                
        return {'status': 'recorded', 'prompt_performance': performance}
    
    def optimize_prompt(self, base_prompt: str, context: Dict[str, Any]) -> 'OptimizedPrompt':
        """Optimize a prompt for better performance.
        
        Args:
            base_prompt: The original prompt text to optimize
            context: Context dictionary with optimization parameters
            
        Returns:
            OptimizedPrompt object with improved prompt text and metadata
        """
        prompt_id = f"optimized_{str(uuid.uuid4())[:8]}"
        
        # Basic optimization - add structure and clarity
        optimized_text = base_prompt
        if "domain" in context and context["domain"] == "technical":
            optimized_text += "\n\nProvide a structured, technical response with clear examples."
        elif "task_type" in context and context["task_type"] == "analysis":
            optimized_text += "\n\nPlease analyze systematically and provide evidence-based conclusions."
        else:
            optimized_text += "\n\nProvide a clear, well-structured response."
            
        return OptimizedPrompt(
            prompt_id=prompt_id,
            prompt_text=optimized_text,
            context=context,
            optimization_applied=True
        )
    
    def record_prompt_performance(self, prompt_id: str, success_rate: float, 
                                quality_metrics: Dict[str, float], context: Dict[str, Any]):
        """Record performance metrics for a prompt.
        
        Args:
            prompt_id: Unique identifier for the prompt
            success_rate: Success rate (0.0-1.0)
            quality_metrics: Dictionary of quality metrics
            context: Context information
        """
        # Create synthetic PromptMetrics from the provided data
        metrics = PromptMetrics(
            output_quality=quality_metrics.get('coherence', success_rate),
            confidence=success_rate,
            completion_time=2.0,  # Default reasonable time
            token_efficiency=success_rate * 0.8,
            truncation_occurred=False,
            context=PromptContext.GENERAL  # Default context
        )
        
        self.metrics_tracker.record_metrics(prompt_id, metrics)
    
    def get_prompt_performance(self, prompt_id: str) -> List[Dict[str, Any]]:
        """Get performance history for a prompt.
        
        Args:
            prompt_id: The prompt identifier
            
        Returns:
            List of performance records
        """
        performance = self.metrics_tracker.get_prompt_performance(prompt_id)
        if performance:
            return [performance]  # Return as list for compatibility
        return []
        
    def _trigger_optimization(self, prompt_id: str, prompt_text: str, context: PromptContext):
        """Trigger optimization for a poorly performing prompt."""
        if prompt_id in self.active_optimizations:
            return  # Already optimizing
            
        # Generate improved variations
        improved_prompts = self.failure_analyzer.generate_improved_prompts(prompt_text, context)
        
        if improved_prompts:
            test_id = f"{prompt_id}_optimization_{int(time.time())}"
            self.prompt_tester.create_test(test_id, prompt_text, improved_prompts)
            
            self.active_optimizations[prompt_id] = {
                'test_id': test_id,
                'started_at': time.time(),
                'context': context,
                'original_prompt': prompt_text
            }
            
    def get_optimization_recommendations(self, context: PromptContext) -> Dict[str, Any]:
        """Get optimization recommendations for a specific context."""
        # Get top performing prompts
        best_prompts = self.metrics_tracker.get_best_prompts(context, limit=3)
        
        # Get failure analysis
        improvement_suggestions = self.failure_analyzer.get_improvement_suggestions(context)
        
        # Get context performance
        context_perf = self.metrics_tracker.get_context_performance(context)
        
        return {
            'context': context.value,
            'context_performance': context_perf,
            'best_prompts': best_prompts,
            'improvement_suggestions': improvement_suggestions,
            'active_tests': len([t for t in self.prompt_tester.active_tests.values() 
                               if any(v.test_results and v.test_results[-1].context == context 
                                     for v in t)])
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and performance."""
        total_prompts = len(self.metrics_tracker.metrics_history)
        active_tests = len(self.prompt_tester.active_tests)
        completed_tests = len(self.prompt_tester.completed_tests)
        failure_patterns = len(self.failure_analyzer.failure_patterns)
        
        # Calculate overall system performance
        all_scores = []
        for context in PromptContext:
            context_perf = self.metrics_tracker.get_context_performance(context)
            if context_perf:
                all_scores.append(context_perf['mean_score'])
                
        overall_performance = statistics.mean(all_scores) if all_scores else 0.0
        
        return {
            'total_prompts_tracked': total_prompts,
            'active_tests': active_tests,
            'completed_tests': completed_tests,
            'failure_patterns_identified': failure_patterns,
            'overall_performance': overall_performance,
            'active_optimizations': len(self.active_optimizations),
            'system_health': 'excellent' if overall_performance > 0.8 else 
                           'good' if overall_performance > 0.6 else
                           'needs_improvement'
        }


@dataclass
class OptimizedPrompt:
    """Represents an optimized prompt with metadata."""
    prompt_id: str
    prompt_text: str
    context: Dict[str, Any]
    optimization_applied: bool = True
    timestamp: float = field(default_factory=time.time)
