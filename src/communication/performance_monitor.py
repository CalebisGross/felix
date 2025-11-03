"""
Performance Monitor for the Felix Framework.

Handles performance metrics collection and analysis for performance benchmarking,
tracking message throughput, communication overhead, and scaling performance.

Key Features:
- Message processing throughput calculation
- Communication overhead measurement
- Scaling metrics (agent count vs processing time)
- Communication efficiency benchmarking (O(N) vs O(N²) comparison)
- Performance summary generation
- Async processing metrics

This module was extracted from CentralPost to improve separation of concerns
and maintainability while preserving all functionality.
"""

import time
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitors and tracks system performance metrics.

    Responsibilities:
    - Track message processing throughput
    - Measure communication overhead
    - Record scaling performance (agent count vs time)
    - Calculate average overhead ratios
    - Generate comprehensive performance summaries
    - Support performance benchmarking and analysis
    """

    def __init__(self, metrics_enabled: bool = True):
        """
        Initialize Performance Monitor.

        Args:
            metrics_enabled: Enable/disable metrics collection
        """
        self._metrics_enabled = metrics_enabled
        self._start_time = time.time()

        # Message processing metrics
        self._total_messages_processed: int = 0
        self._processing_times: List[float] = []

        # Communication overhead metrics (efficiency benchmarking)
        self._overhead_ratios: List[float] = []

        # Scaling metrics (agent count → processing time)
        self._scaling_metrics: Dict[int, float] = {}

        if metrics_enabled:
            logger.info("✓ PerformanceMonitor initialized (metrics enabled)")
        else:
            logger.info("✓ PerformanceMonitor initialized (metrics disabled)")

    def increment_message_count(self, count: int = 1) -> None:
        """Increment total messages processed counter."""
        if self._metrics_enabled:
            self._total_messages_processed += count

    def record_processing_time(self, processing_time: float) -> None:
        """
        Record message processing time.

        Args:
            processing_time: Time taken to process a message (seconds)
        """
        if self._metrics_enabled:
            self._processing_times.append(processing_time)

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

        # Calculate communication overhead from average message time
        if self._processing_times:
            avg_msg_time = sum(self._processing_times) / len(self._processing_times)
            communication_overhead = avg_msg_time * num_messages
            return communication_overhead

        return 0.0

    def record_overhead_ratio(self, overhead_ratio: float) -> None:
        """
        Record overhead ratio for performance benchmarking.

        Args:
            overhead_ratio: Communication overhead / processing time ratio
        """
        if self._metrics_enabled:
            self._overhead_ratios.append(overhead_ratio)

    def get_average_overhead_ratio(self) -> float:
        """
        Get average overhead ratio across all measurements.

        Returns:
            Average overhead ratio (for efficiency benchmarking)
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
            logger.debug(f"Scaling metric recorded: {agent_count} agents → {processing_time:.3f}s")

    def get_scaling_metrics(self) -> Dict[int, float]:
        """
        Get scaling performance metrics.

        Returns:
            Dictionary mapping agent count to processing time
        """
        return self._scaling_metrics.copy()

    def get_performance_summary(self, active_connections: int = 0,
                               async_processors: int = 0,
                               async_queue_size: int = 0) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for analysis.

        Args:
            active_connections: Number of active agent connections
            async_processors: Number of async processors running
            async_queue_size: Size of async message queue

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
            "active_connections": active_connections,
            "uptime": time.time() - self._start_time,
            "async_processors": async_processors,
            "async_queue_size": async_queue_size,
            "avg_processing_time": sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0.0,
            "total_measurements": len(self._processing_times)
        }

    def reset_metrics(self) -> None:
        """Reset all metrics (for testing/benchmarking)."""
        self._total_messages_processed = 0
        self._processing_times.clear()
        self._overhead_ratios.clear()
        self._scaling_metrics.clear()
        self._start_time = time.time()
        logger.info("Performance metrics reset")

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._metrics_enabled

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self._start_time

    def get_total_messages(self) -> int:
        """Get total messages processed."""
        return self._total_messages_processed
