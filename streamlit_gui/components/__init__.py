"""
Reusable Streamlit components for the Felix GUI.
"""

from .metrics_display import (
    display_metric_card,
    display_system_status,
    create_confidence_gauge,
    create_activity_timeline,
    display_metrics_grid,
    create_agent_network_graph,
    display_progress_bar,
    create_comparison_chart,
    display_alert
)

from .agent_visualizer import (
    create_helix_visualization,
    create_agent_timeline,
    display_agent_cards,
    create_agent_heatmap,
    create_agent_spawning_chart,
    display_confidence_distribution,
    create_agent_communication_graph
)

from .log_monitor import (
    LogMonitor,
    create_auto_refresh_component,
    display_activity_feed,
    create_status_indicator,
    monitor_system_resources,
    create_alert_system,
    save_dashboard_state,
    load_dashboard_state
)

from .web_search_monitor import WebSearchMonitor
from .truth_assessment_display import TruthAssessmentDisplay

__all__ = [
    # Metrics display
    'display_metric_card',
    'display_system_status',
    'create_confidence_gauge',
    'create_activity_timeline',
    'display_metrics_grid',
    'create_agent_network_graph',
    'display_progress_bar',
    'create_comparison_chart',
    'display_alert',

    # Agent visualizer
    'create_helix_visualization',
    'create_agent_timeline',
    'display_agent_cards',
    'create_agent_heatmap',
    'create_agent_spawning_chart',
    'display_confidence_distribution',
    'create_agent_communication_graph',

    # Log monitor
    'LogMonitor',
    'create_auto_refresh_component',
    'display_activity_feed',
    'create_status_indicator',
    'monitor_system_resources',
    'create_alert_system',
    'save_dashboard_state',
    'load_dashboard_state',

    # Web search monitor
    'WebSearchMonitor',

    # Truth assessment display
    'TruthAssessmentDisplay'
]