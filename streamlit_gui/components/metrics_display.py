"""
Reusable metrics display components for the Streamlit GUI.

Provides pre-styled metric cards and displays for consistent UI.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional


def display_metric_card(title: str, value: Any, delta: Optional[str] = None,
                        help_text: Optional[str] = None):
    """
    Display a styled metric card.

    Args:
        title: Metric title
        value: Main metric value
        delta: Change indicator (optional)
        help_text: Tooltip text (optional)
    """
    with st.container():
        if help_text:
            st.metric(title, value, delta, help=help_text)
        else:
            st.metric(title, value, delta)


def display_system_status(is_running: bool, additional_info: Dict[str, Any] = None):
    """
    Display system status with visual indicators.

    Args:
        is_running: Whether Felix system is running
        additional_info: Additional status information
    """
    status_color = "green" if is_running else "red"
    status_icon = "🟢" if is_running else "🔴"
    status_text = "Running" if is_running else "Stopped"

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown(f"<h2 style='color: {status_color};'>{status_icon}</h2>",
                   unsafe_allow_html=True)

    with col2:
        st.markdown(f"### System Status: {status_text}")

        if additional_info:
            for key, value in additional_info.items():
                st.text(f"{key}: {value}")


def create_confidence_gauge(confidence: float, title: str = "Confidence") -> go.Figure:
    """
    Create a gauge chart for confidence visualization.

    Args:
        confidence: Confidence value (0-1)
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 75},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 25], 'color': "lightgray"},
                   {'range': [25, 50], 'color': "gray"},
                   {'range': [50, 75], 'color': "lightblue"},
                   {'range': [75, 100], 'color': "blue"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 90}}
    ))
    fig.update_layout(height=250)
    return fig


def create_activity_timeline(data: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """
    Create an activity timeline chart.

    Args:
        data: DataFrame with activity data
        x_col: Column name for x-axis (time)
        y_col: Column name for y-axis (values)

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add line trace
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        name='Activity',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8, color='navy')
    ))

    # Add fill under line
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        fill='tozeroy',
        fillcolor='rgba(135, 206, 250, 0.3)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="Activity Over Time",
        xaxis_title="Time",
        yaxis_title="Activity Level",
        hovermode='x unified',
        height=350
    )

    return fig


def display_metrics_grid(metrics: Dict[str, Dict[str, Any]], columns: int = 3):
    """
    Display metrics in a grid layout.

    Args:
        metrics: Dictionary of metrics with format:
                 {name: {'value': x, 'delta': y, 'help': z}}
        columns: Number of columns in grid
    """
    cols = st.columns(columns)

    for idx, (name, data) in enumerate(metrics.items()):
        with cols[idx % columns]:
            display_metric_card(
                name,
                data.get('value'),
                data.get('delta'),
                data.get('help')
            )


def create_agent_network_graph(agents: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a network graph visualization of agent interactions.

    Args:
        agents: List of agent data dictionaries

    Returns:
        Plotly figure object
    """
    if not agents:
        return go.Figure()

    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    import math

    num_agents = len(agents)
    for i, agent in enumerate(agents):
        angle = 2 * math.pi * i / num_agents
        x = math.cos(angle)
        y = math.sin(angle)

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Agent: {agent.get('agent_id', 'Unknown')}<br>"
                        f"Confidence: {agent.get('confidence', 0):.2f}")
        node_size.append(20 + agent.get('confidence', 0.5) * 30)

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[a.get('agent_id', '') for a in agents],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_size,
            color=[a.get('confidence', 0) for a in agents],
            colorbar=dict(
                thickness=15,
                title="Confidence",
                xanchor="left",
                titleside="right"
            ),
            line_width=2
        ),
        hovertext=node_text
    )

    # Create figure
    fig = go.Figure(data=[node_trace])

    # Update layout using modern approach
    fig.update_layout(
        title="Agent Network",
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )

    return fig


def display_progress_bar(current: int, total: int, label: str = "Progress"):
    """
    Display a custom progress bar.

    Args:
        current: Current progress value
        total: Total/maximum value
        label: Progress bar label
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{label}: {current}/{total} ({progress*100:.1f}%)")


def create_comparison_chart(data1: pd.DataFrame, data2: pd.DataFrame,
                           label1: str, label2: str, metric: str) -> go.Figure:
    """
    Create a comparison chart between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        label1: Label for first dataset
        label2: Label for second dataset
        metric: Metric column to compare

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add first dataset
    if not data1.empty and metric in data1.columns:
        fig.add_trace(go.Bar(
            name=label1,
            x=data1.index,
            y=data1[metric],
            marker_color='indianred'
        ))

    # Add second dataset
    if not data2.empty and metric in data2.columns:
        fig.add_trace(go.Bar(
            name=label2,
            x=data2.index,
            y=data2[metric],
            marker_color='lightsalmon'
        ))

    fig.update_layout(
        title=f"{metric} Comparison",
        barmode='group',
        xaxis_title="Index",
        yaxis_title=metric,
        height=400
    )

    return fig


def display_alert(message: str, alert_type: str = "info"):
    """
    Display a styled alert message.

    Args:
        message: Alert message
        alert_type: Type of alert ('info', 'success', 'warning', 'error')
    """
    icon_map = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }

    color_map = {
        'info': '#d1ecf1',
        'success': '#d4edda',
        'warning': '#fff3cd',
        'error': '#f8d7da'
    }

    icon = icon_map.get(alert_type, 'ℹ️')
    bg_color = color_map.get(alert_type, '#d1ecf1')

    st.markdown(
        f"""
        <div style='background-color: {bg_color}; padding: 1rem;
                    border-radius: 0.5rem; margin: 1rem 0;'>
            {icon} <strong>{message}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )