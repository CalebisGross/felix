"""Agent visualization components for Felix Framework."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import math


def create_helix_visualization(agents: List[Dict[str, Any]],
                              helix_config: Optional[Dict[str, Any]] = None) -> go.Figure:
    """
    Create 3D helix visualization showing agent positions.

    Args:
        agents: List of agent data with positions
        helix_config: Helix configuration parameters

    Returns:
        Plotly 3D figure
    """
    if helix_config is None:
        helix_config = {
            'top_radius': 3.0,
            'bottom_radius': 0.5,
            'height': 8.0,
            'turns': 2
        }

    # Generate helix path
    t = np.linspace(0, 2*np.pi*helix_config['turns'], 100)
    z = np.linspace(0, helix_config['height'], 100)

    # Calculate radius at each height
    top_r = helix_config['top_radius']
    bottom_r = helix_config['bottom_radius']
    r = top_r - (top_r - bottom_r) * (z / helix_config['height'])

    # Generate helix coordinates
    x = r * np.cos(t)
    y = r * np.sin(t)

    # Create helix trace
    helix_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        name='Helix Path',
        line=dict(
            color=z,
            colorscale='Viridis',
            width=6,
            cmin=0,
            cmax=helix_config['height']
        ),
        hoverinfo='skip'
    )

    # Add agent positions if available
    traces = [helix_trace]

    if agents:
        agent_x = []
        agent_y = []
        agent_z = []
        agent_text = []
        agent_colors = []

        for agent in agents:
            # Calculate agent position on helix
            depth = agent.get('depth', np.random.uniform(0, helix_config['height']))
            angle = agent.get('angle', np.random.uniform(0, 2*np.pi*helix_config['turns']))

            # Interpolate radius at this depth
            radius = top_r - (top_r - bottom_r) * (depth / helix_config['height'])

            agent_x.append(radius * np.cos(angle))
            agent_y.append(radius * np.sin(angle))
            agent_z.append(depth)

            agent_text.append(
                f"Agent: {agent.get('agent_id', 'Unknown')}<br>"
                f"Confidence: {agent.get('confidence', 0):.2%}<br>"
                f"Domain: {agent.get('domain', 'General')}<br>"
                f"Depth: {depth:.2f}"
            )

            agent_colors.append(agent.get('confidence', 0.5))

        # Create agent scatter trace
        agent_trace = go.Scatter3d(
            x=agent_x, y=agent_y, z=agent_z,
            mode='markers+text',
            name='Agents',
            marker=dict(
                size=12,
                color=agent_colors,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title="Confidence",
                    x=1.1
                ),
                line=dict(color='black', width=0.5)
            ),
            text=[f"A{i+1}" for i in range(len(agents))],
            textposition="top center",
            hovertext=agent_text,
            hoverinfo='text'
        )
        traces.append(agent_trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title="Agent Positions on Helix",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2)
        ),
        height=600,
        showlegend=True,
        hovermode='closest'
    )

    return fig


def create_agent_timeline(agent_data: pd.DataFrame) -> go.Figure:
    """
    Create a timeline showing agent activity over time.

    Args:
        agent_data: DataFrame with agent activity data

    Returns:
        Plotly figure
    """
    if agent_data.empty:
        return go.Figure().add_annotation(
            text="No agent activity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    fig = go.Figure()

    # Group by agent and create traces
    for agent_id in agent_data['agent_id'].unique():
        agent_df = agent_data[agent_data['agent_id'] == agent_id]

        fig.add_trace(go.Scatter(
            x=pd.to_datetime(agent_df['timestamp'], unit='s'),
            y=agent_df['confidence'],
            mode='lines+markers',
            name=str(agent_id),
            line=dict(width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title="Agent Activity Timeline",
        xaxis_title="Time",
        yaxis_title="Confidence",
        hovermode='x unified',
        height=400,
        yaxis=dict(range=[0, 1])
    )

    return fig


def display_agent_cards(agents: List[Dict[str, Any]], columns: int = 3):
    """
    Display agent information as cards in a grid.

    Args:
        agents: List of agent data
        columns: Number of columns in grid
    """
    if not agents:
        st.info("No active agents to display")
        return

    cols = st.columns(columns)

    for idx, agent in enumerate(agents):
        with cols[idx % columns]:
            # Create card container
            with st.container():
                # Card styling
                st.markdown("""
                    <style>
                        .agent-card {
                            background-color: #f8f9fa;
                            border-radius: 10px;
                            padding: 15px;
                            margin: 10px 0;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }
                    </style>
                """, unsafe_allow_html=True)

                # Agent header
                agent_id = agent.get('agent_id', 'Unknown')
                confidence = agent.get('confidence', 0)

                # Confidence color
                if confidence > 0.8:
                    conf_color = "green"
                elif confidence > 0.5:
                    conf_color = "orange"
                else:
                    conf_color = "red"

                st.markdown(f"""
                    <div class="agent-card">
                        <h4>🤖 {agent_id}</h4>
                        <p><strong>Confidence:</strong>
                           <span style="color: {conf_color};">{confidence:.1%}</span></p>
                        <p><strong>Domain:</strong> {agent.get('domain', 'General')}</p>
                        <p><strong>Outputs:</strong> {agent.get('output_count', 0)}</p>
                    </div>
                """, unsafe_allow_html=True)


def create_agent_heatmap(agent_metrics: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing agent performance metrics.

    Args:
        agent_metrics: DataFrame with agent performance data

    Returns:
        Plotly heatmap figure
    """
    if agent_metrics.empty:
        return go.Figure().add_annotation(
            text="No agent metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Prepare data for heatmap
    metrics_cols = ['avg_confidence', 'min_confidence', 'max_confidence', 'output_count']
    available_cols = [col for col in metrics_cols if col in agent_metrics.columns]

    if not available_cols:
        return go.Figure()

    # Normalize values for better visualization
    heatmap_data = agent_metrics[available_cols].copy()

    # Normalize output_count if present
    if 'output_count' in heatmap_data.columns:
        max_count = heatmap_data['output_count'].max()
        if max_count > 0:
            heatmap_data['output_count'] = heatmap_data['output_count'] / max_count

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.T.values,
        x=agent_metrics['agent_id'].values,
        y=available_cols,
        colorscale='RdYlGn',
        hoverongaps=False
    ))

    fig.update_layout(
        title="Agent Performance Heatmap",
        xaxis_title="Agent ID",
        yaxis_title="Metric",
        height=300
    )

    return fig


def create_agent_spawning_chart(spawn_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart showing agent spawning patterns.

    Args:
        spawn_data: List of spawning events

    Returns:
        Plotly figure
    """
    if not spawn_data:
        return go.Figure().add_annotation(
            text="No spawning data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Convert to DataFrame
    df = pd.DataFrame(spawn_data)

    # Create Gantt-style chart
    fig = go.Figure()

    for idx, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row.get('start_time'), row.get('end_time', row.get('start_time'))],
            y=[row.get('agent_id')] * 2,
            mode='lines+markers',
            name=row.get('agent_id'),
            line=dict(width=10),
            marker=dict(size=12),
            showlegend=False
        ))

    fig.update_layout(
        title="Agent Spawning Timeline",
        xaxis_title="Time",
        yaxis_title="Agent",
        height=400,
        hovermode='x'
    )

    return fig


def display_confidence_distribution(agents: List[Dict[str, Any]]):
    """
    Display confidence distribution across agents.

    Args:
        agents: List of agent data with confidence scores
    """
    if not agents:
        st.info("No agent confidence data available")
        return

    confidences = [a.get('confidence', 0) for a in agents]

    # Create histogram
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Agent Confidence Distribution",
        labels={'x': 'Confidence', 'y': 'Count'},
        color_discrete_sequence=['skyblue']
    )

    fig.add_vline(
        x=np.mean(confidences),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(confidences):.2f}"
    )

    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{np.mean(confidences):.3f}")
    with col2:
        st.metric("Median", f"{np.median(confidences):.3f}")
    with col3:
        st.metric("Min", f"{np.min(confidences):.3f}")
    with col4:
        st.metric("Max", f"{np.max(confidences):.3f}")


def create_agent_communication_graph(messages: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a network graph showing agent communication patterns.

    Args:
        messages: List of message data between agents

    Returns:
        Plotly network graph
    """
    if not messages:
        return go.Figure().add_annotation(
            text="No communication data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Extract unique agents
    agents = set()
    edges = []

    for msg in messages:
        sender = msg.get('sender')
        receiver = msg.get('receiver')
        if sender and receiver:
            agents.add(sender)
            agents.add(receiver)
            edges.append((sender, receiver))

    agents = list(agents)
    n = len(agents)

    if n == 0:
        return go.Figure()

    # Position nodes in a circle
    pos = {}
    for i, agent in enumerate(agents):
        angle = 2 * math.pi * i / n
        pos[agent] = (math.cos(angle), math.sin(angle))

    # Create edge traces
    edge_traces = []
    for sender, receiver in edges:
        if sender in pos and receiver in pos:
            x0, y0 = pos[sender]
            x1, y1 = pos[receiver]

            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))

    # Create node trace
    node_x = [pos[agent][0] for agent in agents]
    node_y = [pos[agent][1] for agent in agents]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=agents,
        textposition="top center",
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(color='darkblue', width=2)
        ),
        hovertext=agents,
        hoverinfo='text'
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title="Agent Communication Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    return fig