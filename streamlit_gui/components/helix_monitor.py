"""Real-time helix visualization component for Felix Framework."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import AgentRegistry


class HelixMonitor:
    """Real-time monitoring and visualization of agents on the helix."""

    def __init__(self, helix_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the helix monitor.

        Args:
            helix_config: Helix configuration parameters
        """
        if helix_config is None:
            helix_config = {
                'top_radius': 3.0,
                'bottom_radius': 0.5,
                'height': 8.0,
                'turns': 2
            }

        self.helix_config = helix_config
        self.helix = HelixGeometry(
            top_radius=helix_config['top_radius'],
            bottom_radius=helix_config['bottom_radius'],
            height=helix_config['height'],
            turns=helix_config['turns']
        )

        # Load visualization settings
        self.viz_config = self.load_visualization_config()

    def load_visualization_config(self) -> Dict[str, Any]:
        """Load visualization configuration from file."""
        config_path = Path("felix_visualization_config.json")

        default_config = {
            "visualization": {
                "show_spokes": True,
                "show_trails": False,
                "trail_length": 5,
                "update_speed": 1.0,
                "camera_position": "isometric",
                "agent_size": "medium",
                "color_scheme": "confidence",
                "show_phase_boundaries": True,
                "show_agent_labels": True,
                "fps": 10
            }
        }

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except:
                pass

        return default_config["visualization"]

    def save_visualization_config(self, config: Dict[str, Any]):
        """Save visualization configuration to file."""
        config_path = Path("felix_visualization_config.json")
        full_config = {"visualization": config}

        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)

    def create_helix_path(self) -> go.Scatter3d:
        """Create the helix path trace."""
        # Generate helix points
        t = np.linspace(0, 1, 200)

        # Calculate positions - FIXED ORIENTATION
        positions = []
        for t_val in t:
            x, y, z = self.helix.get_position(t_val)
            positions.append((x, y, z))

        x_vals = [p[0] for p in positions]
        y_vals = [p[1] for p in positions]
        z_vals = [p[2] for p in positions]

        # Create helix trace with gradient coloring
        helix_trace = go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='lines',
            name='Helix Path',
            line=dict(
                color=z_vals,
                colorscale='Viridis',
                width=8,
                cmin=0,
                cmax=self.helix_config['height'],
                colorbar=dict(
                    title="Height<br>(Descent)",
                    x=1.02,
                    y=0.5,
                    thickness=15
                )
            ),
            hoverinfo='skip'
        )

        return helix_trace

    def create_center_post(self) -> go.Scatter3d:
        """Create the central post representing CentralPost hub."""
        height = self.helix_config['height']

        center_post = go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[0, height],
            mode='lines+markers',
            name='CentralPost Hub',
            line=dict(
                color='silver',
                width=10
            ),
            marker=dict(
                size=[12, 12],
                color='red',
                symbol=['diamond', 'diamond']
            ),
            hovertext=['Hub Base', 'Hub Top'],
            hoverinfo='text'
        )

        return center_post

    def create_phase_boundaries(self) -> List[go.Mesh3d]:
        """Create transparent planes showing phase boundaries."""
        if not self.viz_config.get("show_phase_boundaries", True):
            return []

        height = self.helix_config['height']
        max_radius = self.helix_config['top_radius']

        boundaries = []

        # Phase boundaries: exploration (0.3), analysis (0.7)
        phase_heights = [
            (height * 0.7, 'Exploration→Analysis', 'rgba(88, 166, 255, 0.1)'),
            (height * 0.3, 'Analysis→Synthesis', 'rgba(255, 166, 88, 0.1)')
        ]

        for z_pos, label, color in phase_heights:
            # Create circular mesh at this height
            theta = np.linspace(0, 2*np.pi, 20)
            r = self.helix.get_radius(z_pos)

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = [z_pos] * len(theta)

            # Add center point for triangulation
            x = np.append(x, 0)
            y = np.append(y, 0)
            z = np.append(z, z_pos)

            boundaries.append(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=0.1, color=color),
                name=label,
                hoverinfo='name'
            ))

        return boundaries

    def create_agents(self, agent_positions: List[Dict[str, Any]]) -> List[go.Scatter3d]:
        """Create agent traces with current positions."""
        if not agent_positions:
            return []

        traces = []

        # Group agents by type for different styling
        agent_types = {}
        for agent in agent_positions:
            agent_type = agent.get('type', 'generic')
            if agent_type not in agent_types:
                agent_types[agent_type] = []
            agent_types[agent_type].append(agent)

        # Type-specific colors
        type_colors = {
            'research': '#3fb950',  # Green
            'analysis': '#58a6ff',  # Blue
            'critic': '#d29922',     # Amber
            'generic': '#8b949e'     # Gray
        }

        for agent_type, agents in agent_types.items():
            x_vals = []
            y_vals = []
            z_vals = []
            hover_texts = []
            colors = []
            sizes = []

            for agent in agents:
                # Get position from agent data
                t = agent.get('progress', 0.0)
                x, y, z = self.helix.get_position(t)

                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)

                # Create hover text
                hover_text = (
                    f"Agent: {agent.get('id', 'Unknown')}<br>"
                    f"Type: {agent_type}<br>"
                    f"Progress: {t:.1%}<br>"
                    f"Confidence: {agent.get('confidence', 0):.2%}"
                )
                hover_texts.append(hover_text)

                # Color based on configuration
                if self.viz_config['color_scheme'] == 'confidence':
                    colors.append(agent.get('confidence', 0.5))
                else:
                    colors.append(type_colors.get(agent_type, '#8b949e'))

                # Size based on confidence and configuration
                base_size = {'small': 8, 'medium': 12, 'large': 16}[self.viz_config['agent_size']]
                sizes.append(base_size * (0.5 + agent.get('confidence', 0.5)))

            # Create agent trace
            if self.viz_config['color_scheme'] == 'confidence':
                trace = go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers+text' if self.viz_config['show_agent_labels'] else 'markers',
                    name=f'{agent_type.title()} Agents',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=1,
                        showscale=(agent_type == list(agent_types.keys())[0]),
                        colorbar=dict(
                            title="Confidence",
                            x=1.1
                        ) if agent_type == list(agent_types.keys())[0] else None,
                        line=dict(color='black', width=0.5)
                    ),
                    text=[agent.get('id', '')[:5] for agent in agents] if self.viz_config['show_agent_labels'] else None,
                    textposition="top center",
                    hovertext=hover_texts,
                    hoverinfo='text'
                )
            else:
                trace = go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers+text' if self.viz_config['show_agent_labels'] else 'markers',
                    name=f'{agent_type.title()} Agents',
                    marker=dict(
                        size=sizes,
                        color=type_colors.get(agent_type, '#8b949e'),
                        line=dict(color='black', width=0.5)
                    ),
                    text=[agent.get('id', '')[:5] for agent in agents] if self.viz_config['show_agent_labels'] else None,
                    textposition="top center",
                    hovertext=hover_texts,
                    hoverinfo='text'
                )

            traces.append(trace)

            # Add spokes if enabled
            if self.viz_config['show_spokes']:
                spoke_x = []
                spoke_y = []
                spoke_z = []

                for agent in agents:
                    t = agent.get('progress', 0.0)
                    x, y, z = self.helix.get_position(t)

                    # Line from center to agent
                    spoke_x.extend([0, x, None])
                    spoke_y.extend([0, y, None])
                    spoke_z.extend([z, z, None])

                spoke_trace = go.Scatter3d(
                    x=spoke_x,
                    y=spoke_y,
                    z=spoke_z,
                    mode='lines',
                    name=f'{agent_type.title()} Spokes',
                    line=dict(
                        color='rgba(88, 166, 255, 0.2)',
                        width=2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )

                traces.append(spoke_trace)

        return traces

    def create_visualization(self, agent_positions: List[Dict[str, Any]] = None) -> go.Figure:
        """
        Create complete 3D visualization with agents.

        Args:
            agent_positions: List of agent position data

        Returns:
            Plotly figure with helix and agents
        """
        traces = []

        # Add helix path
        traces.append(self.create_helix_path())

        # Add center post
        traces.append(self.create_center_post())

        # Add phase boundaries
        traces.extend(self.create_phase_boundaries())

        # Add agents and spokes
        if agent_positions:
            traces.extend(self.create_agents(agent_positions))

        # Create figure
        fig = go.Figure(data=traces)

        # Camera positions
        camera_presets = {
            'isometric': dict(eye=dict(x=1.5, y=1.5, z=0.7)),
            'top': dict(eye=dict(x=0, y=0, z=2)),
            'side': dict(eye=dict(x=2, y=0, z=0.5)),
            'front': dict(eye=dict(x=0, y=2, z=0.5))
        }

        camera = camera_presets.get(
            self.viz_config.get('camera_position', 'isometric'),
            camera_presets['isometric']
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text="Felix Helix: Real-Time Agent Monitor",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Height (Top→Bottom)",
                camera=camera,
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=2)
            ),
            height=700,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            ),
            uirevision='helix_view'  # Preserve camera position/zoom during updates
        )

        return fig

    def render_controls(self) -> Dict[str, Any]:
        """Render control panel and return settings."""
        st.markdown("### Visualization Controls")

        # Real-time mode toggle
        real_time = st.checkbox("🔴 Real-Time Mode", value=False, key="helix_real_time")

        # Speed control
        speed = st.slider(
            "Animation Speed",
            min_value=0.1,
            max_value=2.0,
            value=self.viz_config.get('update_speed', 1.0),
            step=0.1,
            key="helix_speed"
        )

        # Display options
        with st.expander("Display Options"):
            show_spokes = st.checkbox(
                "Show Spokes",
                value=self.viz_config.get('show_spokes', True),
                key="helix_spokes"
            )

            show_trails = st.checkbox(
                "Show Trails",
                value=self.viz_config.get('show_trails', False),
                key="helix_trails"
            )

            show_labels = st.checkbox(
                "Show Agent Labels",
                value=self.viz_config.get('show_agent_labels', True),
                key="helix_labels"
            )

            show_boundaries = st.checkbox(
                "Show Phase Boundaries",
                value=self.viz_config.get('show_phase_boundaries', True),
                key="helix_boundaries"
            )

            agent_size = st.select_slider(
                "Agent Size",
                options=['small', 'medium', 'large'],
                value=self.viz_config.get('agent_size', 'medium'),
                key="helix_agent_size"
            )

            color_scheme = st.radio(
                "Color Scheme",
                options=['confidence', 'type'],
                index=0 if self.viz_config.get('color_scheme', 'confidence') == 'confidence' else 1,
                key="helix_color_scheme"
            )

        # Camera position
        camera_pos = st.selectbox(
            "Camera View",
            options=['isometric', 'top', 'side', 'front'],
            index=['isometric', 'top', 'side', 'front'].index(
                self.viz_config.get('camera_position', 'isometric')
            ),
            key="helix_camera"
        )

        # Update configuration
        new_config = {
            'show_spokes': show_spokes,
            'show_trails': show_trails,
            'trail_length': 5,
            'update_speed': speed,
            'camera_position': camera_pos,
            'agent_size': agent_size,
            'color_scheme': color_scheme,
            'show_phase_boundaries': show_boundaries,
            'show_agent_labels': show_labels,
            'fps': 10
        }

        # Save button
        if st.button("💾 Save Settings", key="helix_save"):
            self.viz_config = new_config
            self.save_visualization_config(new_config)
            st.success("Settings saved!")

        return {
            'real_time': real_time,
            'config': new_config
        }


def generate_sample_agents(num_agents: int = 10, randomize_progress: bool = False) -> List[Dict[str, Any]]:
    """
    Generate sample agent data for testing.

    Args:
        num_agents: Number of agents to generate
        randomize_progress: If True, agents start at random positions. If False, all start at top (0.0)

    Returns:
        List of agent dictionaries with position data
    """
    agents = []
    agent_types = ['research', 'analysis', 'critic']

    for i in range(num_agents):
        # Start at top (0.0) by default, or random if specified
        progress = np.random.uniform(0, 1) if randomize_progress else 0.0

        agent = {
            'id': f'agent_{i:03d}',
            'type': np.random.choice(agent_types),
            'progress': progress,
            'confidence': 0.5 + np.random.uniform(-0.2, 0.3)  # Start with good confidence
        }
        agents.append(agent)

    return agents