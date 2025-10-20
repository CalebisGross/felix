"""
Configuration page for Felix Framework.

Provides read-only configuration viewing, helix geometry visualization,
and configuration export functionality.
"""

import streamlit as st
import yaml
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Felix Configuration",
    page_icon="⚙️",
    layout="wide"
)

def load_config_file(file_path: str) -> dict:
    """
    Load configuration from file.

    Args:
        file_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not Path(file_path).exists():
        return {}

    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}


def create_helix_3d_visualization(helix_config: dict) -> go.Figure:
    """
    Create 3D visualization of helix geometry.

    Args:
        helix_config: Helix configuration parameters

    Returns:
        Plotly 3D figure
    """
    # Extract parameters with defaults
    turns = helix_config.get('turns', 2)
    top_radius = helix_config.get('top_radius', 3.0)
    bottom_radius = helix_config.get('bottom_radius', 0.5)
    height = helix_config.get('height', 8.0)

    # Generate helix points
    t = np.linspace(0, 2*np.pi*turns, 200)
    z = np.linspace(0, height, 200)

    # Calculate radius at each height (linear interpolation)
    r = top_radius - (top_radius - bottom_radius) * (z / height)

    # Generate x, y coordinates
    x = r * np.cos(t)
    y = r * np.sin(t)

    # Create the helix trace
    helix_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        name='Helix Path',
        line=dict(
            color=z,
            colorscale='Viridis',
            width=8,
            cmin=0,
            cmax=height,
            colorbar=dict(
                title="Depth",
                x=1.02,
                y=0.5,
                thickness=15
            )
        )
    )

    # Add reference points for phases
    phase_points = []
    phase_labels = [
        ("Exploration", 0, top_radius),
        ("Analysis", height * 0.33, top_radius * 0.7 + bottom_radius * 0.3),
        ("Synthesis", height * 0.66, top_radius * 0.3 + bottom_radius * 0.7),
        ("Conclusion", height, bottom_radius)
    ]

    for label, z_pos, radius in phase_labels:
        # Position on helix at this height
        angle = (z_pos / height) * 2 * np.pi * turns
        x_pos = radius * np.cos(angle)
        y_pos = radius * np.sin(angle)

        phase_points.append(
            go.Scatter3d(
                x=[x_pos],
                y=[y_pos],
                z=[z_pos],
                mode='markers+text',
                name=label,
                marker=dict(size=12, symbol='diamond'),
                text=[label],
                textposition="top center"
            )
        )

    # Create figure
    fig = go.Figure(data=[helix_trace] + phase_points)

    # Update layout
    fig.update_layout(
        title=dict(
            text="Helix Geometry Visualization",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth (Progression)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.7),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2)
        ),
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
    )

    return fig


def compare_configurations(config1: dict, config2: dict, path: str = "") -> list:
    """
    Compare two configurations and return differences.

    Args:
        config1: First configuration
        config2: Second configuration
        path: Current path in configuration tree

    Returns:
        List of differences
    """
    differences = []

    # Get all keys from both configs
    all_keys = set(config1.keys()) | set(config2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        if key not in config1:
            differences.append({
                'path': current_path,
                'status': 'added',
                'value1': None,
                'value2': config2[key]
            })
        elif key not in config2:
            differences.append({
                'path': current_path,
                'status': 'removed',
                'value1': config1[key],
                'value2': None
            })
        elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
            # Recursively compare nested dictionaries
            nested_diffs = compare_configurations(config1[key], config2[key], current_path)
            differences.extend(nested_diffs)
        elif config1[key] != config2[key]:
            differences.append({
                'path': current_path,
                'status': 'modified',
                'value1': config1[key],
                'value2': config2[key]
            })

    return differences


def main():
    st.title("⚙️ Configuration Viewer")
    st.markdown("View and export Felix configuration (read-only)")

    # Configuration sources
    config_sources = {
        "felix_gui_config.json": "tkinter GUI Configuration",
        "streamlit_config.yaml": "Streamlit Configuration",
        "configs/default_config.yaml": "Default Configuration",
        "exp/optimal_parameters.yaml": "Optimal Parameters"
    }

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📋 View Config", "📊 Compare", "💾 Export"])

    with tab1:
        st.subheader("Current Configuration")

        # Select config source
        selected_source = st.selectbox(
            "Configuration Source",
            options=list(config_sources.keys()),
            format_func=lambda x: config_sources[x]
        )

        # Load configuration
        config = load_config_file(selected_source)

        if config:
            # Display configuration sections
            col1, col2 = st.columns(2)

            with col1:
                # Helix Geometry
                st.markdown("### 🌀 Helix Geometry")
                helix_config = config.get('helix', {
                    'top_radius': 3.0,
                    'bottom_radius': 0.5,
                    'height': 8.0,
                    'turns': 2
                })

                for key, value in helix_config.items():
                    st.text(f"{key}: {value}")

                # Agent Configuration
                st.markdown("### 🤖 Agent Configuration")
                agent_config = {
                    'max_agents': config.get('max_agents', 25),
                    'base_token_budget': config.get('base_token_budget', 2500),
                    'temperature_top': config.get('temperature', {}).get('top', 1.0),
                    'temperature_bottom': config.get('temperature', {}).get('bottom', 0.2)
                }

                for key, value in agent_config.items():
                    st.text(f"{key}: {value}")

            with col2:
                # LM Studio Connection
                st.markdown("### 🔌 LM Studio Connection")
                lm_config = {
                    'host': config.get('lm_host', '127.0.0.1'),
                    'port': config.get('lm_port', 1234),
                    'model': config.get('model', 'default'),
                    'timeout': config.get('timeout', 30)
                }

                for key, value in lm_config.items():
                    st.text(f"{key}: {value}")

                # Dynamic Spawning
                st.markdown("### 🔄 Dynamic Spawning")
                spawn_config = config.get('spawning', {
                    'confidence_threshold': 0.75,
                    'max_depth': 5,
                    'spawn_delay': 0.5
                })

                for key, value in spawn_config.items():
                    st.text(f"{key}: {value}")

            # Full configuration in expandable section
            with st.expander("📄 View Full Configuration"):
                st.json(config)

            # Visualization of helix geometry
            st.divider()
            st.markdown("### 🎨 Helix Geometry Visualization")

            # Create 3D visualization
            fig = create_helix_3d_visualization(helix_config)
            st.plotly_chart(fig, use_container_width=True)

            # Display helix characteristics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Turns", helix_config.get('turns', 2))
            with col2:
                st.metric("Height", f"{helix_config.get('height', 8.0)} units")
            with col3:
                ratio = helix_config.get('top_radius', 3.0) / helix_config.get('bottom_radius', 0.5)
                st.metric("Taper Ratio", f"{ratio:.2f}:1")
            with col4:
                volume = np.pi * helix_config.get('height', 8.0) * (
                    helix_config.get('top_radius', 3.0)**2 +
                    helix_config.get('bottom_radius', 0.5)**2 +
                    helix_config.get('top_radius', 3.0) * helix_config.get('bottom_radius', 0.5)
                ) / 3
                st.metric("Volume", f"{volume:.2f} cubic units")

        else:
            st.warning(f"Configuration file '{selected_source}' not found or could not be loaded.")

    with tab2:
        st.subheader("Configuration Comparison")
        st.info("Compare two configuration files to identify differences.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Configuration A")
            config_a_source = st.selectbox(
                "Select first config",
                options=list(config_sources.keys()),
                key="config_a",
                format_func=lambda x: config_sources[x]
            )
            config_a = load_config_file(config_a_source)

            if config_a:
                st.success(f"Loaded: {config_sources[config_a_source]}")
            else:
                st.error("Could not load Configuration A")

        with col2:
            st.markdown("### Configuration B")
            config_b_source = st.selectbox(
                "Select second config",
                options=list(config_sources.keys()),
                key="config_b",
                format_func=lambda x: config_sources[x]
            )
            config_b = load_config_file(config_b_source)

            if config_b:
                st.success(f"Loaded: {config_sources[config_b_source]}")
            else:
                st.error("Could not load Configuration B")

        if config_a and config_b:
            st.divider()

            # Compare configurations
            differences = compare_configurations(config_a, config_b)

            if differences:
                st.markdown(f"### Found {len(differences)} Differences")

                # Group differences by status
                added = [d for d in differences if d['status'] == 'added']
                removed = [d for d in differences if d['status'] == 'removed']
                modified = [d for d in differences if d['status'] == 'modified']

                if added:
                    with st.expander(f"➕ Added ({len(added)})"):
                        for diff in added:
                            st.markdown(f"**{diff['path']}**: {diff['value2']}")

                if removed:
                    with st.expander(f"➖ Removed ({len(removed)})"):
                        for diff in removed:
                            st.markdown(f"**{diff['path']}**: ~~{diff['value1']}~~")

                if modified:
                    with st.expander(f"🔄 Modified ({len(modified)})"):
                        for diff in modified:
                            st.markdown(f"**{diff['path']}**:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text(f"Before: {diff['value1']}")
                            with col2:
                                st.text(f"After: {diff['value2']}")

                # Side-by-side JSON view
                with st.expander("📋 Side-by-Side View"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Configuration A**")
                        st.json(config_a)
                    with col2:
                        st.markdown("**Configuration B**")
                        st.json(config_b)

            else:
                st.success("✅ Configurations are identical")

    with tab3:
        st.subheader("Export Configuration")

        if config:
            st.info("Export the current configuration in different formats.")

            col1, col2, col3 = st.columns(3)

            with col1:
                # YAML export
                yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
                st.download_button(
                    "📥 Download as YAML",
                    data=yaml_str,
                    file_name="felix_config_export.yaml",
                    mime="text/yaml"
                )

            with col2:
                # JSON export
                json_str = json.dumps(config, indent=2)
                st.download_button(
                    "📥 Download as JSON",
                    data=json_str,
                    file_name="felix_config_export.json",
                    mime="application/json"
                )

            with col3:
                # Original format
                if selected_source:
                    with open(selected_source, 'r') as f:
                        original_content = f.read()

                    file_ext = Path(selected_source).suffix
                    st.download_button(
                        "📥 Download Original",
                        data=original_content,
                        file_name=f"felix_config_original{file_ext}",
                        mime="text/plain"
                    )

            # Custom export options
            st.divider()
            st.markdown("### Custom Export Options")

            # Select sections to export
            sections = list(config.keys())
            selected_sections = st.multiselect(
                "Select sections to export",
                options=sections,
                default=sections
            )

            if selected_sections:
                # Create filtered config
                filtered_config = {k: v for k, v in config.items() if k in selected_sections}

                # Export filtered config
                col1, col2 = st.columns(2)

                with col1:
                    filtered_yaml = yaml.dump(filtered_config, default_flow_style=False)
                    st.download_button(
                        "📥 Export Selected (YAML)",
                        data=filtered_yaml,
                        file_name="felix_config_filtered.yaml",
                        mime="text/yaml"
                    )

                with col2:
                    filtered_json = json.dumps(filtered_config, indent=2)
                    st.download_button(
                        "📥 Export Selected (JSON)",
                        data=filtered_json,
                        file_name="felix_config_filtered.json",
                        mime="application/json"
                    )

                # Preview
                with st.expander("Preview Filtered Configuration"):
                    st.json(filtered_config)

        else:
            st.warning("No configuration loaded to export")

    # Important note about read-only nature
    st.divider()
    st.warning(
        "⚠️ **Read-Only Mode**: Configuration viewing and export only. "
        "To modify settings, use the tkinter GUI Settings tab."
    )

    # Configuration validation info
    with st.expander("ℹ️ Configuration Guidelines"):
        st.markdown("""
        ### Optimal Configuration Ranges

        **Helix Geometry:**
        - Top Radius: 2.0 - 5.0 (wider = more exploration)
        - Bottom Radius: 0.3 - 1.0 (narrower = more focus)
        - Height: 5.0 - 10.0 (deeper = more processing stages)
        - Turns: 1 - 3 (more turns = more complex path)

        **Agent Settings:**
        - Max Agents: 5 - 25 (based on system resources)
        - Token Budget: 1000 - 4000 (based on LLM context window)
        - Temperature Top: 0.7 - 1.0 (exploration creativity)
        - Temperature Bottom: 0.1 - 0.3 (synthesis precision)

        **Dynamic Spawning:**
        - Confidence Threshold: 0.6 - 0.9 (lower = more spawning)
        - Max Depth: 3 - 7 (prevents infinite recursion)
        - Spawn Delay: 0.1 - 1.0 seconds (prevents overload)
        """)


if __name__ == "__main__":
    main()