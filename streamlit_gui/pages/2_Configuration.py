"""Configuration page for Felix Framework."""

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
    # Handle special default config marker
    if file_path == "<defaults>":
        return get_default_felix_config()

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


def get_default_felix_config() -> dict:
    """
    Return default Felix configuration based on CLAUDE.md documentation.

    Returns:
        Default configuration dictionary
    """
    return {
        # Helix geometry
        'helix_top_radius': 3.0,
        'helix_bottom_radius': 0.5,
        'helix_height': 8.0,
        'helix_turns': 2,

        # Agent configuration
        'max_agents': 10,
        'base_token_budget': 2048,
        'temperature_top': 1.0,
        'temperature_bottom': 0.2,

        # Dynamic spawning
        'confidence_threshold': 0.80,
        'enable_dynamic_spawning': True,
        'volatility_threshold': 0.15,
        'time_window_minutes': 5.0,

        # LM Studio connection
        'lm_host': '127.0.0.1',
        'lm_port': 1234,
        'model': 'default',
        'enable_streaming': True,

        # Memory and compression
        'compression_ratio': 0.3,
        'compression_strategy': 'abstractive',
        'compression_target_length': 100,
        'enable_compression': True,
        'enable_memory': True,

        # Web search
        'web_search_enabled': False,
        'web_search_provider': 'duckduckgo',
        'web_search_max_results': 5,
        'web_search_confidence_threshold': 0.7,

        # Feature toggles
        'enable_metrics': True,
        'enable_spoke_topology': True,
        'verbose_llm_logging': False
    }


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

    # Generate helix points using correct Felix geometry
    # t ∈ [0,1] where t=0 is top/wide (exploration), t=1 is bottom/narrow (synthesis)
    t_values = np.linspace(0, 1, 200)

    # Calculate positions along helix
    # z = height * (1 - t), so agents start at z=height (top) and descend to z=0 (bottom)
    z = height * (1.0 - t_values)

    # Calculate radius at each position using exponential tapering
    # R(z) = R_bottom * (R_top/R_bottom)^(z/height)
    radius_ratio = top_radius / bottom_radius
    height_fraction = z / height
    r = bottom_radius * np.power(radius_ratio, height_fraction)

    # Calculate angle: θ(t) = 2πnt
    angles = t_values * turns * 2.0 * np.pi

    # Generate x, y coordinates
    x = r * np.cos(angles)
    y = r * np.sin(angles)

    # Create the helix trace
    helix_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        name='Helix Path',
        line=dict(
            color=t_values,  # Color by progression (0=top/exploration, 1=bottom/synthesis)
            colorscale='Viridis',
            width=8,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="Progression<br>(0=Top/Exploration<br>1=Bottom/Synthesis)",
                x=1.02,
                y=0.5,
                thickness=15,
                len=0.7
            )
        )
    )

    # Add CentralPost hub visualization (vertical line at center)
    hub_trace = go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, height],
        mode='lines',
        name='CentralPost Hub',
        line=dict(
            color='red',
            width=12,
            dash='solid'
        ),
        showlegend=True
    )

    # Add reference points for phases (with correct z positions)
    phase_points = []
    spoke_lines = []

    # Phase definitions: (label, t_value, color)
    # t=0 is top (z=height), t=1 is bottom (z=0)
    phase_definitions = [
        ("Exploration Start", 0.0, "green"),
        ("Analysis Phase", 0.4, "blue"),
        ("Synthesis Phase", 0.7, "orange"),
        ("Conclusion", 1.0, "red")
    ]

    for label, t_val, color in phase_definitions:
        # Calculate position on helix at this t value
        z_pos = height * (1.0 - t_val)

        # Calculate radius using exponential tapering
        height_frac = z_pos / height
        radius = bottom_radius * pow(radius_ratio, height_frac)

        # Calculate angle
        angle = t_val * turns * 2.0 * np.pi
        x_pos = radius * np.cos(angle)
        y_pos = radius * np.sin(angle)

        # Add phase marker
        phase_points.append(
            go.Scatter3d(
                x=[x_pos],
                y=[y_pos],
                z=[z_pos],
                mode='markers+text',
                name=label,
                marker=dict(size=10, color=color, symbol='diamond'),
                text=[label],
                textposition="top center",
                textfont=dict(size=10),
                showlegend=False
            )
        )

        # Add spoke line from center to agent position (hub-spoke communication)
        spoke_lines.append(
            go.Scatter3d(
                x=[0, x_pos],
                y=[0, y_pos],
                z=[z_pos, z_pos],
                mode='lines',
                line=dict(
                    color=color,
                    width=2,
                    dash='dot'
                ),
                showlegend=False,
                hoverinfo='skip'
            )
        )

    # Create figure with all traces
    fig = go.Figure(data=[helix_trace, hub_trace] + phase_points + spoke_lines)

    # Update layout
    fig.update_layout(
        title=dict(
            text="Felix Helix Geometry: Hub-Spoke Architecture<br><sub>Agents descend from wide exploration (top) to narrow synthesis (bottom)</sub>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height (Agents descend from top→bottom)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),  # Better viewing angle
                center=dict(x=0, y=0, z=0.4)
            ),
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
    # Real data indicator badge - MUST be at the very top before anything else
    st.success("✅ **Real Data**: This page displays actual Felix configuration files from your project.")

    st.title("⚙️ Configuration Viewer")
    st.markdown("View and export Felix configuration (read-only)")

    # Configuration sources - only list files that exist
    # Build path from current script location
    project_root = Path(__file__).parent.parent.parent

    config_sources = {}
    potential_configs = {
        "felix_gui_config.json": "Felix GUI Configuration (Main)",
    }

    # Only add configs that exist
    for file_path, description in potential_configs.items():
        full_path = project_root / file_path
        if full_path.exists():
            config_sources[str(full_path)] = description

    # If no configs found, show default
    if not config_sources:
        st.warning("⚠️ No configuration files found. Using default Felix parameters.")
        config_sources = {"<defaults>": "Default Felix Configuration"}

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📋 View Config", "📊 Compare", "💾 Export"])

    with tab1:
        st.subheader("Current Configuration")

        # Select config source
        selected_source = st.selectbox(
            "Configuration Source",
            options=list(config_sources.keys()),
            format_func=lambda x: config_sources[x],
            key="config_source_selector"
        )

        # Load configuration
        config = load_config_file(selected_source)

        # Store in session state for reference
        if 'current_config' not in st.session_state or st.session_state.get('last_source') != selected_source:
            st.session_state['current_config'] = config
            st.session_state['last_source'] = selected_source

        if config:
            # Display configuration sections in organized layout
            col1, col2 = st.columns(2)

            with col1:
                # Helix Geometry
                st.markdown("### 🌀 Helix Geometry")
                helix_config = {
                    'top_radius': config.get('helix_top_radius', config.get('helix', {}).get('top_radius', 3.0)),
                    'bottom_radius': config.get('helix_bottom_radius', config.get('helix', {}).get('bottom_radius', 0.5)),
                    'height': config.get('helix_height', config.get('helix', {}).get('height', 8.0)),
                    'turns': config.get('helix_turns', config.get('helix', {}).get('turns', 2))
                }

                hcol1, hcol2 = st.columns(2)
                with hcol1:
                    st.metric("Top Radius", f"{helix_config['top_radius']:.1f}", help="Wide exploration phase breadth")
                    st.metric("Height", f"{helix_config['height']:.1f}", help="Total progression depth")
                with hcol2:
                    st.metric("Bottom Radius", f"{helix_config['bottom_radius']:.1f}", help="Narrow synthesis focus")
                    st.metric("Turns", int(helix_config['turns']), help="Spiral complexity")

                # Agent Configuration
                st.markdown("### 🤖 Agent Configuration")
                acol1, acol2 = st.columns(2)
                with acol1:
                    max_agents = config.get('max_agents', 25)
                    st.metric("Max Agents", max_agents, help="Team size limit")
                    temp_top = config.get('temperature_top', config.get('temperature', {}).get('top', 1.0))
                    st.metric("Temp (Exploration)", f"{temp_top:.2f}", help="Temperature at helix top")
                with acol2:
                    token_budget = config.get('base_token_budget', 2048)
                    st.metric("Base Token Budget", token_budget, help="Per-agent token allocation")
                    temp_bottom = config.get('temperature_bottom', config.get('temperature', {}).get('bottom', 0.2))
                    st.metric("Temp (Synthesis)", f"{temp_bottom:.2f}", help="Temperature at helix bottom")

                # Memory & Compression
                st.markdown("### 💾 Memory & Compression")
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    comp_ratio = config.get('compression_ratio', 0.3)
                    st.metric("Compression Ratio", f"{comp_ratio:.1f}", help="Context compression ratio")
                    comp_enabled = config.get('enable_compression', True)
                    st.metric("Compression", "Enabled" if comp_enabled else "Disabled")
                with mcol2:
                    comp_strategy = config.get('compression_strategy', 'abstractive')
                    st.metric("Strategy", comp_strategy.title(), help="Compression approach")
                    comp_target = config.get('compression_target_length', 100)
                    st.metric("Target Length", comp_target, help="Compressed context length")

            with col2:
                # LM Studio Connection
                st.markdown("### 🔌 LM Studio Connection")
                lcol1, lcol2 = st.columns(2)
                with lcol1:
                    lm_host = config.get('lm_host', '127.0.0.1')
                    st.metric("Host", lm_host)
                    model_name = config.get('model', 'default')
                    st.metric("Model", model_name)
                with lcol2:
                    lm_port = config.get('lm_port', 1234)
                    st.metric("Port", lm_port)
                    streaming = config.get('enable_streaming', True)
                    st.metric("Streaming", "Enabled" if streaming else "Disabled")

                # Dynamic Spawning
                st.markdown("### 🔄 Dynamic Spawning")
                scol1, scol2 = st.columns(2)
                with scol1:
                    conf_threshold = config.get('confidence_threshold', config.get('spawning', {}).get('confidence_threshold', 0.80))
                    st.metric("Confidence Threshold", f"{conf_threshold:.2f}", help="Trigger for dynamic spawning")
                    spawn_enabled = config.get('enable_dynamic_spawning', True)
                    st.metric("Dynamic Spawning", "Enabled" if spawn_enabled else "Disabled")
                with scol2:
                    volatility = config.get('volatility_threshold', 0.15)
                    st.metric("Volatility Threshold", f"{volatility:.2f}", help="Confidence variance limit")
                    time_window = config.get('time_window_minutes', 5.0)
                    st.metric("Time Window", f"{time_window:.0f} min", help="Metric aggregation period")

                # Web Search Configuration
                st.markdown("### 🔍 Web Search")
                wcol1, wcol2 = st.columns(2)
                with wcol1:
                    web_enabled = config.get('web_search_enabled', True)
                    st.metric("Web Search", "Enabled" if web_enabled else "Disabled")
                    web_max_results = config.get('web_search_max_results', 5)
                    st.metric("Max Results", web_max_results)
                with wcol2:
                    web_provider = config.get('web_search_provider', 'duckduckgo')
                    st.metric("Provider", web_provider.title())
                    web_confidence = config.get('web_search_confidence_threshold', 0.7)
                    st.metric("Min Confidence", f"{web_confidence:.2f}")

            # Web Search Configuration Details
            st.divider()
            st.markdown("### 🔍 Web Search Configuration Details")

            wscol1, wscol2 = st.columns(2)

            with wscol1:
                # Status and max results
                web_enabled = config.get('web_search_enabled', False)
                st.metric(
                    "Status",
                    "Enabled" if web_enabled else "Disabled",
                    help="Whether web search is active during workflows"
                )

                web_max_results = config.get('web_search_max_results', 5)
                st.metric(
                    "Max Results per Query",
                    web_max_results,
                    help="Maximum search results to retrieve per query"
                )

                # Blocked domains
                blocked_domains = config.get('web_search_blocked_domains', '')
                if blocked_domains:
                    domain_list = [d.strip() for d in blocked_domains.split('\n') if d.strip()]
                    st.metric(
                        "Blocked Domains",
                        len(domain_list),
                        help="Number of domains filtered from results"
                    )
                    with st.expander("View Blocked Domains"):
                        for domain in domain_list:
                            st.text(f"• {domain}")
                else:
                    st.metric("Blocked Domains", 0, help="No domains blocked")

            with wscol2:
                # Provider
                web_provider = config.get('web_search_provider', 'duckduckgo')
                st.metric(
                    "Search Provider",
                    web_provider.title(),
                    help="Search engine backend"
                )

                # Max queries
                web_max_queries = config.get('web_search_max_queries', 3)
                st.metric(
                    "Max Queries per Workflow",
                    web_max_queries,
                    help="Maximum number of search queries per task"
                )

                # Confidence threshold
                web_conf_threshold = config.get('web_search_confidence_threshold', 0.7)
                st.metric(
                    "Confidence Threshold",
                    f"{web_conf_threshold * 100:.0f}%",
                    help="Minimum confidence to trigger web search"
                )

            # SearxNG URL if applicable
            if web_provider.lower() == 'searxng':
                searxng_url = config.get('searxng_url', '')
                if searxng_url:
                    st.metric(
                        "SearxNG Instance URL",
                        searxng_url,
                        help="Custom SearxNG server endpoint"
                    )
                else:
                    st.warning("⚠️ SearxNG selected but no URL configured")

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