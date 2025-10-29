"""Phase Performance Breakdown Component for Felix Streamlit GUI.

This component identifies bottlenecks in workflow progression by showing
performance metrics grouped by helix phase (Exploration, Analysis, Synthesis).
"""

import sqlite3
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PhasePerformanceAnalyzer:
    """Analyzes agent performance metrics by helix phase."""

    def __init__(self, db_path: str):
        """
        Initialize the analyzer.

        Args:
            db_path: Path to felix_agent_performance.db
        """
        self.db_path = db_path

    def get_phase_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Query agent performance by helix phase.

        Returns:
            Dictionary mapping phase names to metrics:
            {
                'Exploration': {
                    'avg_confidence': float,
                    'avg_tokens_used': float,
                    'avg_processing_time': float,
                    'agent_count': int
                },
                ...
            }
        """
        if not Path(self.db_path).exists():
            logger.warning(f"Database not found: {self.db_path}")
            return self._empty_metrics()

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Query grouped by phase
            query = """
                SELECT
                    phase,
                    AVG(confidence) as avg_confidence,
                    AVG(tokens_used) as avg_tokens,
                    AVG(processing_time) as avg_time,
                    COUNT(*) as count
                FROM agent_performance
                GROUP BY phase
            """

            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()

            if not results:
                logger.info("No agent performance data found")
                return self._empty_metrics()

            # Build metrics dictionary
            metrics = {}
            for row in results:
                phase, avg_conf, avg_tokens, avg_time, count = row

                # Map phase names if they don't match expected format
                phase_name = self._normalize_phase_name(phase)

                metrics[phase_name] = {
                    'avg_confidence': float(avg_conf) if avg_conf else 0.0,
                    'avg_tokens_used': float(avg_tokens) if avg_tokens else 0.0,
                    'avg_processing_time': float(avg_time) if avg_time else 0.0,
                    'agent_count': int(count) if count else 0
                }

            # Ensure all phases are present
            for phase in ['Exploration', 'Analysis', 'Synthesis']:
                if phase not in metrics:
                    metrics[phase] = {
                        'avg_confidence': 0.0,
                        'avg_tokens_used': 0.0,
                        'avg_processing_time': 0.0,
                        'agent_count': 0
                    }

            return metrics

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return self._empty_metrics()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self._empty_metrics()

    def _normalize_phase_name(self, phase: str) -> str:
        """
        Normalize phase name to standard format.

        Args:
            phase: Raw phase name from database

        Returns:
            Standardized phase name
        """
        phase_lower = phase.lower() if phase else ''

        if 'expl' in phase_lower or 'research' in phase_lower:
            return 'Exploration'
        elif 'anal' in phase_lower or 'process' in phase_lower:
            return 'Analysis'
        elif 'synth' in phase_lower or 'final' in phase_lower:
            return 'Synthesis'

        # Default mapping based on phase string
        return phase.capitalize() if phase else 'Unknown'

    def _empty_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Return empty metrics structure."""
        return {
            'Exploration': {
                'avg_confidence': 0.0,
                'avg_tokens_used': 0.0,
                'avg_processing_time': 0.0,
                'agent_count': 0
            },
            'Analysis': {
                'avg_confidence': 0.0,
                'avg_tokens_used': 0.0,
                'avg_processing_time': 0.0,
                'agent_count': 0
            },
            'Synthesis': {
                'avg_confidence': 0.0,
                'avg_tokens_used': 0.0,
                'avg_processing_time': 0.0,
                'agent_count': 0
            }
        }

    def detect_bottleneck(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify the phase with lowest confidence as a bottleneck.

        Args:
            metrics: Phase metrics dictionary

        Returns:
            Dictionary with bottleneck detection results:
            {
                'phase': str,
                'confidence': float,
                'recommendation': str
            }
        """
        if not metrics:
            return {
                'phase': None,
                'confidence': 0.0,
                'recommendation': 'No data available for bottleneck detection'
            }

        # Find phase with lowest confidence (excluding zero counts)
        valid_phases = {
            phase: data for phase, data in metrics.items()
            if data['agent_count'] > 0
        }

        if not valid_phases:
            return {
                'phase': None,
                'confidence': 0.0,
                'recommendation': 'No agent activity detected'
            }

        bottleneck_phase = min(
            valid_phases.keys(),
            key=lambda p: valid_phases[p]['avg_confidence']
        )
        bottleneck_confidence = valid_phases[bottleneck_phase]['avg_confidence']

        # Generate recommendation
        recommendations = {
            'Exploration': 'Consider adding more research agents to broaden the initial investigation',
            'Analysis': 'Increase token budget for analysis agents to allow deeper processing',
            'Synthesis': 'Review synthesis temperature settings and ensure adequate context compression'
        }

        return {
            'phase': bottleneck_phase,
            'confidence': bottleneck_confidence,
            'recommendation': recommendations.get(
                bottleneck_phase,
                'Review phase configuration and agent allocation'
            )
        }


def create_phase_performance_chart(metrics: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create grouped bar chart for phase performance visualization.

    Args:
        metrics: Phase metrics from PhasePerformanceAnalyzer

    Returns:
        Plotly figure object
    """
    phases = ['Exploration', 'Analysis', 'Synthesis']

    # Extract and normalize values
    confidences = []
    tokens = []
    times = []

    # Get max values for normalization
    max_tokens = max(
        [metrics[p]['avg_tokens_used'] for p in phases if metrics[p]['avg_tokens_used'] > 0],
        default=1.0
    )
    max_time = max(
        [metrics[p]['avg_processing_time'] for p in phases if metrics[p]['avg_processing_time'] > 0],
        default=1.0
    )

    for phase in phases:
        data = metrics[phase]
        confidences.append(data['avg_confidence'])

        # Normalize tokens and time to 0-1 scale
        tokens.append(data['avg_tokens_used'] / max_tokens if max_tokens > 0 else 0)
        times.append(data['avg_processing_time'] / max_time if max_time > 0 else 0)

    # Create grouped bar chart
    fig = go.Figure()

    # Add confidence bars
    fig.add_trace(go.Bar(
        name='Avg Confidence',
        x=phases,
        y=confidences,
        marker_color='#2ecc71',  # Green
        text=[f'{v:.2f}' for v in confidences],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>'
    ))

    # Add normalized tokens bars
    fig.add_trace(go.Bar(
        name=f'Avg Tokens (norm)',
        x=phases,
        y=tokens,
        marker_color='#3498db',  # Blue
        text=[f'{v:.2f}' for v in tokens],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Normalized Tokens: %{y:.3f}<extra></extra>'
    ))

    # Add normalized time bars
    fig.add_trace(go.Bar(
        name=f'Avg Time (norm)',
        x=phases,
        y=times,
        marker_color='#e67e22',  # Orange
        text=[f'{v:.2f}' for v in times],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Normalized Time: %{y:.3f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Performance Breakdown by Helix Phase',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title='Helix Phase',
        yaxis_title='Normalized Values (0-1 scale)',
        yaxis={'range': [0, 1.2]},  # Extra space for text labels
        barmode='group',
        height=500,
        hovermode='x unified',
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif', 'size': 12}
    )

    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def render_phase_performance(db_path: str):
    """
    Render the complete phase performance component in Streamlit.

    Args:
        db_path: Path to felix_agent_performance.db
    """
    st.subheader('Performance Breakdown by Helix Phase')
    st.caption(
        'Analyze agent performance across the three helix phases to identify '
        'bottlenecks and optimization opportunities.'
    )

    # Initialize analyzer
    analyzer = PhasePerformanceAnalyzer(db_path)

    # Get metrics
    with st.spinner('Analyzing phase performance...'):
        metrics = analyzer.get_phase_metrics()

    # Check if we have data
    total_agents = sum(m['agent_count'] for m in metrics.values())

    if total_agents == 0:
        st.info(
            'No agent performance data available yet. '
            'Run a workflow to collect performance metrics.'
        )
        return

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Total Agents', total_agents)

    with col2:
        avg_conf = sum(
            m['avg_confidence'] * m['agent_count'] for m in metrics.values()
        ) / total_agents if total_agents > 0 else 0
        st.metric('Avg Confidence', f'{avg_conf:.3f}')

    with col3:
        total_tokens = sum(
            m['avg_tokens_used'] * m['agent_count'] for m in metrics.values()
        )
        st.metric('Total Tokens', f'{int(total_tokens):,}')

    with col4:
        avg_time = sum(
            m['avg_processing_time'] * m['agent_count'] for m in metrics.values()
        ) / total_agents if total_agents > 0 else 0
        st.metric('Avg Time (s)', f'{avg_time:.2f}')

    # Display phase breakdown table
    st.markdown('---')
    st.markdown('#### Phase Details')

    phase_df = pd.DataFrame([
        {
            'Phase': phase,
            'Agent Count': data['agent_count'],
            'Avg Confidence': f"{data['avg_confidence']:.3f}",
            'Avg Tokens': f"{data['avg_tokens_used']:.1f}",
            'Avg Time (s)': f"{data['avg_processing_time']:.2f}"
        }
        for phase, data in metrics.items()
    ])

    st.dataframe(
        phase_df,
        use_container_width=True,
        hide_index=True
    )

    # Create and display chart
    st.markdown('---')
    fig = create_phase_performance_chart(metrics)
    st.plotly_chart(fig, use_container_width=True)

    # Detect and display bottleneck
    st.markdown('---')
    bottleneck = analyzer.detect_bottleneck(metrics)

    if bottleneck['phase']:
        st.warning(f"⚠️ **Bottleneck Detected in {bottleneck['phase']} Phase**")
        st.info(f"**Recommendation:** {bottleneck['recommendation']}")

        # Additional context
        with st.expander('Understanding Phase Bottlenecks'):
            st.markdown("""
            **Phase Characteristics:**

            - **Exploration Phase** (Depth 0.0-0.33): Agents broadly investigate the problem space
              - Low confidence here suggests insufficient research or unclear task framing
              - Solution: Add more research agents or improve initial context

            - **Analysis Phase** (Depth 0.33-0.66): Agents process and synthesize information
              - Low confidence indicates inadequate processing depth or token constraints
              - Solution: Increase token budgets or add specialized analysis agents

            - **Synthesis Phase** (Depth 0.66-1.0): Final integration and output generation
              - Low confidence suggests poor context compression or temperature settings
              - Solution: Review synthesis configuration and memory management

            **Optimal Performance Profile:**
            - Confidence should increase progressively from Exploration → Synthesis
            - Token usage typically peaks in Analysis phase
            - Processing time should be balanced across phases
            """)
    else:
        st.success('✅ No significant bottlenecks detected. Performance is well-balanced across phases.')


if __name__ == '__main__':
    """Example usage and testing."""
    import sys
    import os

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Test with actual database
    db_path = project_root / 'felix_agent_performance.db'

    print("Phase Performance Analyzer - Standalone Test")
    print("=" * 60)

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Creating sample data for demonstration...")

        # Create sample database for testing
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                workflow_id INTEGER,
                agent_type TEXT NOT NULL,
                spawn_time REAL NOT NULL,
                checkpoint REAL NOT NULL,
                confidence REAL NOT NULL,
                tokens_used INTEGER NOT NULL,
                processing_time REAL NOT NULL,
                depth_ratio REAL NOT NULL,
                phase TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)

        # Insert sample data
        import random
        import time

        phases = ['Exploration', 'Analysis', 'Synthesis']
        agent_types = ['Research', 'Analysis', 'Critic']

        for i in range(30):
            phase = random.choice(phases)
            cursor.execute("""
                INSERT INTO agent_performance
                (agent_id, workflow_id, agent_type, spawn_time, checkpoint,
                 confidence, tokens_used, processing_time, depth_ratio, phase, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f'agent_{i}',
                1,
                random.choice(agent_types),
                time.time() - 1000 + i * 10,
                random.random(),
                random.uniform(0.6, 0.95),
                random.randint(500, 2500),
                random.uniform(0.5, 5.0),
                random.random(),
                phase,
                time.time() - 1000 + i * 10
            ))

        conn.commit()
        conn.close()
        print("Sample database created.")

    # Analyze metrics
    analyzer = PhasePerformanceAnalyzer(str(db_path))
    metrics = analyzer.get_phase_metrics()

    print("\nPhase Metrics:")
    print("-" * 60)
    for phase, data in metrics.items():
        print(f"\n{phase}:")
        print(f"  Agent Count: {data['agent_count']}")
        print(f"  Avg Confidence: {data['avg_confidence']:.3f}")
        print(f"  Avg Tokens: {data['avg_tokens_used']:.1f}")
        print(f"  Avg Time: {data['avg_processing_time']:.2f}s")

    # Detect bottleneck
    print("\nBottleneck Analysis:")
    print("-" * 60)
    bottleneck = analyzer.detect_bottleneck(metrics)
    if bottleneck['phase']:
        print(f"Phase: {bottleneck['phase']}")
        print(f"Confidence: {bottleneck['confidence']:.3f}")
        print(f"Recommendation: {bottleneck['recommendation']}")
    else:
        print("No bottlenecks detected")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
