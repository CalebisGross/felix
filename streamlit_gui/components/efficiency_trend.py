"""
Workflow Efficiency Trend Component

Shows if Felix is getting better over time (learning curve).
Efficiency is measured as confidence per agent (confidence / agent_count).

Based on historical workflow data from felix_workflow_history.db.
Includes trendline analysis to show improvement direction.
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EfficiencyTrendChart:
    """Analyze and visualize workflow efficiency trends."""

    def __init__(self, db_path: str = "felix_workflow_history.db"):
        """
        Initialize efficiency trend analyzer.

        Args:
            db_path: Path to workflow history database
        """
        self.db_path = Path(db_path)

    def get_workflow_efficiency(self, limit: int = 20) -> pd.DataFrame:
        """
        Get workflow efficiency data for recent workflows.

        Args:
            limit: Number of recent workflows to analyze (default: 20)

        Returns:
            DataFrame with columns:
                - workflow_id: Workflow identifier
                - workflow_num: Sequential workflow number (1, 2, 3, ...)
                - confidence: Workflow confidence score (0-1)
                - agents_count: Number of agents used
                - efficiency: Efficiency score (confidence / agents_count)
                - processing_time: Time to complete
                - created_at: Timestamp
        """
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return pd.DataFrame(columns=[
                'workflow_id', 'workflow_num', 'confidence', 'agents_count',
                'efficiency', 'processing_time', 'created_at'
            ])

        try:
            conn = sqlite3.connect(str(self.db_path))

            # Query recent completed workflows
            query = """
            SELECT
                workflow_id,
                confidence,
                agents_count,
                processing_time,
                created_at
            FROM workflow_outputs
            WHERE status = 'completed'
                AND confidence IS NOT NULL
                AND agents_count > 0
            ORDER BY created_at DESC
            LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()

            if df.empty:
                return df

            # Sort by time (oldest first for proper sequencing)
            df = df.sort_values('created_at')

            # Calculate efficiency: confidence per agent
            # Higher is better (more confidence with fewer agents)
            df['efficiency'] = df['confidence'] / df['agents_count']

            # Add sequential workflow number
            df['workflow_num'] = range(1, len(df) + 1)

            # Reorder columns
            df = df[[
                'workflow_id', 'workflow_num', 'confidence', 'agents_count',
                'efficiency', 'processing_time', 'created_at'
            ]]

            return df

        except Exception as e:
            logger.error(f"Error getting workflow efficiency: {e}")
            return pd.DataFrame(columns=[
                'workflow_id', 'workflow_num', 'confidence', 'agents_count',
                'efficiency', 'processing_time', 'created_at'
            ])

    def calculate_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trend statistics for efficiency data.

        Args:
            df: DataFrame from get_workflow_efficiency()

        Returns:
            Dictionary containing:
                - slope: Trendline slope
                - intercept: Trendline intercept
                - trend_direction: 'improving', 'declining', or 'stable'
                - percent_change: Percentage change from first to last
                - avg_efficiency: Average efficiency score
                - std_efficiency: Standard deviation
        """
        if df.empty or len(df) < 2:
            return {
                'slope': 0,
                'intercept': 0,
                'trend_direction': 'insufficient_data',
                'percent_change': 0,
                'avg_efficiency': 0,
                'std_efficiency': 0
            }

        # Calculate linear regression
        x = df['workflow_num'].values
        y = df['efficiency'].values

        # Use numpy polyfit for linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Determine trend direction (using threshold of 0.001 for stability)
        if slope > 0.001:
            trend_direction = 'improving'
        elif slope < -0.001:
            trend_direction = 'declining'
        else:
            trend_direction = 'stable'

        # Calculate percent change
        first_efficiency = y[0]
        last_efficiency = y[-1]
        percent_change = ((last_efficiency - first_efficiency) / first_efficiency * 100
                         if first_efficiency > 0 else 0)

        return {
            'slope': slope,
            'intercept': intercept,
            'trend_direction': trend_direction,
            'percent_change': percent_change,
            'avg_efficiency': y.mean(),
            'std_efficiency': y.std(),
            'trendline_start': intercept + slope * x[0],
            'trendline_end': intercept + slope * x[-1]
        }


def create_efficiency_chart(df: pd.DataFrame, trend_stats: Dict[str, Any]) -> go.Figure:
    """
    Create efficiency trend visualization.

    Args:
        df: DataFrame with efficiency data
        trend_stats: Trend statistics from calculate_trend()

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    if df.empty:
        # Empty state
        fig.add_annotation(
            text="No workflow data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Main efficiency line with markers
    fig.add_trace(go.Scatter(
        x=df['workflow_num'],
        y=df['efficiency'],
        mode='lines+markers',
        name='Efficiency',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, color='#2563eb', line=dict(width=2, color='white')),
        hovertemplate=(
            '<b>Workflow %{x}</b><br>' +
            'Efficiency: %{y:.3f}<br>' +
            'Confidence: %{customdata[0]:.2f}<br>' +
            'Agents: %{customdata[1]}<br>' +
            '<extra></extra>'
        ),
        customdata=df[['confidence', 'agents_count']].values
    ))

    # Add trendline
    if len(df) >= 2 and trend_stats['slope'] != 0:
        x_trend = df['workflow_num'].values
        y_trend = trend_stats['intercept'] + trend_stats['slope'] * x_trend

        # Color based on trend direction
        if trend_stats['trend_direction'] == 'improving':
            trend_color = '#22c55e'  # Green
            trend_name = 'Trend (Improving)'
        elif trend_stats['trend_direction'] == 'declining':
            trend_color = '#ef4444'  # Red
            trend_name = 'Trend (Declining)'
        else:
            trend_color = '#6b7280'  # Gray
            trend_name = 'Trend (Stable)'

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name=trend_name,
            line=dict(color=trend_color, width=2, dash='dash'),
            hovertemplate='Trendline: %{y:.3f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': "Workflow Efficiency Trend: Learning Over Time",
            'font': {'size': 18, 'color': '#ffffff'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Workflow Number",
            tickmode='linear',
            tick0=1,
            dtick=1 if len(df) <= 20 else 2,
            gridcolor='rgba(255,255,255,0.1)',
            color='#ffffff'
        ),
        yaxis=dict(
            title="Efficiency Score (Confidence / Agents)",
            gridcolor='rgba(255,255,255,0.1)',
            color='#ffffff'
        ),
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='#ffffff')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff')
    )

    return fig


def render_efficiency_trend(
    db_path: str = "felix_workflow_history.db",
    limit: int = 20,
    show_statistics: bool = True
) -> None:
    """
    Render workflow efficiency trend visualization.

    Args:
        db_path: Path to workflow history database
        limit: Number of recent workflows to analyze (default: 20)
        show_statistics: Whether to show detailed statistics
    """
    analyzer = EfficiencyTrendChart(db_path)
    df = analyzer.get_workflow_efficiency(limit)

    if df.empty:
        st.warning(f"No workflow data available for efficiency analysis")
        st.info("Efficiency trends will appear once Felix completes multiple workflows.")
        return

    # Calculate trend statistics
    trend_stats = analyzer.calculate_trend(df)

    # Display interpretation banner
    if trend_stats['trend_direction'] == 'improving':
        st.success(
            f"✅ System improving over time "
            f"(+{trend_stats['percent_change']:.1f}% efficiency gain)"
        )
    elif trend_stats['trend_direction'] == 'declining':
        st.error(
            f"⚠️ Efficiency declining "
            f"({trend_stats['percent_change']:.1f}% decrease)"
        )
    elif trend_stats['trend_direction'] == 'stable':
        st.info("→ Stable performance (no significant trend)")
    else:
        st.warning("Insufficient data for trend analysis")

    # Main chart
    fig = create_efficiency_chart(df, trend_stats)
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed statistics
    if show_statistics and not df.empty:
        st.markdown("### Efficiency Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Workflows Analyzed",
                len(df),
                help="Number of completed workflows in analysis"
            )

        with col2:
            st.metric(
                "Avg Efficiency",
                f"{trend_stats['avg_efficiency']:.3f}",
                help="Mean efficiency score across all workflows"
            )

        with col3:
            current_efficiency = df['efficiency'].iloc[-1]
            first_efficiency = df['efficiency'].iloc[0]
            delta = current_efficiency - first_efficiency

            st.metric(
                "Current Efficiency",
                f"{current_efficiency:.3f}",
                delta=f"{delta:+.3f}",
                help="Most recent workflow efficiency"
            )

        with col4:
            st.metric(
                "Trend Slope",
                f"{trend_stats['slope']:+.4f}",
                help="Rate of efficiency change per workflow"
            )

        # Additional insights
        with st.expander("📊 Detailed Insights"):
            st.markdown("**What is Efficiency?**")
            st.markdown(
                "Efficiency = Confidence ÷ Agent Count. "
                "Higher scores mean Felix achieves better results with fewer agents."
            )

            st.markdown("**Interpretation:**")
            if trend_stats['trend_direction'] == 'improving':
                st.markdown(
                    "- Felix is learning to work more efficiently over time\n"
                    "- Recent workflows achieve higher confidence with fewer agents\n"
                    "- System optimization is working as expected"
                )
            elif trend_stats['trend_direction'] == 'declining':
                st.markdown(
                    "- Recent workflows are less efficient\n"
                    "- May indicate: more complex tasks, or agent spawning issues\n"
                    "- Consider reviewing recent workflow configurations"
                )
            else:
                st.markdown(
                    "- Performance is consistent across workflows\n"
                    "- System has reached stable operation\n"
                    "- Efficiency is neither improving nor declining"
                )

            # Show data table
            st.markdown("**Recent Workflows:**")
            display_df = df[['workflow_num', 'workflow_id', 'confidence',
                            'agents_count', 'efficiency']].copy()
            display_df['confidence'] = display_df['confidence'].round(3)
            display_df['efficiency'] = display_df['efficiency'].round(3)

            st.dataframe(
                display_df.tail(10),
                use_container_width=True,
                hide_index=True
            )


# Example usage and demo
if __name__ == "__main__":
    st.set_page_config(page_title="Efficiency Trend Demo", layout="wide")

    st.title("Workflow Efficiency Trend Analysis")

    st.markdown("""
    This component shows if Felix is **learning and improving over time**.

    **Efficiency Metric:**
    - Formula: `Confidence ÷ Agent Count`
    - Higher = Better (more confidence with fewer agents)
    - Positive trend = System is learning
    - Negative trend = Performance declining
    - Flat trend = Stable operation

    **Why This Matters:**
    - Validates adaptive agent spawning
    - Shows system optimization working
    - Identifies performance regressions early
    """)

    st.markdown("---")

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Workflows to analyze", 5, 50, 20)
    with col2:
        show_stats = st.checkbox("Show detailed statistics", value=True)

    st.markdown("---")

    # Render component
    render_efficiency_trend(
        db_path="felix_workflow_history.db",
        limit=limit,
        show_statistics=show_stats
    )
