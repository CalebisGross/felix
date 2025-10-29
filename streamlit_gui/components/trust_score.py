"""
System Autonomy Trust Score Component

Calculates and displays a trust metric (0-100) that answers:
"Should I trust Felix with system commands?"

Based on historical command execution data from felix_system_actions.db.
Trust score formula: (success_rate * 70) + ((1 - denial_rate) * 30)
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrustScoreCalculator:
    """Calculate system autonomy trust score from execution history."""

    def __init__(self, db_path: str = "felix_system_actions.db"):
        """
        Initialize trust score calculator.

        Args:
            db_path: Path to system actions database
        """
        self.db_path = Path(db_path)

    def calculate_trust_score(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate trust score based on recent execution history.

        Args:
            days: Number of days to analyze (default: 7)

        Returns:
            Dictionary containing:
                - trust_score: Float 0-100
                - success_rate: Float 0-1
                - denial_rate: Float 0-1
                - total_requests: Int
                - successful: Int
                - failed: Int
                - denied: Int
                - data_available: Bool
        """
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
            return self._empty_result()

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Calculate cutoff timestamp (N days ago)
            cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()

            # Query execution statistics
            query = """
            SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN executed = 1 AND success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN executed = 1 AND success = 0 THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as total_executed
            FROM command_executions
            WHERE timestamp >= ?
            """

            cursor.execute(query, (cutoff_time,))
            row = cursor.fetchone()

            if not row or row[0] == 0:
                conn.close()
                return self._empty_result()

            total_requests = row[0] or 0
            successful = row[1] or 0
            failed = row[2] or 0
            total_executed = row[3] or 0
            denied = total_requests - total_executed

            # Calculate rates
            success_rate = successful / total_executed if total_executed > 0 else 0
            denial_rate = denied / total_requests if total_requests > 0 else 0

            # Trust score formula: weight success heavily (70%) and non-denial moderately (30%)
            trust_score = (success_rate * 70) + ((1 - denial_rate) * 30)

            conn.close()

            return {
                'trust_score': trust_score,
                'success_rate': success_rate,
                'denial_rate': denial_rate,
                'total_requests': total_requests,
                'successful': successful,
                'failed': failed,
                'denied': denied,
                'total_executed': total_executed,
                'data_available': True
            }

        except Exception as e:
            logger.error(f"Error calculating trust score: {e}")
            return self._empty_result()

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'trust_score': 0.0,
            'success_rate': 0.0,
            'denial_rate': 0.0,
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'denied': 0,
            'total_executed': 0,
            'data_available': False
        }

    def get_trust_history(self, days: int = 30, bucket_hours: int = 24) -> pd.DataFrame:
        """
        Get trust score history over time.

        Args:
            days: Number of days to retrieve
            bucket_hours: Hours per data point (default: 24 for daily)

        Returns:
            DataFrame with columns: timestamp, trust_score, success_rate, total_commands
        """
        if not self.db_path.exists():
            return pd.DataFrame(columns=['timestamp', 'trust_score', 'success_rate', 'total_commands'])

        try:
            conn = sqlite3.connect(str(self.db_path))

            cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()

            query = f"""
            SELECT
                datetime(timestamp, 'unixepoch', 'localtime') as time_bucket,
                COUNT(*) as total_commands,
                SUM(CASE WHEN executed = 1 AND success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as total_executed
            FROM command_executions
            WHERE timestamp >= ?
            GROUP BY strftime('%Y-%m-%d %H', datetime(timestamp, 'unixepoch', 'localtime'))
            ORDER BY time_bucket ASC
            """

            df = pd.read_sql_query(query, conn, params=(cutoff_time,))
            conn.close()

            if df.empty:
                return pd.DataFrame(columns=['timestamp', 'trust_score', 'success_rate', 'total_commands'])

            # Calculate metrics for each bucket
            df['success_rate'] = df.apply(
                lambda row: row['successful'] / row['total_executed'] if row['total_executed'] > 0 else 0,
                axis=1
            )

            df['denial_rate'] = df.apply(
                lambda row: (row['total_commands'] - row['total_executed']) / row['total_commands']
                if row['total_commands'] > 0 else 0,
                axis=1
            )

            df['trust_score'] = (df['success_rate'] * 70) + ((1 - df['denial_rate']) * 30)

            # Clean up for output
            df = df.rename(columns={'time_bucket': 'timestamp'})
            df = df[['timestamp', 'trust_score', 'success_rate', 'total_commands']]

            return df

        except Exception as e:
            logger.error(f"Error getting trust history: {e}")
            return pd.DataFrame(columns=['timestamp', 'trust_score', 'success_rate', 'total_commands'])


def create_trust_gauge(trust_score: float, success_count: int, total_executed: int) -> go.Figure:
    """
    Create a gauge chart for trust score visualization.

    Args:
        trust_score: Trust score value (0-100)
        success_count: Number of successful commands
        total_executed: Total executed commands

    Returns:
        Plotly figure object
    """
    # Determine gauge color based on thresholds
    if trust_score >= 80:
        bar_color = "#22c55e"  # Green
        threshold_color = "#16a34a"
    elif trust_score >= 60:
        bar_color = "#eab308"  # Yellow
        threshold_color = "#ca8a04"
    else:
        bar_color = "#ef4444"  # Red
        threshold_color = "#dc2626"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "System Autonomy Trust Score",
            'font': {'size': 20, 'color': '#ffffff'}
        },
        number={'suffix': "", 'font': {'size': 48, 'color': '#ffffff'}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#ffffff",
                'tickfont': {'color': '#ffffff'}
            },
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#ffffff",
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.2)"},  # Red zone
                {'range': [60, 80], 'color': "rgba(234, 179, 8, 0.2)"},  # Yellow zone
                {'range': [80, 100], 'color': "rgba(34, 197, 94, 0.2)"}  # Green zone
            ],
            'threshold': {
                'line': {'color': threshold_color, 'width': 4},
                'thickness': 0.75,
                'value': trust_score
            }
        }
    ))

    # Update layout for dark theme compatibility
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 14}
    )

    return fig


def render_trust_score(
    db_path: str = "felix_system_actions.db",
    days: int = 7,
    show_history: bool = False
) -> None:
    """
    Render trust score visualization and metrics.

    Args:
        db_path: Path to system actions database
        days: Number of days to analyze (default: 7)
        show_history: Whether to show trust score history chart
    """
    calculator = TrustScoreCalculator(db_path)
    result = calculator.calculate_trust_score(days)

    if not result['data_available']:
        st.warning(f"No execution data available for the last {days} days")
        st.info("Trust score will be calculated once Felix starts executing commands.")
        return

    # Main gauge
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_trust_gauge(
            result['trust_score'],
            result['successful'],
            result['total_executed']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Trust Level")

        # Trust level badge
        trust_score = result['trust_score']
        if trust_score >= 80:
            st.success("High Trust")
            interpretation = "Felix is performing reliably. Safe for autonomous operation."
        elif trust_score >= 60:
            st.warning("Medium Trust")
            interpretation = "Felix is mostly reliable but requires some oversight."
        else:
            st.error("Low Trust")
            interpretation = "Felix requires close supervision. Review executions carefully."

        st.markdown(f"**Score:** {trust_score:.1f}/100")
        st.markdown(f"*{interpretation}*")

    # Metrics grid
    st.markdown("### Execution Statistics")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            "Total Requests",
            result['total_requests'],
            help="Total commands requested by agents"
        )

    with metric_col2:
        st.metric(
            "Successful",
            result['successful'],
            delta=f"{result['success_rate']*100:.1f}%" if result['total_executed'] > 0 else "N/A",
            help="Successfully executed commands"
        )

    with metric_col3:
        st.metric(
            "Failed",
            result['failed'],
            delta=f"-{(result['failed']/result['total_executed']*100):.1f}%" if result['total_executed'] > 0 else "N/A",
            delta_color="inverse",
            help="Commands that executed but failed"
        )

    with metric_col4:
        st.metric(
            "Denied",
            result['denied'],
            delta=f"-{result['denial_rate']*100:.1f}%" if result['total_requests'] > 0 else "N/A",
            delta_color="inverse",
            help="Commands denied before execution"
        )

    # Breakdown text
    st.caption(
        f"{result['successful']}/{result['total_executed']} successful commands "
        f"over {days} days"
    )

    # Optional: Show history chart
    if show_history:
        st.markdown("### Trust Score History")
        history_df = calculator.get_trust_history(days=days)

        if not history_df.empty:
            fig_history = go.Figure()

            fig_history.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['trust_score'],
                mode='lines+markers',
                name='Trust Score',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6)
            ))

            # Add threshold lines
            fig_history.add_hline(y=80, line_dash="dash", line_color="green",
                                 annotation_text="High Trust", annotation_position="right")
            fig_history.add_hline(y=60, line_dash="dash", line_color="orange",
                                 annotation_text="Medium Trust", annotation_position="right")

            fig_history.update_layout(
                xaxis_title="Time",
                yaxis_title="Trust Score",
                yaxis_range=[0, 105],
                height=300,
                hovermode='x unified',
                showlegend=False
            )

            st.plotly_chart(fig_history, use_container_width=True)
        else:
            st.info("Not enough history data to display trend")


# Example usage and demo
if __name__ == "__main__":
    st.set_page_config(page_title="Trust Score Demo", layout="wide")

    st.title("System Autonomy Trust Score")

    st.markdown("""
    This component answers the critical question: **"Should I trust Felix with system commands?"**

    The trust score (0-100) is calculated from execution history using:
    - **70% weight**: Success rate of executed commands
    - **30% weight**: Approval rate (inverse of denial rate)

    **Trust Levels:**
    - **High (80-100)**: Safe for autonomous operation
    - **Medium (60-80)**: Requires some oversight
    - **Low (0-60)**: Close supervision needed
    """)

    st.markdown("---")

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Days to analyze", 1, 30, 7)
    with col2:
        show_history = st.checkbox("Show trust history", value=True)

    st.markdown("---")

    # Render component
    render_trust_score(
        db_path="felix_system_actions.db",
        days=days,
        show_history=show_history
    )
