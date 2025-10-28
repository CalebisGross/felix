"""Command Risk Heatmap Component for Felix Streamlit GUI.

This component visualizes risk distribution of system commands to identify
patterns and inform trust rule updates.
"""

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RiskHeatmapGenerator:
    """Generates risk heatmap for command executions."""

    def __init__(self, db_path: str):
        """
        Initialize the generator.

        Args:
            db_path: Path to felix_system_actions.db
        """
        self.db_path = db_path

    def get_risk_matrix(self) -> pd.DataFrame:
        """
        Query command executions and build risk matrix.

        Returns:
            DataFrame with command_type as rows, trust_level as columns,
            and average risk scores as values
        """
        if not Path(self.db_path).exists():
            logger.warning(f"Database not found: {self.db_path}")
            return self._empty_matrix()

        try:
            conn = sqlite3.connect(self.db_path)

            # Query command executions with calculated risk scores
            query = """
                SELECT
                    command,
                    trust_level,
                    success,
                    exit_code,
                    error_category,
                    executed
                FROM command_executions
                WHERE command IS NOT NULL
                  AND trust_level IS NOT NULL
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                logger.info("No command execution data found")
                return self._empty_matrix()

            # Extract command type (first word of command)
            df['command_type'] = df['command'].apply(self._extract_command_type)

            # Calculate risk score for each command
            df['risk_score'] = df.apply(self._calculate_risk_score, axis=1)

            # Group by command_type and trust_level
            risk_matrix = df.groupby(['command_type', 'trust_level']).agg({
                'risk_score': 'mean',
                'command': 'count'
            }).reset_index()

            risk_matrix.columns = ['command_type', 'trust_level', 'avg_risk', 'count']

            # Pivot to create matrix format
            pivot_df = risk_matrix.pivot(
                index='command_type',
                columns='trust_level',
                values='avg_risk'
            )

            # Fill missing values with 0
            pivot_df = pivot_df.fillna(0)

            # Ensure standard trust levels exist as columns
            for level in ['SAFE', 'REVIEW', 'BLOCKED']:
                if level not in pivot_df.columns:
                    pivot_df[level] = 0

            # Reorder columns
            pivot_df = pivot_df[['SAFE', 'REVIEW', 'BLOCKED']]

            # Store count matrix for reference
            self.count_matrix = risk_matrix.pivot(
                index='command_type',
                columns='trust_level',
                values='count'
            ).fillna(0)

            return pivot_df

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return self._empty_matrix()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self._empty_matrix()

    def _extract_command_type(self, command: str) -> str:
        """
        Extract command type from full command string.

        Args:
            command: Full command string

        Returns:
            First word of command (the command type)
        """
        if not command:
            return 'unknown'

        # Handle common patterns
        command = command.strip()

        # Extract first word (command name)
        parts = command.split()
        if not parts:
            return 'unknown'

        cmd_type = parts[0].lower()

        # Normalize common variations
        if cmd_type.startswith('python'):
            return 'python'
        elif cmd_type.startswith('node'):
            return 'node'
        elif cmd_type.startswith('npm'):
            return 'npm'
        elif cmd_type.startswith('git'):
            return 'git'

        return cmd_type

    def _calculate_risk_score(self, row: pd.Series) -> float:
        """
        Calculate risk score for a command execution.

        Risk scoring:
        - Base risk from trust level (SAFE: 0.1, REVIEW: 0.5, BLOCKED: 0.9)
        - Success factor: +0.0 if success, +0.3 if failed
        - Exit code factor: +0.1 if non-zero exit
        - Error category factor: +0.2 if error present
        - Execution factor: -0.1 if not executed (lower risk)

        Args:
            row: DataFrame row with command execution data

        Returns:
            Risk score between 0.0 and 1.0
        """
        risk = 0.0

        # Base risk from trust level
        trust_level_risk = {
            'SAFE': 0.1,
            'REVIEW': 0.5,
            'BLOCKED': 0.9
        }
        risk += trust_level_risk.get(row['trust_level'], 0.5)

        # Success factor
        if pd.notna(row['success']) and not row['success']:
            risk += 0.3

        # Exit code factor
        if pd.notna(row['exit_code']) and row['exit_code'] != 0:
            risk += 0.1

        # Error category factor
        if pd.notna(row['error_category']) and row['error_category']:
            risk += 0.2

        # Execution factor (not executed = lower actual risk)
        if pd.notna(row['executed']) and not row['executed']:
            risk -= 0.1

        # Normalize to 0-1 range
        return max(0.0, min(1.0, risk))

    def _empty_matrix(self) -> pd.DataFrame:
        """Return empty risk matrix."""
        return pd.DataFrame(
            columns=['SAFE', 'REVIEW', 'BLOCKED']
        )

    def identify_high_risk_patterns(
        self,
        risk_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Identify command patterns with high risk scores.

        Args:
            risk_matrix: Risk matrix DataFrame
            threshold: Risk threshold for flagging (default: 0.7)

        Returns:
            List of high-risk pattern dictionaries
        """
        if risk_matrix.empty:
            return []

        high_risk = []

        for command_type in risk_matrix.index:
            for trust_level in risk_matrix.columns:
                risk_score = risk_matrix.loc[command_type, trust_level]

                if risk_score >= threshold:
                    # Get count if available
                    count = 0
                    if hasattr(self, 'count_matrix') and command_type in self.count_matrix.index:
                        if trust_level in self.count_matrix.columns:
                            count = int(self.count_matrix.loc[command_type, trust_level])

                    high_risk.append({
                        'command_type': command_type,
                        'trust_level': trust_level,
                        'avg_risk': risk_score,
                        'count': count
                    })

        # Sort by risk score descending
        high_risk.sort(key=lambda x: x['avg_risk'], reverse=True)

        return high_risk

    def generate_recommendations(
        self,
        high_risk_patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate actionable recommendations based on high-risk patterns.

        Args:
            high_risk_patterns: List of high-risk pattern dictionaries

        Returns:
            List of recommendation strings
        """
        if not high_risk_patterns:
            return ['No high-risk patterns detected. Trust rules are working effectively.']

        recommendations = []

        for pattern in high_risk_patterns[:5]:  # Top 5 patterns
            cmd = pattern['command_type']
            level = pattern['trust_level']
            risk = pattern['avg_risk']

            if level == 'SAFE' and risk > 0.6:
                recommendations.append(
                    f"⚠️ '{cmd}' is marked SAFE but has high risk ({risk:.2f}). "
                    f"Consider moving to REVIEW level."
                )
            elif level == 'REVIEW' and risk > 0.8:
                recommendations.append(
                    f"⚠️ '{cmd}' at REVIEW level has very high risk ({risk:.2f}). "
                    f"Consider blocking or adding stricter approval rules."
                )
            elif level == 'BLOCKED' and risk < 0.5:
                recommendations.append(
                    f"ℹ️ '{cmd}' is BLOCKED but shows low risk ({risk:.2f}). "
                    f"May be safe to move to REVIEW with monitoring."
                )

        if not recommendations:
            recommendations.append(
                'High-risk patterns detected, but trust levels appear appropriately assigned.'
            )

        return recommendations


def create_risk_heatmap(risk_matrix: pd.DataFrame, count_matrix: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create interactive heatmap visualization of command risks.

    Args:
        risk_matrix: DataFrame with command types as rows, trust levels as columns
        count_matrix: Optional DataFrame with execution counts

    Returns:
        Plotly figure object
    """
    if risk_matrix.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No command execution data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Command Risk Distribution",
            height=400
        )
        return fig

    # Prepare hover text with counts
    hover_text = []
    for i, command_type in enumerate(risk_matrix.index):
        hover_row = []
        for j, trust_level in enumerate(risk_matrix.columns):
            risk_val = risk_matrix.iloc[i, j]
            text = f"Command: {command_type}<br>"
            text += f"Trust Level: {trust_level}<br>"
            text += f"Avg Risk: {risk_val:.3f}<br>"

            if count_matrix is not None and command_type in count_matrix.index:
                if trust_level in count_matrix.columns:
                    count_val = int(count_matrix.loc[command_type, trust_level])
                    text += f"Count: {count_val}"

            hover_row.append(text)
        hover_text.append(hover_row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=risk_matrix.values,
        x=risk_matrix.columns,
        y=risk_matrix.index,
        colorscale=[
            [0.0, '#2ecc71'],    # Green (low risk)
            [0.4, '#f1c40f'],    # Yellow (medium-low risk)
            [0.6, '#e67e22'],    # Orange (medium-high risk)
            [1.0, '#e74c3c']     # Red (high risk)
        ],
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title='Risk Score',
            tickmode='linear',
            tick0=0,
            dtick=0.2
        ),
        zmid=0.5,
        zmin=0.0,
        zmax=1.0
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Command Risk Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title='Trust Level',
        yaxis_title='Command Type',
        height=max(400, len(risk_matrix) * 30),  # Dynamic height based on commands
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif', 'size': 12},
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}  # Top to bottom
    )

    return fig


def render_risk_heatmap(db_path: str):
    """
    Render the complete risk heatmap component in Streamlit.

    Args:
        db_path: Path to felix_system_actions.db
    """
    st.subheader('Command Risk Distribution')
    st.caption(
        'Visualize risk patterns across command types and trust levels '
        'to identify opportunities for trust rule optimization.'
    )

    # Initialize generator
    generator = RiskHeatmapGenerator(db_path)

    # Get risk matrix
    with st.spinner('Analyzing command risk patterns...'):
        risk_matrix = generator.get_risk_matrix()

    if risk_matrix.empty:
        st.info(
            'No command execution data available yet. '
            'System commands will be logged here once workflows are run.'
        )
        return

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_commands = len(risk_matrix)
        st.metric('Command Types', total_commands)

    with col2:
        avg_risk = risk_matrix.values.mean()
        st.metric('Avg Risk Score', f'{avg_risk:.3f}')

    with col3:
        high_risk_count = (risk_matrix.values >= 0.7).sum()
        st.metric('High Risk Cells', high_risk_count)

    with col4:
        if hasattr(generator, 'count_matrix'):
            total_executions = int(generator.count_matrix.values.sum())
            st.metric('Total Executions', f'{total_executions:,}')
        else:
            st.metric('Total Executions', 'N/A')

    # Display heatmap
    st.markdown('---')
    count_matrix = getattr(generator, 'count_matrix', None)
    fig = create_risk_heatmap(risk_matrix, count_matrix)
    st.plotly_chart(fig, use_container_width=True)

    # Identify high-risk patterns
    st.markdown('---')
    st.markdown('#### High-Risk Pattern Analysis')

    high_risk_patterns = generator.identify_high_risk_patterns(risk_matrix, threshold=0.7)

    if high_risk_patterns:
        st.warning(f"⚠️ **{len(high_risk_patterns)} high-risk pattern(s) detected**")

        # Display high-risk patterns table
        risk_df = pd.DataFrame(high_risk_patterns)
        risk_df['avg_risk'] = risk_df['avg_risk'].apply(lambda x: f'{x:.3f}')

        st.dataframe(
            risk_df,
            column_config={
                'command_type': 'Command',
                'trust_level': 'Trust Level',
                'avg_risk': 'Avg Risk',
                'count': 'Executions'
            },
            hide_index=True,
            use_container_width=True
        )

        # Generate and display recommendations
        st.markdown('#### Recommendations')
        recommendations = generator.generate_recommendations(high_risk_patterns)

        for rec in recommendations:
            if rec.startswith('⚠️'):
                st.warning(rec)
            elif rec.startswith('ℹ️'):
                st.info(rec)
            else:
                st.write(rec)

    else:
        st.success('✅ No high-risk patterns detected. Trust rules are effectively managing command risk.')

    # Display trust level distribution
    st.markdown('---')
    st.markdown('#### Trust Level Distribution')

    col1, col2 = st.columns(2)

    with col1:
        # Commands by trust level
        trust_counts = {}
        for trust_level in risk_matrix.columns:
            # Count non-zero risk scores (indicates command exists at this level)
            trust_counts[trust_level] = (risk_matrix[trust_level] > 0).sum()

        trust_df = pd.DataFrame([
            {'Trust Level': level, 'Command Types': count}
            for level, count in trust_counts.items()
        ])

        fig_trust = px.bar(
            trust_df,
            x='Trust Level',
            y='Command Types',
            color='Trust Level',
            color_discrete_map={
                'SAFE': '#2ecc71',
                'REVIEW': '#f1c40f',
                'BLOCKED': '#e74c3c'
            },
            title='Command Types by Trust Level'
        )
        fig_trust.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_trust, use_container_width=True)

    with col2:
        # Average risk by trust level
        avg_risks = {
            level: risk_matrix[level][risk_matrix[level] > 0].mean()
            for level in risk_matrix.columns
        }

        avg_risk_df = pd.DataFrame([
            {'Trust Level': level, 'Avg Risk': risk}
            for level, risk in avg_risks.items()
            if not pd.isna(risk)
        ])

        if not avg_risk_df.empty:
            fig_avg = px.bar(
                avg_risk_df,
                x='Trust Level',
                y='Avg Risk',
                color='Trust Level',
                color_discrete_map={
                    'SAFE': '#2ecc71',
                    'REVIEW': '#f1c40f',
                    'BLOCKED': '#e74c3c'
                },
                title='Average Risk Score by Trust Level'
            )
            fig_avg.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_avg, use_container_width=True)

    # Additional guidance
    with st.expander('Understanding Risk Scores'):
        st.markdown("""
        **Risk Score Calculation:**

        Risk scores are calculated based on multiple factors:
        - **Trust Level Base Risk:**
          - SAFE: 0.1 base risk
          - REVIEW: 0.5 base risk
          - BLOCKED: 0.9 base risk

        - **Execution Outcome Modifiers:**
          - Failed execution: +0.3 risk
          - Non-zero exit code: +0.1 risk
          - Error category present: +0.2 risk
          - Not executed: -0.1 risk (blocked before execution)

        **Color Scale:**
        - 🟢 Green (0.0-0.4): Low risk - commands are generally safe
        - 🟡 Yellow (0.4-0.6): Medium risk - review recommended
        - 🟠 Orange (0.6-0.8): High risk - requires attention
        - 🔴 Red (0.8-1.0): Very high risk - immediate action needed

        **Action Guidelines:**
        - SAFE commands with high risk should be moved to REVIEW
        - REVIEW commands with very high risk may need BLOCKED status
        - BLOCKED commands with low risk might be over-restricted
        - Monitor patterns over time to refine trust rules
        """)


if __name__ == '__main__':
    """Example usage and testing."""
    import sys
    import os
    import time
    import random

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Test with actual database
    db_path = project_root / 'felix_system_actions.db'

    print("Risk Heatmap Generator - Standalone Test")
    print("=" * 60)

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Creating sample data for demonstration...")

        # Create sample database for testing
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS command_executions (
                execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id INTEGER,
                agent_id TEXT NOT NULL,
                command TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                executed BOOLEAN NOT NULL DEFAULT 0,
                exit_code INTEGER,
                success BOOLEAN,
                error_category TEXT,
                timestamp REAL NOT NULL
            )
        """)

        # Insert sample data
        commands = [
            ('ls -la', 'SAFE'),
            ('cat file.txt', 'SAFE'),
            ('rm -rf temp/', 'REVIEW'),
            ('git push origin main', 'REVIEW'),
            ('python script.py', 'SAFE'),
            ('npm install', 'REVIEW'),
            ('sudo systemctl restart', 'BLOCKED'),
            ('chmod 777', 'BLOCKED'),
        ]

        for i in range(50):
            cmd, default_level = random.choice(commands)
            # Occasionally change trust level
            trust_level = default_level if random.random() > 0.2 else random.choice(['SAFE', 'REVIEW', 'BLOCKED'])
            executed = trust_level != 'BLOCKED' or random.random() > 0.8
            success = executed and random.random() > 0.2
            exit_code = 0 if success else random.choice([1, 2, 127])
            error_category = None if success else random.choice(['permission', 'not_found', 'syntax'])

            cursor.execute("""
                INSERT INTO command_executions
                (workflow_id, agent_id, command, trust_level, executed,
                 exit_code, success, error_category, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                random.randint(1, 5),
                f'agent_{random.randint(1, 10)}',
                cmd,
                trust_level,
                executed,
                exit_code if executed else None,
                success if executed else None,
                error_category,
                time.time() - random.randint(0, 86400)
            ))

        conn.commit()
        conn.close()
        print("Sample database created.")

    # Generate risk matrix
    generator = RiskHeatmapGenerator(str(db_path))
    risk_matrix = generator.get_risk_matrix()

    print("\nRisk Matrix:")
    print("-" * 60)
    print(risk_matrix)

    # Identify high-risk patterns
    print("\nHigh-Risk Patterns:")
    print("-" * 60)
    high_risk = generator.identify_high_risk_patterns(risk_matrix, threshold=0.7)
    for pattern in high_risk:
        print(f"  {pattern['command_type']} [{pattern['trust_level']}]: "
              f"Risk={pattern['avg_risk']:.3f}, Count={pattern['count']}")

    # Generate recommendations
    print("\nRecommendations:")
    print("-" * 60)
    recommendations = generator.generate_recommendations(high_risk)
    for rec in recommendations:
        print(f"  {rec}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
