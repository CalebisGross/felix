"""
Testing page for Felix Framework.

Provides workflow result analysis, test execution monitoring,
and performance evaluation tools.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sys
import os
from typing import List, Dict, Any

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit_gui.backend.system_monitor import SystemMonitor
from streamlit_gui.backend.db_reader import DatabaseReader

st.set_page_config(
    page_title="Felix Testing",
    page_icon="🧪",
    layout="wide"
)

@st.cache_resource
def get_monitor():
    """Get cached SystemMonitor instance."""
    return SystemMonitor()

@st.cache_resource
def get_db_reader():
    """Get cached DatabaseReader instance."""
    return DatabaseReader()


def analyze_workflow_results(workflows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze workflow execution results.

    Args:
        workflows: List of workflow results

    Returns:
        Analysis summary
    """
    if not workflows:
        return {
            'total_workflows': 0,
            'success_rate': 0,
            'avg_agents': 0,
            'avg_duration': 0,
            'common_patterns': []
        }

    total = len(workflows)
    successful = sum(1 for w in workflows if w.get('success', False))
    avg_agents = sum(w.get('agent_count', 0) for w in workflows) / total if total > 0 else 0

    # Calculate average duration if timestamps available
    durations = []
    for w in workflows:
        if 'start_time' in w and 'end_time' in w:
            try:
                duration = w['end_time'] - w['start_time']
                durations.append(duration)
            except:
                pass

    avg_duration = sum(durations) / len(durations) if durations else 0

    # Extract common patterns
    tasks = [w.get('task', '') for w in workflows if w.get('task')]
    common_patterns = []
    if tasks:
        # Simple pattern extraction - count common words
        from collections import Counter
        words = []
        for task in tasks:
            words.extend(task.lower().split())

        word_counts = Counter(words)
        common_patterns = [word for word, count in word_counts.most_common(5) if count > 1]

    return {
        'total_workflows': total,
        'success_rate': (successful / total * 100) if total > 0 else 0,
        'avg_agents': avg_agents,
        'avg_duration': avg_duration,
        'common_patterns': common_patterns
    }


def create_confidence_distribution_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create confidence distribution chart.

    Args:
        data: DataFrame with confidence values

    Returns:
        Plotly figure
    """
    if data.empty or 'confidence' not in data.columns:
        return go.Figure().add_annotation(
            text="No confidence data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    fig = go.Figure()

    # Create histogram
    fig.add_trace(go.Histogram(
        x=data['confidence'],
        nbinsx=20,
        name='Confidence Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))

    # Add mean line
    mean_conf = data['confidence'].mean()
    fig.add_vline(
        x=mean_conf,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_conf:.3f}"
    )

    # Add median line
    median_conf = data['confidence'].median()
    fig.add_vline(
        x=median_conf,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_conf:.3f}"
    )

    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence",
        yaxis_title="Frequency",
        bargap=0.1,
        height=400
    )

    return fig


def create_workflow_timeline(workflows: List[Dict[str, Any]]) -> go.Figure:
    """
    Create timeline visualization of workflow executions.

    Args:
        workflows: List of workflow data

    Returns:
        Plotly timeline figure
    """
    if not workflows:
        return go.Figure().add_annotation(
            text="No workflow data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Prepare data for timeline
    timeline_data = []
    for idx, workflow in enumerate(workflows):
        try:
            # Convert timestamp to datetime
            timestamp = pd.to_datetime(workflow.get('timestamp', ''), unit='s')

            timeline_data.append({
                'Task': workflow.get('task', 'Unknown')[:30] + '...',
                'Start': timestamp,
                'End': timestamp + timedelta(minutes=5),  # Assume 5 min duration
                'Status': 'Success' if workflow.get('success', True) else 'Failed',
                'Agents': workflow.get('agent_count', 0)
            })
        except:
            continue

    if not timeline_data:
        return go.Figure().add_annotation(
            text="Could not parse workflow timestamps",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    df = pd.DataFrame(timeline_data)

    # Create Gantt chart
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Task",
        color="Status",
        hover_data=["Agents"],
        title="Workflow Execution Timeline",
        color_discrete_map={"Success": "green", "Failed": "red"}
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(400, len(timeline_data) * 40))

    return fig


def main():
    st.title("🧪 Testing & Analysis")
    st.markdown("Analyze workflow results and test execution performance")

    # Real data indicator
    st.success("✅ **Real Data**: This page displays actual workflow execution data from Felix databases.")

    monitor = get_monitor()
    db_reader = get_db_reader()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Workflow Results",
        "📈 Performance Analysis",
        "🔍 Test Details",
        "📝 Reports"
    ])

    with tab1:
        st.subheader("Workflow Execution Results")

        st.info("""
        **What this shows**: Historical results from workflows executed through the tkinter GUI or command line.
        Each workflow represents a complete task processed by Felix's multi-agent system.
        """)

        # Get workflow data
        workflows = monitor.get_workflow_results(limit=50)

        if workflows:
            # Analysis summary
            analysis = analyze_workflow_results(workflows)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Workflows", analysis['total_workflows'], help="Number of complete workflow executions recorded in database")
            with col2:
                st.metric("Success Rate", f"{analysis['success_rate']:.1f}%", help="Percentage of workflows that completed successfully without errors")
            with col3:
                st.metric("Avg Agents/Workflow", f"{analysis['avg_agents']:.1f}", help="Average number of agents spawned per workflow (Research, Analysis, Synthesis, Critic)")
            with col4:
                duration_str = f"{analysis['avg_duration']:.1f}s" if analysis['avg_duration'] > 0 else "N/A"
                st.metric("Avg Duration", duration_str, help="Average time from workflow start to completion (if timestamp data available)")

            # Common patterns
            if analysis['common_patterns']:
                st.info(f"Common patterns: {', '.join(analysis['common_patterns'])}")

            st.divider()

            # Workflow timeline
            st.subheader("Execution Timeline")
            timeline_fig = create_workflow_timeline(workflows)
            st.plotly_chart(timeline_fig, width='stretch')

            # Workflow details table
            st.subheader("Recent Workflows")

            # Prepare DataFrame
            df_workflows = pd.DataFrame(workflows)

            if not df_workflows.empty:
                # Format timestamps
                if 'timestamp' in df_workflows.columns:
                    df_workflows['timestamp'] = pd.to_datetime(df_workflows['timestamp'], unit='s', errors='coerce')
                    df_workflows['timestamp'] = df_workflows['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                # Select columns to display
                display_cols = ['task', 'timestamp', 'success', 'agent_count']
                available_cols = [col for col in display_cols if col in df_workflows.columns]

                if available_cols:
                    display_df = df_workflows[available_cols].copy()

                    # Format success column
                    if 'success' in display_df.columns:
                        display_df['success'] = display_df['success'].apply(
                            lambda x: '✅' if x else '❌'
                        )

                    # Truncate task names
                    if 'task' in display_df.columns:
                        display_df['task'] = display_df['task'].apply(
                            lambda x: x[:50] + '...' if len(str(x)) > 50 else x
                        )

                    st.dataframe(display_df, width='stretch', hide_index=True)

        else:
            st.info("No workflow results available. Run workflows from the tkinter GUI to see results here.")

    with tab2:
        st.subheader("Performance Analysis")

        # Get performance data
        knowledge_df = db_reader.get_knowledge_entries(limit=500)

        if not knowledge_df.empty:
            # Confidence distribution
            conf_fig = create_confidence_distribution_chart(knowledge_df)
            st.plotly_chart(conf_fig, width='stretch')

            # Performance over time
            st.subheader("Performance Trends")

            # Time series analysis
            ts_df = db_reader.get_time_series_metrics(hours=48)

            if not ts_df.empty:
                # Create dual-axis chart
                fig = go.Figure()

                # Add entries count
                fig.add_trace(go.Bar(
                    x=pd.to_datetime(ts_df['time']),
                    y=ts_df['entries'],
                    name='Entry Count',
                    yaxis='y',
                    marker_color='lightblue',
                    opacity=0.7
                ))

                # Add confidence line
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(ts_df['time']),
                    y=ts_df['avg_confidence'],
                    name='Avg Confidence',
                    yaxis='y2',
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))

                fig.update_layout(
                    title='Performance Metrics Over Time',
                    xaxis_title='Time',
                    yaxis=dict(
                        title='Entry Count',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Average Confidence',
                        overlaying='y',
                        side='right',
                        range=[0, 1]
                    ),
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig, width='stretch')

            # Agent performance comparison
            st.subheader("Agent Performance Comparison")

            agent_metrics = db_reader.get_agent_metrics()
            if not agent_metrics.empty:
                # Create bubble chart
                fig = px.scatter(
                    agent_metrics,
                    x='output_count',
                    y='avg_confidence',
                    size='output_count',
                    color='avg_confidence',
                    hover_data=['agent_id', 'min_confidence', 'max_confidence'],
                    title='Agent Performance Matrix',
                    labels={
                        'output_count': 'Number of Outputs',
                        'avg_confidence': 'Average Confidence'
                    },
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 1]
                )

                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

                # Performance table
                st.dataframe(
                    agent_metrics[['agent_id', 'output_count', 'avg_confidence']],
                    width='stretch',
                    hide_index=True
                )

        else:
            st.info("No performance data available yet.")

    with tab3:
        st.subheader("Test Execution Details")

        # Test case selector
        workflow_tasks = [w['task'] for w in workflows] if workflows else []

        if workflow_tasks:
            selected_task = st.selectbox(
                "Select workflow to analyze",
                options=workflow_tasks[:20],  # Limit to recent 20
                format_func=lambda x: x[:60] + '...' if len(x) > 60 else x
            )

            # Find selected workflow
            selected_workflow = next(
                (w for w in workflows if w['task'] == selected_task),
                None
            )

            if selected_workflow:
                st.markdown("### Workflow Details")

                # Display workflow information
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Task:** " + selected_workflow.get('task', 'N/A'))
                    st.markdown("**Status:** " + ('✅ Success' if selected_workflow.get('success') else '❌ Failed'))
                    st.markdown("**Agent Count:** " + str(selected_workflow.get('agent_count', 0)))

                with col2:
                    timestamp = selected_workflow.get('timestamp', '')
                    if timestamp:
                        try:
                            ts = pd.to_datetime(timestamp, unit='s')
                            st.markdown("**Timestamp:** " + ts.strftime('%Y-%m-%d %H:%M:%S'))
                        except:
                            st.markdown("**Timestamp:** " + str(timestamp))

                # Display synthesis if available
                if selected_workflow.get('final_synthesis'):
                    st.markdown("### Final Synthesis")
                    st.text_area(
                        "Synthesis Output",
                        selected_workflow['final_synthesis'],
                        height=200,
                        disabled=True
                    )

                # Pattern analysis
                if selected_workflow.get('pattern'):
                    st.markdown("### Pattern Analysis")
                    st.json(selected_workflow.get('pattern'))

                # Related knowledge entries
                st.markdown("### Related Knowledge Entries")

                # Query for related entries (simplified - in production would filter by workflow ID)
                recent_knowledge = db_reader.get_knowledge_entries(limit=10)
                if not recent_knowledge.empty:
                    st.dataframe(
                        recent_knowledge[['agent_id', 'domain', 'confidence', 'content_preview']],
                        width='stretch',
                        hide_index=True
                    )

        else:
            st.info("No workflows available for detailed analysis.")

    with tab4:
        st.subheader("Test Reports")

        st.info("""
        **Report Types Explained**:
        - **Summary**: High-level overview with key metrics and trends
        - **Detailed**: Complete breakdown of all workflow executions with timestamps
        - **Performance**: Focus on execution times, resource usage, and bottlenecks
        - **Confidence**: Agent confidence scores and progression analysis
        """)

        # Report generation options
        col1, col2 = st.columns(2)

        with col1:
            report_type = st.selectbox(
                "Report Type",
                options=["Summary", "Detailed", "Performance", "Confidence"],
                help="Choose the level of detail and focus area for the report"
            )

            time_range = st.selectbox(
                "Time Range",
                options=["Last Hour", "Last 24 Hours", "Last Week", "All Time"],
                help="Filter workflows by execution time"
            )

        with col2:
            include_charts = st.checkbox("Include Charts", value=True, help="Add visualizations to the report")
            include_raw_data = st.checkbox("Include Raw Data", value=False, help="Include raw database entries (increases report size)")

        if st.button("Generate Report"):
            st.markdown("### Test Execution Report")

            # Report metadata
            st.markdown(f"""
            **Report Type:** {report_type}
            **Time Range:** {time_range}
            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)

            st.divider()

            # Generate report content based on type
            if report_type == "Summary":
                # Summary report
                st.markdown("#### Executive Summary")

                if workflows:
                    analysis = analyze_workflow_results(workflows)

                    st.markdown(f"""
                    - **Total Workflows Executed:** {analysis['total_workflows']}
                    - **Overall Success Rate:** {analysis['success_rate']:.1f}%
                    - **Average Agents per Workflow:** {analysis['avg_agents']:.1f}
                    - **Common Task Patterns:** {', '.join(analysis['common_patterns']) if analysis['common_patterns'] else 'None identified'}
                    """)

                    if include_charts:
                        # Include timeline chart
                        timeline_fig = create_workflow_timeline(workflows[:10])
                        st.plotly_chart(timeline_fig, width='stretch')

            elif report_type == "Performance":
                # Performance report
                st.markdown("#### Performance Analysis Report")

                # Get metrics
                agent_metrics = db_reader.get_agent_metrics()
                if not agent_metrics.empty:
                    st.markdown("##### Agent Performance Summary")

                    avg_conf = agent_metrics['avg_confidence'].mean()
                    total_outputs = agent_metrics['output_count'].sum()
                    active_agents = len(agent_metrics)

                    st.markdown(f"""
                    - **Active Agents:** {active_agents}
                    - **Total Outputs:** {total_outputs}
                    - **Overall Average Confidence:** {avg_conf:.3f}
                    - **Best Performing Agent:** {agent_metrics.iloc[0]['agent_id'] if not agent_metrics.empty else 'N/A'}
                    """)

                    if include_charts:
                        # Performance scatter plot
                        fig = px.scatter(
                            agent_metrics,
                            x='output_count',
                            y='avg_confidence',
                            title='Agent Performance Distribution'
                        )
                        st.plotly_chart(fig, width='stretch')

            elif report_type == "Confidence":
                # Confidence analysis report
                st.markdown("#### Confidence Analysis Report")

                knowledge_df = db_reader.get_knowledge_entries(limit=500)
                if not knowledge_df.empty and 'confidence' in knowledge_df.columns:
                    confidence_stats = knowledge_df['confidence'].describe()

                    st.markdown("##### Confidence Statistics")
                    st.dataframe(confidence_stats, width='content')

                    if include_charts:
                        conf_fig = create_confidence_distribution_chart(knowledge_df)
                        st.plotly_chart(conf_fig, width='stretch')

            else:
                # Detailed report
                st.markdown("#### Detailed Test Report")

                if workflows:
                    for idx, workflow in enumerate(workflows[:5]):  # Limit to 5 for brevity
                        with st.expander(f"Workflow {idx+1}: {workflow['task'][:50]}..."):
                            st.json(workflow)

            # Export options
            st.divider()
            st.markdown("### Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📥 Export as JSON"):
                    report_data = {
                        'type': report_type,
                        'time_range': time_range,
                        'generated': datetime.now().isoformat(),
                        'workflows': workflows[:20] if workflows else [],
                        'analysis': analyze_workflow_results(workflows) if workflows else {}
                    }

                    st.download_button(
                        "Download JSON Report",
                        data=json.dumps(report_data, indent=2, default=str),
                        file_name=f"felix_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

            with col2:
                if st.button("📥 Export as CSV"):
                    if workflows:
                        df = pd.DataFrame(workflows)
                        csv = df.to_csv(index=False)

                        st.download_button(
                            "Download CSV Report",
                            data=csv,
                            file_name=f"felix_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

            with col3:
                if st.button("📥 Export as HTML"):
                    st.info("HTML export coming soon")

    # Status indicator
    st.divider()
    felix_status = "🟢 Felix Running" if monitor.check_felix_running() else "🔴 Felix Stopped"
    st.caption(f"System Status: {felix_status}")


if __name__ == "__main__":
    main()