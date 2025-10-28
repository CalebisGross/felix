"""Testing page for Felix Framework."""
# -*- coding: utf-8 -*-

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
    # Real data indicator badge - MUST be at the very top before anything else
    st.success("✅ **Real Data**: This page displays actual workflow execution data from Felix databases.")

    st.title("🧪 Testing & Analysis")
    st.markdown("Analyze workflow results and test execution performance")

    monitor = get_monitor()
    db_reader = get_db_reader()

    # Create simplified 2-tab layout
    tab1, tab2 = st.tabs([
        "📊 Testing",
        "📈 Benchmarking & Reports"
    ])

    with tab1:
        st.subheader("Workflow Execution Results")

        # Get workflow data
        workflows = monitor.get_workflow_results(limit=50)

        if workflows:
            # Analysis summary
            analysis = analyze_workflow_results(workflows)

            # Display metrics - simplified, no documentation expanders
            st.markdown("### Key Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Workflows",
                    analysis['total_workflows']
                )

            with col2:
                st.metric(
                    "Success Rate",
                    f"{analysis['success_rate']:.1f}%"
                )

            with col3:
                st.metric(
                    "Avg Agents",
                    f"{analysis['avg_agents']:.1f}"
                )

            with col4:
                duration_str = f"{analysis['avg_duration']:.1f}s" if analysis['avg_duration'] > 0 else "N/A"
                st.metric(
                    "Avg Duration",
                    duration_str
                )

            # Common patterns - minimal display
            if analysis['common_patterns']:
                st.caption(f"Common patterns: {', '.join(analysis['common_patterns'])}")

            st.divider()

            # Workflow timeline
            st.subheader("Execution Timeline")
            timeline_fig = create_workflow_timeline(workflows)
            st.plotly_chart(timeline_fig, use_container_width=True)

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

                    # Truncate task names and ensure UTF-8 safe display
                    if 'task' in display_df.columns:
                        def format_task(x):
                            # Handle bytes objects
                            if isinstance(x, bytes):
                                x = x.decode('utf-8', errors='replace')
                            # Convert to string and truncate
                            x_str = str(x)
                            return x_str[:50] + '...' if len(x_str) > 50 else x_str

                        display_df['task'] = display_df['task'].apply(format_task)

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Test Details Section
            st.divider()
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
                        # Ensure UTF-8 safe display of task text
                        task_text = selected_workflow.get('task', 'N/A')
                        if isinstance(task_text, bytes):
                            task_text = task_text.decode('utf-8', errors='replace')
                        st.markdown(f"**Task:** {task_text}")
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
                        st.subheader("Final Synthesis")
                        synthesis_text = selected_workflow['final_synthesis']
                        # Ensure UTF-8 safe display
                        if isinstance(synthesis_text, bytes):
                            synthesis_text = synthesis_text.decode('utf-8', errors='replace')
                        st.text_area(
                            "Synthesis Output",
                            synthesis_text,
                            height=200,
                            disabled=True
                        )

                    # Pattern analysis
                    if selected_workflow.get('pattern'):
                        st.subheader("Pattern Analysis")
                        st.json(selected_workflow.get('pattern'))

                    st.divider()
                    st.subheader("Related Knowledge Entries")

                    # Query for related entries (simplified - in production would filter by workflow ID)
                    recent_knowledge = db_reader.get_knowledge_entries(limit=10)
                    if not recent_knowledge.empty:
                        st.dataframe(
                            recent_knowledge[['agent_id', 'domain', 'confidence', 'content_preview']],
                            use_container_width=True,
                            hide_index=True
                        )

        else:
            st.info("No workflow results available. Run workflows from the tkinter GUI to see results here.")

    with tab2:
        st.subheader("Performance Analysis")

        # Get performance data
        knowledge_df = db_reader.get_knowledge_entries(limit=500)

        if not knowledge_df.empty:
            # Confidence distribution
            conf_fig = create_confidence_distribution_chart(knowledge_df)
            st.plotly_chart(conf_fig, use_container_width=True)

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

                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

                # Performance table
                st.dataframe(
                    agent_metrics[['agent_id', 'output_count', 'avg_confidence']],
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.info("No performance data available yet.")

        # Benchmarking Section
        st.divider()
        st.subheader("Benchmarking & Reports")

        # Simplified report generation - single button with format dropdown
        col1, col2 = st.columns([2, 1])

        with col1:
            report_type = st.selectbox(
                "Report Type",
                options=["Summary", "Detailed", "Performance", "Confidence"]
            )

        with col2:
            generate_report = st.button("Generate Report", use_container_width=True)

        if generate_report:
            st.markdown("### Test Execution Report")

            # Report metadata
            st.markdown(f"""
            **Report Type:** {report_type}
            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)

            st.divider()

            # Generate report content based on type
            workflows = monitor.get_workflow_results(limit=50)

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

                    # Include timeline chart
                    timeline_fig = create_workflow_timeline(workflows[:10])
                    st.plotly_chart(timeline_fig, use_container_width=True)

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

                    # Performance scatter plot
                    fig = px.scatter(
                        agent_metrics,
                        x='output_count',
                        y='avg_confidence',
                        title='Agent Performance Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif report_type == "Confidence":
                # Confidence analysis report
                st.markdown("#### Confidence Analysis Report")

                knowledge_df = db_reader.get_knowledge_entries(limit=500)
                if not knowledge_df.empty and 'confidence' in knowledge_df.columns:
                    confidence_stats = knowledge_df['confidence'].describe()

                    st.markdown("##### Confidence Statistics")
                    st.dataframe(confidence_stats, width='content')

                    conf_fig = create_confidence_distribution_chart(knowledge_df)
                    st.plotly_chart(conf_fig, use_container_width=True)

            else:
                # Detailed report
                st.markdown("#### Detailed Test Report")

                if workflows:
                    for idx, workflow in enumerate(workflows[:5]):  # Limit to 5 for brevity
                        with st.expander(f"Workflow {idx+1}: {workflow['task'][:50]}..."):
                            st.json(workflow)

            # Export options
            st.divider()
            st.markdown("### Export Report")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Export as JSON", key="export_json"):
                    if workflows:
                        report_data = {
                            'type': report_type,
                            'generated': datetime.now().isoformat(),
                            'workflows': workflows[:20],
                            'analysis': analyze_workflow_results(workflows)
                        }

                        # Ensure UTF-8 encoding for JSON export
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(report_data, indent=2, default=str, ensure_ascii=False),
                            file_name=f"felix_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json;charset=utf-8",
                            key="download_json"
                        )

            with col2:
                if st.button("Export as CSV", key="export_csv"):
                    if workflows:
                        df = pd.DataFrame(workflows)
                        # Ensure UTF-8 encoding for CSV export
                        csv = df.to_csv(index=False, encoding='utf-8-sig')

                        st.download_button(
                            "Download CSV",
                            data=csv,
                            file_name=f"felix_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv;charset=utf-8",
                            key="download_csv"
                        )

            with col3:
                st.button("Export as HTML (coming soon)", disabled=True)

    # Status indicator
    st.divider()
    felix_status = "🟢 Felix Running" if monitor.check_felix_running() else "🔴 Felix Stopped"
    st.caption(f"System Status: {felix_status}")


if __name__ == "__main__":
    main()
