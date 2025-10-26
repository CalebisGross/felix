"""Workflow history viewer component for Streamlit GUI."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json


class WorkflowHistoryViewer:
    """Workflow history visualization component with search, filter, and detailed views."""

    def __init__(self, db_reader):
        """Initialize viewer with database reader."""
        self.db_reader = db_reader

    def render(self):
        """Render the complete workflow history viewer."""
        st.header("📜 Workflow History")
        st.markdown("Browse and analyze past workflow executions")

        # Summary Metrics
        self._render_summary_metrics()

        st.divider()

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.selectbox(
                "Status",
                ["all", "completed", "failed"],
                help="Filter by workflow status"
            )

        with col2:
            days_back = st.number_input(
                "Days Back",
                min_value=1,
                max_value=90,
                value=7,
                help="Show workflows from last N days"
            )

        with col3:
            limit = st.number_input(
                "Max Results",
                min_value=10,
                max_value=500,
                value=100,
                help="Maximum workflows to display"
            )

        # Search
        search_query = st.text_input(
            "Search Task Description",
            placeholder="Enter keywords to search...",
            help="Search in task input text"
        )

        # Load workflows
        workflows_df = self.db_reader.get_workflow_history(
            limit=limit,
            status_filter=None if status_filter == "all" else status_filter,
            days_back=days_back,
            search_query=search_query if search_query else None
        )

        if workflows_df.empty:
            st.info("No workflows found matching criteria")
            return

        st.success(f"Found {len(workflows_df)} workflow(s)")

        # Display workflow list
        self._render_workflow_list(workflows_df)

        st.divider()

        # Charts
        self._render_workflow_charts(workflows_df)

    def _render_summary_metrics(self):
        """Render summary metrics from workflow history."""
        stats = self.db_reader.get_workflow_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Workflows", stats['total_count'])

        with col2:
            success_rate = (stats['completed_count'] / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        with col3:
            st.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")

        with col4:
            st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.1f}s")

    def _render_workflow_list(self, df: pd.DataFrame):
        """Render workflow list with selection."""
        st.subheader("Workflow List")

        # Format display dataframe
        display_df = df.copy()
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
        display_df['processing_time'] = display_df['processing_time'].round(2).astype(str) + 's'

        # Truncate task and output for table view
        display_df['task_preview'] = display_df['task_input'].str[:50] + '...'
        display_df['output_preview'] = display_df['final_synthesis'].str[:80] + '...'

        # Select columns for display
        table_df = display_df[[
            'workflow_id', 'task_preview', 'output_preview', 'status',
            'confidence', 'agents_count', 'tokens_used',
            'processing_time', 'created_at'
        ]]

        # Display with clickable rows
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "workflow_id": st.column_config.NumberColumn("ID", width="small"),
                "task_preview": st.column_config.TextColumn("Task", width="medium"),
                "output_preview": st.column_config.TextColumn("Output Preview", width="large"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "agents_count": st.column_config.NumberColumn("Agents", width="small"),
                "tokens_used": st.column_config.NumberColumn("Tokens", width="small"),
                "processing_time": st.column_config.TextColumn("Time", width="small"),
                "created_at": st.column_config.TextColumn("Date", width="medium"),
            }
        )

        # Detailed view selector
        selected_id = st.number_input(
            "View Workflow Details (enter ID)",
            min_value=1,
            max_value=int(df['workflow_id'].max()) if not df.empty else 1,
            value=None,
            placeholder="Enter workflow ID...",
            help="Enter a workflow ID to view full details"
        )

        if selected_id:
            workflow = self.db_reader.get_workflow_by_id(selected_id)
            if workflow:
                self._render_workflow_detail(workflow)
            else:
                st.error(f"Workflow {selected_id} not found")

    def _render_workflow_detail(self, workflow: Dict[str, Any]):
        """Render detailed view of a single workflow."""
        st.subheader(f"Workflow #{workflow['workflow_id']} Details")

        # Status header
        status_icon = "✅" if workflow['status'] == 'completed' else "❌"
        st.markdown(f"### {status_icon} Status: {workflow['status']}")

        # PROMINENT FINAL OUTPUT DISPLAY
        st.divider()
        st.markdown("### 📄 Final Synthesis Output")
        st.markdown("This is the complete output from the workflow:")

        # Display full synthesis in a large, readable text area
        st.text_area(
            label="Final Output",
            value=workflow['final_synthesis'],
            height=300,  # Large height for readability
            disabled=True,
            label_visibility="collapsed"
        )

        # Add copy button
        if st.button("📋 Copy Output to Clipboard"):
            st.code(workflow['final_synthesis'], language=None)
            st.success("Output displayed above - use browser copy function")

        st.divider()

        # Metrics (move below the output)
        st.markdown("### 📊 Workflow Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Confidence", f"{workflow['confidence']*100:.1f}%")
        with col2:
            st.metric("Agents", workflow['agents_count'])
        with col3:
            st.metric("Tokens", f"{workflow['tokens_used']:,}")
        with col4:
            st.metric("Temperature", f"{workflow['temperature']:.2f}")
        with col5:
            st.metric("Time", f"{workflow['processing_time']:.2f}s")

        # Task Input (move below metrics)
        st.divider()
        st.markdown("### 📝 Task Input")
        st.text_area(
            "Task",
            value=workflow['task_input'],
            height=100,
            disabled=True,
            label_visibility="collapsed"
        )

        # Metadata last
        if workflow.get('metadata'):
            with st.expander("View Technical Metadata"):
                try:
                    metadata = json.loads(workflow['metadata']) if isinstance(workflow['metadata'], str) else workflow['metadata']
                    st.json(metadata)
                except:
                    st.text(workflow['metadata'])

        # Export (include full output)
        st.divider()
        if st.button("💾 Export Complete Workflow"):
            export_data = {
                'workflow_id': workflow['workflow_id'],
                'task_input': workflow['task_input'],
                'final_output': workflow['final_synthesis'],  # Include full output
                'status': workflow['status'],
                'confidence': workflow['confidence'],
                'metrics': {
                    'agents_count': workflow['agents_count'],
                    'tokens_used': workflow['tokens_used'],
                    'processing_time': workflow['processing_time'],
                    'temperature': workflow['temperature']
                },
                'timestamps': {
                    'created_at': workflow.get('created_at'),
                    'completed_at': workflow.get('completed_at')
                }
            }

            export_json = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "Download JSON",
                data=export_json,
                file_name=f"workflow_{workflow['workflow_id']}_output.json",
                mime="application/json"
            )

    def _render_workflow_charts(self, df: pd.DataFrame):
        """Render workflow analytics charts."""
        st.subheader("Workflow Analytics")

        tab1, tab2, tab3 = st.tabs([
            "Confidence Trend",
            "Token Usage",
            "Processing Time"
        ])

        with tab1:
            # Confidence over time
            df_sorted = df.sort_values('created_at')
            fig = px.line(
                df_sorted,
                x='created_at',
                y='confidence',
                title="Confidence Score Trend",
                markers=True
            )
            fig.update_yaxes(title="Confidence", tickformat=".0%")
            fig.update_xaxes(title="Date")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Token usage distribution
            fig = px.histogram(
                df,
                x='tokens_used',
                title="Token Usage Distribution",
                nbins=30
            )
            fig.update_xaxes(title="Tokens Used")
            fig.update_yaxes(title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Processing time vs agents
            fig = px.scatter(
                df,
                x='agents_count',
                y='processing_time',
                size='tokens_used',
                color='confidence',
                title="Processing Time vs Agent Count",
                hover_data=['workflow_id', 'status']
            )
            fig.update_xaxes(title="Number of Agents")
            fig.update_yaxes(title="Processing Time (s)")
            st.plotly_chart(fig, use_container_width=True)
