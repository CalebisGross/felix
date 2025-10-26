## Phase 3: Complete Integration
**Goal:** Polish and advanced features
**Timeline:** Nice-to-have enhancements
**Outcome:** Complete feature parity with main branch

### 3.1 Truth Assessment Visualization

**File:** `streamlit_gui/components/truth_assessment_display.py`

```python
"""
Truth Assessment Display Component.

Visualizes truth assessment workflow results including
source validation and claim verification.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List


class TruthAssessmentDisplay:
    """Display truth assessment workflow results."""

    def __init__(self, db_reader):
        self.db_reader = db_reader

    def render(self):
        """Render truth assessment interface."""
        st.header("✓ Truth Assessment")
        st.markdown("Source validation and claim verification analysis")

        # Get truth assessment workflows
        workflows = self._get_truth_workflows()

        if not workflows:
            st.info("""
            No truth assessment workflows found.

            Truth assessment analyzes claims and validates sources
            using the workflows/truth_assessment.py module.
            """)
            return

        # Display workflows
        for workflow in workflows:
            self._render_assessment(workflow)

    def _get_truth_workflows(self) -> List[Dict[str, Any]]:
        """Get workflows that used truth assessment."""
        # Query workflow history for truth assessment tasks
        query = """
            SELECT *
            FROM workflow_outputs
            WHERE task_input LIKE '%truth%'
               OR task_input LIKE '%validate%'
               OR task_input LIKE '%verify%'
            ORDER BY created_at DESC
            LIMIT 20
        """

        df = self.db_reader._read_query("workflow_history", query)
        if df is None or df.empty:
            return []

        return df.to_dict('records')

    def _render_assessment(self, workflow: Dict[str, Any]):
        """Render single truth assessment."""
        with st.expander(f"Assessment #{workflow['workflow_id']} - {workflow['created_at']}"):
            st.markdown(f"**Task:** {workflow['task_input']}")
            st.markdown(f"**Confidence:** {workflow['confidence']*100:.1f}%")
            st.text_area(
                "Assessment Result",
                value=workflow['final_synthesis'],
                height=150,
                disabled=True
            )
```

### 3.2 Configuration Page Web Search Display

**File:** `streamlit_gui/pages/2_Configuration.py`

**Add web search section:**
```python
def display_web_search_config():
    """Display web search configuration (read-only)."""
    st.subheader("🔍 Web Search Configuration")

    # Try to load config from felix_gui_config.json
    config_path = Path("felix_gui_config.json")

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        web_config = config.get('web_search', {})

        col1, col2 = st.columns(2)

        with col1:
            enabled = web_config.get('enabled', False)
            st.metric("Web Search", "Enabled" if enabled else "Disabled")

            provider = web_config.get('provider', 'duckduckgo')
            st.metric("Provider", provider.title())

        with col2:
            max_results = web_config.get('max_results_per_query', 5)
            st.metric("Max Results/Query", max_results)

            max_queries = web_config.get('max_queries_per_agent', 3)
            st.metric("Max Queries/Agent", max_queries)

        if provider == 'searxng':
            url = web_config.get('searxng_url', 'Not configured')
            st.text_input("SearxNG URL", value=url, disabled=True)
    else:
        st.info("No web search configuration found. Configure via Tkinter GUI Settings tab.")

# Add to main():
st.divider()
display_web_search_config()
```

### 3.3 Enhanced Analytics

**File:** `streamlit_gui/pages/1_Dashboard.py`

**Add advanced analytics tab:**
```python
with tab7:  # New advanced tab
    st.subheader("📊 Advanced Analytics")

    # Hypothesis performance tracking
    st.markdown("### Hypothesis Performance Tracker")

    # Load latest benchmark results
    runner = get_benchmark_runner()
    results = runner.get_latest_results()

    if results:
        # Create comparison chart
        hypotheses = []
        targets = []
        actuals = []

        if 'H1' in results:
            hypotheses.append('H1')
            targets.append(20.0)
            actuals.append(results['H1']['average_improvement'] * 100)

        if 'H2' in results:
            hypotheses.append('H2')
            targets.append(15.0)
            actuals.append(results['H2']['average_improvement'] * 100)

        if 'H3' in results:
            hypotheses.append('H3')
            targets.append(25.0)
            actuals.append(results['H3']['average_improvement'] * 100)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Target',
            x=hypotheses,
            y=targets,
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            name='Actual',
            x=hypotheses,
            y=actuals,
            marker_color='green'
        ))

        fig.update_layout(
            title="Hypothesis Validation: Target vs Actual",
            yaxis_title="Improvement (%)",
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run hypothesis validation tests to see analytics")

    # Agent efficiency metrics
    st.markdown("### Agent Efficiency Matrix")

    agent_df = db_reader.get_agent_metrics()
    if not agent_df.empty:
        # Efficiency = outputs * avg_confidence
        agent_df['efficiency'] = agent_df['output_count'] * agent_df['avg_confidence']

        fig = px.scatter(
            agent_df,
            x='output_count',
            y='avg_confidence',
            size='efficiency',
            color='efficiency',
            hover_data=['agent_id'],
            title="Agent Efficiency Matrix",
            color_continuous_scale='Viridis'
        )

        st.plotly_chart(fig, use_container_width=True)
```

### 3.4 Testing Phase 3

**Test Checklist:**
- [ ] Truth assessment display renders
- [ ] Web search config shows in Configuration page
- [ ] Advanced analytics charts load
- [ ] All integrated features work together
- [ ] Performance is acceptable with large datasets
- [ ] No memory leaks with long-running sessions
- [ ] All visualizations are readable and clear

---

