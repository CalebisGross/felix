## Phase 2: Enhanced Monitoring
**Goal:** Add web search monitoring and testing integration
**Timeline:** Secondary priority after Phase 1
**Outcome:** Real-time web search tracking + GUI test runner
**Status:** ✅ COMPLETED (Section 2.1 only - 2.2 skipped due to redundancy with Benchmarking page)
**Completed:** 2025-10-26

---

## Implementation Summary

### ✅ Section 2.1: Web Search Monitoring (COMPLETE)

**Files Modified:**
1. `streamlit_gui/backend/db_reader.py` - Added 2 methods (lines 441-558)
   - `get_web_search_activity(limit)` - Extracts web search data from knowledge DB
   - `get_web_search_stats()` - Calculates aggregate statistics

2. `streamlit_gui/components/web_search_monitor.py` - New component (154 lines)
   - WebSearchMonitor class with render methods
   - Summary metrics, search activity table, analytics charts
   - Handles empty state gracefully

3. `streamlit_gui/pages/1_Dashboard.py` - Updated with web search integration
   - Added web search metrics section (4 metrics)
   - Added 6th tab "Web Search" with WebSearchMonitor component

4. `streamlit_gui/components/__init__.py` - Updated exports
   - Added WebSearchMonitor to imports and __all__ list

**Key Features:**
- Real-time web search activity monitoring from knowledge DB
- Search statistics: total searches, unique queries, avg results, 24h activity
- Interactive table with slider (10-100 searches)
- 3 analytics charts: frequency, distribution, top queries
- Empty state handling with helpful info messages

### ⏭️ Section 2.2: Test Runner Integration (SKIPPED)

**Reason:** Benchmarking page (4_Benchmarking.py) already provides comprehensive test running functionality with RealBenchmarkRunner integration. Adding test runner to Testing page would create redundancy.

**Page Responsibilities:**
- **Benchmarking page**: Run new hypothesis validation tests (H1, H2, H3)
- **Testing page**: Analyze historical workflow data and performance trends
- Clear separation maintains focused user experience

---

### 2.1 Add Web Search Monitoring

#### Update Database Reader for Web Search

**File:** `streamlit_gui/backend/db_reader.py`

**Add method:**
```python
def get_web_search_activity(self, limit: int = 50) -> pd.DataFrame:
    """
    Extract web search activity from knowledge entries.

    Research agents store search queries and sources in metadata.

    Args:
        limit: Maximum number of entries to analyze

    Returns:
        DataFrame with web search activity
    """
    query = f"""
        SELECT
            knowledge_id,
            source_agent as agent_id,
            domain,
            created_at as timestamp,
            content_json
        FROM knowledge_entries
        WHERE source_agent LIKE 'research%'
        ORDER BY created_at DESC
        LIMIT {limit}
    """

    df = self._read_query("knowledge", query)
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'agent_id', 'timestamp', 'query', 'sources', 'results_count'
        ])

    # Parse content_json to extract search info
    search_records = []
    for _, row in df.iterrows():
        try:
            content = json.loads(row['content_json'])

            # Look for search queries and sources in metadata
            if 'search_queries' in content:
                for query in content['search_queries']:
                    search_records.append({
                        'agent_id': row['agent_id'],
                        'timestamp': row['timestamp'],
                        'query': query,
                        'sources': content.get('information_sources', []),
                        'results_count': len(content.get('web_search_results', []))
                    })
        except:
            continue

    if not search_records:
        return pd.DataFrame(columns=[
            'agent_id', 'timestamp', 'query', 'sources', 'results_count'
        ])

    return pd.DataFrame(search_records)

def get_web_search_stats(self) -> Dict[str, Any]:
    """
    Get web search usage statistics.

    Returns:
        Dictionary with search statistics
    """
    df = self.get_web_search_activity(limit=1000)

    if df.empty:
        return {
            'total_searches': 0,
            'unique_queries': 0,
            'avg_results_per_search': 0.0,
            'total_sources': 0,
            'searches_last_24h': 0
        }

    # Calculate stats
    total_searches = len(df)
    unique_queries = df['query'].nunique()
    avg_results = df['results_count'].mean()

    # Count unique sources
    all_sources = []
    for sources_list in df['sources']:
        if isinstance(sources_list, list):
            all_sources.extend(sources_list)
    total_sources = len(set(all_sources))

    # Recent searches
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff = datetime.now() - timedelta(hours=24)
    searches_24h = len(df[df['timestamp'] >= cutoff])

    return {
        'total_searches': total_searches,
        'unique_queries': unique_queries,
        'avg_results_per_search': avg_results,
        'total_sources': total_sources,
        'searches_last_24h': searches_24h
    }
```

#### Create Web Search Display Component

**File:** `streamlit_gui/components/web_search_monitor.py`

```python
"""
Web Search Monitor Component for Streamlit GUI.

Displays web search activity and statistics for Research agents.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any


class WebSearchMonitor:
    """
    Web search activity monitoring component.

    Visualizes web search queries, results, and sources
    from Research agent activity.
    """

    def __init__(self, db_reader):
        """
        Initialize monitor.

        Args:
            db_reader: DatabaseReader instance with web search methods
        """
        self.db_reader = db_reader

    def render(self):
        """Render complete web search monitor."""
        st.header("🔍 Web Search Activity")
        st.markdown("Monitor Research agent web search queries and results")

        # Check if web search data exists
        stats = self.db_reader.get_web_search_stats()

        if stats['total_searches'] == 0:
            st.info("""
            No web search activity detected yet.

            Web search features:
            - Requires DuckDuckGo (ddgs package) or SearxNG
            - Research agents perform searches at helix depth 0.0-0.3
            - Enable in Felix GUI Settings > Web Search Configuration
            """)
            return

        # Summary Metrics
        self._render_summary_metrics(stats)

        st.divider()

        # Search Activity Table
        self._render_search_activity()

        st.divider()

        # Charts
        self._render_search_charts()

    def _render_summary_metrics(self, stats: Dict[str, Any]):
        """Render web search summary metrics."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Searches", stats['total_searches'])

        with col2:
            st.metric("Unique Queries", stats['unique_queries'])

        with col3:
            st.metric("Avg Results/Search", f"{stats['avg_results_per_search']:.1f}")

        with col4:
            st.metric("Searches (24h)", stats['searches_last_24h'])

    def _render_search_activity(self):
        """Render recent search activity table."""
        st.subheader("Recent Search Queries")

        limit = st.slider("Show last N searches", 10, 100, 50)

        df = self.db_reader.get_web_search_activity(limit=limit)

        if df.empty:
            st.info("No search activity found")
            return

        # Format for display
        display_df = df.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['sources_preview'] = display_df['sources'].apply(
            lambda x: ', '.join(x[:2]) + ('...' if len(x) > 2 else '') if isinstance(x, list) else ''
        )

        # Show table
        st.dataframe(
            display_df[['agent_id', 'timestamp', 'query', 'results_count', 'sources_preview']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "agent_id": "Agent",
                "timestamp": "Time",
                "query": st.column_config.TextColumn("Search Query", width="large"),
                "results_count": "Results",
                "sources_preview": st.column_config.TextColumn("Sources", width="medium")
            }
        )

    def _render_search_charts(self):
        """Render web search analytics charts."""
        st.subheader("Search Analytics")

        df = self.db_reader.get_web_search_activity(limit=200)

        if df.empty:
            return

        tab1, tab2, tab3 = st.tabs([
            "Search Frequency",
            "Results Distribution",
            "Top Queries"
        ])

        with tab1:
            # Searches over time
            df_time = df.copy()
            df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
            df_time['date'] = df_time['timestamp'].dt.date

            daily_counts = df_time.groupby('date').size().reset_index(name='count')

            fig = px.bar(
                daily_counts,
                x='date',
                y='count',
                title="Searches Per Day",
                labels={'date': 'Date', 'count': 'Number of Searches'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Results count distribution
            fig = px.histogram(
                df,
                x='results_count',
                title="Search Results Distribution",
                nbins=20,
                labels={'results_count': 'Results per Search', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Top queries
            top_queries = df['query'].value_counts().head(10)

            fig = px.bar(
                x=top_queries.values,
                y=top_queries.index,
                orientation='h',
                title="Top 10 Search Queries",
                labels={'x': 'Frequency', 'y': 'Query'}
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
```

#### Integrate into Dashboard

**File:** `streamlit_gui/pages/1_Dashboard.py`

**Add web search metrics to header:**
```python
# After existing header metrics:
st.divider()
st.subheader("🔍 Web Search Activity")

web_stats = db_reader.get_web_search_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Searches", web_stats['total_searches'])

with col2:
    st.metric("Last 24h", web_stats['searches_last_24h'])

with col3:
    st.metric("Unique Queries", web_stats['unique_queries'])

with col4:
    st.metric("Total Sources", web_stats['total_sources'])
```

**Add web search tab:**
```python
# Add to tabs:
with tab6:  # New tab
    from streamlit_gui.components.web_search_monitor import WebSearchMonitor

    search_monitor = WebSearchMonitor(db_reader)
    search_monitor.render()
```

### 2.2 Integrate Test Runner in Testing Page

#### Update Testing Page

**File:** `streamlit_gui/pages/3_Testing.py`

**Add test runner section:**
```python
# Add after imports:
from streamlit_gui.backend.real_benchmark_runner import RealBenchmarkRunner

# Add cached runner:
@st.cache_resource
def get_benchmark_runner():
    return RealBenchmarkRunner()

# Add new section in main():
def main():
    st.title("🧪 Felix Testing & Validation")

    # ... existing code ...

    st.divider()
    st.header("🚀 Run Hypothesis Validation")
    st.markdown("""
    Execute comprehensive hypothesis validation tests using the Felix test suite.
    Results are saved to `tests/results/` directory.
    """)

    runner = get_benchmark_runner()

    if not runner.validate_test_suite_available():
        st.warning("Test suite not available at tests/run_hypothesis_validation.py")
    else:
        col1, col2 = st.columns(2)

        with col1:
            test_hypothesis = st.selectbox(
                "Select Hypothesis",
                ["all", "H1", "H2", "H3"],
                help="Choose which hypothesis to validate"
            )

        with col2:
            test_iterations = st.number_input(
                "Iterations",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of test iterations"
            )

        if st.button("Run Tests", type="primary"):
            with st.spinner("Running validation tests..."):
                results = runner.run_hypothesis_validation(
                    hypothesis=test_hypothesis,
                    iterations=test_iterations,
                    use_real_llm=False
                )

                if 'error' not in results:
                    st.success("Tests completed!")
                    st.json(results)
                else:
                    st.error(f"Test failed: {results['error']}")
```

### 2.3 Testing Phase 2

**Test Checklist (Section 2.1 only):**
- [x] Web search metrics display on Dashboard header
- [x] Web search activity table shows queries with slider control
- [x] Web search charts render properly (3 tabs)
- [x] Handles case of no search data gracefully (info message)
- [x] No errors when databases are empty (returns zeros/empty DataFrames)
- [x] Component follows existing patterns (WorkflowHistoryViewer)
- [x] All exports updated in __init__.py

**Section 2.2 Testing:** N/A - Skipped (Benchmarking page handles test execution)

---

## Next Steps

Phase 2 is complete. To use the web search monitoring:

1. **Start Streamlit GUI**: `streamlit run streamlit_gui/Home.py`
2. **Navigate to Dashboard**: Click "Dashboard" in sidebar
3. **View Web Search**:
   - Scroll to "Web Search Activity" metrics section
   - Click "Web Search" tab to see detailed monitoring
4. **Test with Real Data**: Run Felix workflows with web search enabled to populate data

The monitoring will automatically display web search activity from research agents stored in `felix_knowledge.db` with `domain='web_search'`.

---

