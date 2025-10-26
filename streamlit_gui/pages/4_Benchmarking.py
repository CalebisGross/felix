"""Benchmarking page integrating with tests/run_hypothesis_validation.py."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit_gui.backend.real_benchmark_runner import RealBenchmarkRunner

st.set_page_config(
    page_title="Felix Benchmarking",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def get_benchmark_runner():
    """Get cached benchmark runner instance."""
    return RealBenchmarkRunner()


def display_hypothesis_results(results: dict, hypothesis: str, target: float):
    """Display results for a single hypothesis with box plots."""
    st.subheader(f"{hypothesis}: {results.get('hypothesis', 'Unknown')}")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_improvement = results.get('average_improvement', 0)
        actual_pct = avg_improvement * 100 if avg_improvement < 1 else avg_improvement
        delta = actual_pct - target
        st.metric(
            "Average Improvement",
            f"{actual_pct:.1f}%",
            delta=f"{delta:+.1f}% vs target"
        )

    with col2:
        target_pct = target
        status = "✅ PASSED" if results.get('passed', False) else "❌ FAILED"
        st.metric("Target", f"{target_pct:.0f}%", status)

    with col3:
        success_rate = results.get('success_rate', 0)
        st.metric("Success Rate", f"{success_rate:.1f}%")

    with col4:
        num_tests = results.get('num_tests', 0)
        st.metric("Total Tests", num_tests)

    # Detailed results in expander
    with st.expander(f"View {hypothesis} Details"):
        # Show raw data if available
        if 'h1_results' in results or 'h2_results' in results or 'h3_results' in results:
            # Extract test results
            test_results = []
            for key in ['h1_results', 'h2_results', 'h3_results']:
                if key in results and results[key]:
                    test_results.extend(results[key])

            if test_results:
                df_results = pd.DataFrame([
                    {
                        'Test Name': r.get('test_name', 'Unknown'),
                        'Improvement': f"{r.get('improvement_percentage', 0):.1f}%",
                        'Passed': '✅' if r.get('passed', False) else '❌',
                        'Baseline': f"{r.get('baseline_value', 0):.2f}",
                        'Treatment': f"{r.get('treatment_value', 0):.2f}"
                    }
                    for r in test_results
                ])
                st.dataframe(df_results, use_container_width=True)

        # Box plot visualization if test data available
        if results.get('baseline_data') and results.get('treatment_data'):
            fig = go.Figure()

            # Baseline box plot
            fig.add_trace(go.Box(
                y=results['baseline_data'],
                name="Baseline",
                marker_color='lightblue',
                boxmean='sd'
            ))

            # Treatment box plot
            fig.add_trace(go.Box(
                y=results['treatment_data'],
                name="Felix",
                marker_color='green',
                boxmean='sd'
            ))

            fig.update_layout(
                title=f"{hypothesis} Performance Distribution",
                yaxis_title="Performance Metric",
                showlegend=True,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📊 Felix Hypothesis Validation")
    st.markdown("""
    Run comprehensive hypothesis validation tests using the Felix test suite.
    Tests validate H1 (20% workload improvement), H2 (15% communication efficiency),
    and H3 (25% memory compression gains).
    """)

    runner = get_benchmark_runner()

    # Check if test suite available
    if not runner.validate_test_suite_available():
        st.error("❌ Test suite not found at tests/run_hypothesis_validation.py")
        st.info("The test suite should be available in the tests/ directory.")
        return

    st.success("✅ Test suite available and ready")

    # Test Configuration
    st.header("Test Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        hypothesis = st.selectbox(
            "Hypothesis to Test",
            ["all", "H1", "H2", "H3"],
            help="Select which hypothesis to validate"
        )

    with col2:
        iterations = st.number_input(
            "Iterations",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of test iterations (more = better statistics)"
        )

    with col3:
        use_real_llm = st.checkbox(
            "Use Real LLM",
            value=False,
            help="Use LM Studio on port 1234 (must be running)"
        )

    # Hypothesis explanations
    with st.expander("ℹ️ Understanding the Hypotheses"):
        st.markdown("""
        **H1: Helical Progression (Target: 20% improvement)**
        - Tests workload distribution and adaptive behavior
        - Measures efficiency gains from helical geometry vs linear progression

        **H2: Hub-Spoke Communication (Target: 15% improvement)**
        - Tests communication efficiency and resource allocation
        - Measures O(N) hub-spoke vs O(N²) mesh networking

        **H3: Memory Compression (Target: 25% improvement)**
        - Tests memory compression and attention focus
        - Measures latency reduction from context compression
        """)

    # Run Tests
    if st.button("🚀 Run Validation Tests", type="primary"):
        with st.spinner(f"Running {hypothesis} validation tests..."):
            progress_placeholder = st.empty()

            def progress_callback(msg):
                progress_placeholder.info(msg)

            results = runner.run_hypothesis_validation(
                hypothesis=hypothesis,
                iterations=iterations,
                use_real_llm=use_real_llm,
                callback=progress_callback
            )

            progress_placeholder.empty()

            if 'error' in results:
                st.error(f"❌ Error: {results['error']}")
                if 'message' in results:
                    st.write(results['message'])
                if 'stdout' in results:
                    with st.expander("View stdout"):
                        st.code(results['stdout'])
                if 'stderr' in results:
                    with st.expander("View stderr"):
                        st.code(results['stderr'])
            else:
                st.success("✅ Tests completed successfully!")
                st.session_state['latest_results'] = results

    # Display Latest Results
    st.divider()
    st.header("Latest Results")

    if 'latest_results' not in st.session_state:
        # Try to load from disk
        latest = runner.get_latest_results()
        if latest:
            st.session_state['latest_results'] = latest
        else:
            st.info("No results available. Run tests above to generate results.")
            return

    results = st.session_state['latest_results']

    # Check if we have summary data
    if 'summary' not in results:
        st.warning("Results format not recognized. Please run tests again.")
        return

    summary = results['summary']

    # Summary Metrics
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'h1' in summary:
            h1 = summary['h1']
            h1_status = "✅" if h1.get('passed', False) else "❌"
            h1_improvement = h1.get('average_improvement', 0)
            h1_pct = h1_improvement * 100 if h1_improvement < 1 else h1_improvement
            st.metric(
                "H1: Helical Progression",
                f"{h1_status} {h1_pct:.1f}%",
                delta=f"Target: 20%"
            )

    with col2:
        if 'h2' in summary:
            h2 = summary['h2']
            h2_status = "✅" if h2.get('passed', False) else "❌"
            h2_improvement = h2.get('average_improvement', 0)
            h2_pct = h2_improvement * 100 if h2_improvement < 1 else h2_improvement
            st.metric(
                "H2: Hub-Spoke Communication",
                f"{h2_status} {h2_pct:.1f}%",
                delta=f"Target: 15%"
            )

    with col3:
        if 'h3' in summary:
            h3 = summary['h3']
            h3_status = "✅" if h3.get('passed', False) else "❌"
            h3_improvement = h3.get('average_improvement', 0)
            h3_pct = h3_improvement * 100 if h3_improvement < 1 else h3_improvement
            st.metric(
                "H3: Memory Compression",
                f"{h3_status} {h3_pct:.1f}%",
                delta=f"Target: 25%"
            )

    # Detailed Results by Hypothesis
    st.divider()
    st.subheader("Detailed Results")

    # Prepare data for visualization
    if 'h1' in summary:
        h1_data = summary['h1']
        # Add raw test results if available
        if 'h1_results' in results and results['h1_results']:
            h1_data['h1_results'] = results['h1_results']
        display_hypothesis_results(h1_data, "H1", 20.0)

    if 'h2' in summary:
        h2_data = summary['h2']
        if 'h2_results' in results and results['h2_results']:
            h2_data['h2_results'] = results['h2_results']
        display_hypothesis_results(h2_data, "H2", 15.0)

    if 'h3' in summary:
        h3_data = summary['h3']
        if 'h3_results' in results and results['h3_results']:
            h3_data['h3_results'] = results['h3_results']
        display_hypothesis_results(h3_data, "H3", 25.0)

    # Export Results
    st.divider()
    st.subheader("Export Results")

    if st.button("📥 Download JSON Report"):
        report_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="Download validation_report.json",
            data=report_json,
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
