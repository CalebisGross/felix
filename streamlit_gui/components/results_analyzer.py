"""Results analyzer component for Felix Framework."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import statistics


def analyze_success_patterns(workflows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in successful vs failed workflows.

    Args:
        workflows: List of workflow results

    Returns:
        Pattern analysis dictionary
    """
    successful = [w for w in workflows if w.get('success', False)]
    failed = [w for w in workflows if not w.get('success', False)]

    analysis = {
        'total': len(workflows),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': (len(successful) / len(workflows) * 100) if workflows else 0,
        'success_patterns': {},
        'failure_patterns': {}
    }

    # Analyze successful patterns
    if successful:
        agent_counts = [w.get('agent_count', 0) for w in successful]
        analysis['success_patterns'] = {
            'avg_agents': statistics.mean(agent_counts) if agent_counts else 0,
            'median_agents': statistics.median(agent_counts) if agent_counts else 0,
            'common_domains': extract_common_domains(successful),
            'avg_confidence': calculate_avg_confidence(successful)
        }

    # Analyze failure patterns
    if failed:
        agent_counts = [w.get('agent_count', 0) for w in failed]
        analysis['failure_patterns'] = {
            'avg_agents': statistics.mean(agent_counts) if agent_counts else 0,
            'median_agents': statistics.median(agent_counts) if agent_counts else 0,
            'common_issues': extract_common_issues(failed),
            'avg_confidence': calculate_avg_confidence(failed)
        }

    return analysis


def extract_common_domains(workflows: List[Dict[str, Any]]) -> List[str]:
    """
    Extract common domains from workflows.

    Args:
        workflows: List of workflow data

    Returns:
        List of common domains
    """
    domains = []
    for w in workflows:
        if 'domain' in w:
            domains.append(w['domain'])
        elif 'domains' in w:
            domains.extend(w['domains'])

    if domains:
        domain_counts = Counter(domains)
        return [domain for domain, _ in domain_counts.most_common(3)]
    return []


def extract_common_issues(workflows: List[Dict[str, Any]]) -> List[str]:
    """
    Extract common issues from failed workflows.

    Args:
        workflows: List of failed workflow data

    Returns:
        List of common issues
    """
    issues = []
    for w in workflows:
        if 'error' in w:
            issues.append(w['error'])
        elif 'issues' in w:
            issues.extend(w['issues'])

    if issues:
        issue_counts = Counter(issues)
        return [issue for issue, _ in issue_counts.most_common(3)]
    return ['Unknown issues']


def calculate_avg_confidence(workflows: List[Dict[str, Any]]) -> float:
    """
    Calculate average confidence from workflows.

    Args:
        workflows: List of workflow data

    Returns:
        Average confidence score
    """
    confidences = []
    for w in workflows:
        if 'confidence' in w:
            confidences.append(w['confidence'])
        elif 'avg_confidence' in w:
            confidences.append(w['avg_confidence'])

    return statistics.mean(confidences) if confidences else 0.0


def create_hypothesis_validation_chart(hypothesis_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create chart showing hypothesis validation results.

    Args:
        hypothesis_data: Hypothesis test results

    Returns:
        Plotly figure
    """
    # Extract data for chart
    hypotheses = []
    expected = []
    actual = []
    status = []

    for h_name, h_data in hypothesis_data.items():
        hypotheses.append(h_name)
        expected.append(h_data.get('expected', 0))
        actual.append(h_data.get('actual', 0))
        status.append('Pass' if h_data.get('validated', False) else 'Fail')

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Expected',
        x=hypotheses,
        y=expected,
        marker_color='lightblue',
        text=[f"{v:.1f}%" for v in expected],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        name='Actual',
        x=hypotheses,
        y=actual,
        marker_color=['green' if s == 'Pass' else 'red' for s in status],
        text=[f"{v:.1f}%" for v in actual],
        textposition='auto'
    ))

    fig.update_layout(
        title="Hypothesis Validation Results",
        xaxis_title="Hypothesis",
        yaxis_title="Performance Gain (%)",
        barmode='group',
        height=400,
        showlegend=True
    )

    return fig


def create_performance_heatmap(performance_data: pd.DataFrame) -> go.Figure:
    """
    Create performance heatmap across different metrics.

    Args:
        performance_data: DataFrame with performance metrics

    Returns:
        Plotly heatmap figure
    """
    if performance_data.empty:
        return go.Figure().add_annotation(
            text="No performance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Prepare data for heatmap
    metrics = ['confidence', 'latency', 'throughput', 'accuracy']
    available_metrics = [m for m in metrics if m in performance_data.columns]

    if not available_metrics:
        return go.Figure()

    # Normalize data for visualization
    normalized_data = performance_data[available_metrics].copy()

    for col in available_metrics:
        col_data = normalized_data[col]
        if col_data.std() != 0:
            normalized_data[col] = (col_data - col_data.mean()) / col_data.std()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data.T.values,
        x=list(range(len(normalized_data))),
        y=available_metrics,
        colorscale='RdYlGn',
        colorbar=dict(title="Z-Score"),
        hoverongaps=False
    ))

    fig.update_layout(
        title="Performance Metrics Heatmap",
        xaxis_title="Test Run",
        yaxis_title="Metric",
        height=400
    )

    return fig


def calculate_statistical_significance(
    baseline: List[float],
    treatment: List[float],
    alpha: float = 0.05
) -> Tuple[bool, float, str]:
    """
    Calculate statistical significance between baseline and treatment.

    Args:
        baseline: Baseline measurements
        treatment: Treatment measurements
        alpha: Significance level

    Returns:
        Tuple of (is_significant, p_value, interpretation)
    """
    if not baseline or not treatment:
        return False, 1.0, "Insufficient data"

    try:
        from scipy import stats

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(baseline, treatment)

        is_significant = p_value < alpha

        if is_significant:
            effect_size = (np.mean(treatment) - np.mean(baseline)) / np.std(baseline)

            if abs(effect_size) < 0.2:
                interpretation = "Statistically significant but small effect"
            elif abs(effect_size) < 0.5:
                interpretation = "Statistically significant with medium effect"
            else:
                interpretation = "Statistically significant with large effect"
        else:
            interpretation = "No statistically significant difference"

        return is_significant, p_value, interpretation

    except ImportError:
        # Fallback if scipy not available
        mean_diff = abs(np.mean(treatment) - np.mean(baseline))
        threshold = 0.1 * np.mean(baseline)

        if mean_diff > threshold:
            return True, 0.04, "Significant difference detected (simplified test)"
        else:
            return False, 0.5, "No significant difference (simplified test)"


def create_regression_analysis(data: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """
    Create regression analysis plot.

    Args:
        data: DataFrame with data
        x_col: X column name
        y_col: Y column name

    Returns:
        Plotly figure with regression line
    """
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return go.Figure().add_annotation(
            text="Insufficient data for regression analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Calculate regression line
    x = data[x_col].values
    y = data[y_col].values

    # Simple linear regression
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    # Create scatter plot with regression line
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(size=8, color='blue', opacity=0.6)
    ))

    # Add regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = p(x_line)

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f'Regression (y={z[0]:.3f}x+{z[1]:.3f})',
        line=dict(color='red', width=2)
    ))

    # Calculate R-squared
    residuals = y - p(x)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    fig.update_layout(
        title=f"{y_col} vs {x_col} (R² = {r_squared:.3f})",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=400,
        showlegend=True
    )

    return fig


def display_test_summary(test_results: Dict[str, Any]):
    """
    Display comprehensive test summary.

    Args:
        test_results: Dictionary of test results
    """
    st.markdown("### Test Execution Summary")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Tests Run",
            test_results.get('total_tests', 0),
            delta=test_results.get('new_tests', 0)
        )

    with col2:
        pass_rate = test_results.get('pass_rate', 0)
        st.metric(
            "Pass Rate",
            f"{pass_rate:.1f}%",
            delta=f"{test_results.get('pass_rate_change', 0):.1f}%"
        )

    with col3:
        avg_duration = test_results.get('avg_duration', 0)
        st.metric(
            "Avg Duration",
            f"{avg_duration:.2f}s",
            delta=f"{test_results.get('duration_change', 0):.2f}s"
        )

    with col4:
        coverage = test_results.get('coverage', 0)
        st.metric(
            "Coverage",
            f"{coverage:.1f}%",
            delta=f"{test_results.get('coverage_change', 0):.1f}%"
        )

    # Detailed breakdown
    if 'breakdown' in test_results:
        st.markdown("### Test Category Breakdown")

        breakdown_df = pd.DataFrame(test_results['breakdown'])
        fig = px.pie(
            breakdown_df,
            values='count',
            names='category',
            title='Test Distribution by Category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, width='stretch')


def create_confidence_interval_chart(data: List[float], confidence_level: float = 0.95) -> go.Figure:
    """
    Create confidence interval visualization.

    Args:
        data: List of measurements
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Plotly figure
    """
    if not data:
        return go.Figure().add_annotation(
            text="No data for confidence interval",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    mean = np.mean(data)
    std = np.std(data)
    n = len(data)

    # Calculate confidence interval
    z_score = 1.96 if confidence_level == 0.95 else 2.576  # For 95% or 99%
    margin = z_score * (std / np.sqrt(n))

    lower = mean - margin
    upper = mean + margin

    # Create visualization
    fig = go.Figure()

    # Add data points
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='markers',
        name='Data Points',
        marker=dict(size=6, color='blue', opacity=0.5)
    ))

    # Add mean line
    fig.add_hline(
        y=mean,
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mean: {mean:.3f}"
    )

    # Add confidence interval
    fig.add_hrect(
        y0=lower,
        y1=upper,
        fillcolor="lightgreen",
        opacity=0.3,
        line_width=0,
        annotation_text=f"{confidence_level*100:.0f}% CI"
    )

    fig.update_layout(
        title=f"Data with {confidence_level*100:.0f}% Confidence Interval",
        xaxis_title="Sample",
        yaxis_title="Value",
        height=400
    )

    return fig