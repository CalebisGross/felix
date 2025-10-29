"""
Advanced Analytics for Felix Framework (VIEW-ONLY)

Strategic insights and performance analytics:
- Strategic Insights: Trust score and efficiency trends
- Performance Analytics: Phase performance and risk heatmaps

IMPORTANT: This is a monitoring interface only. No agent control or interaction.
For agent task sending and control, use the tkinter GUI.

NOTE: Agent metrics are available on the Dashboard page.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Page configuration
st.set_page_config(
    page_title="Advanced Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Advanced Analytics")
st.markdown("Strategic insights and performance analytics (read-only)")

st.info("💡 **Agent metrics and details are available on the Dashboard page**")

# Section 1: Strategic Insights
st.divider()
st.subheader("📊 Strategic Insights")
st.markdown("High-level metrics for strategic decision making")

try:
    from streamlit_gui.components.trust_score import render_trust_score
    from streamlit_gui.components.efficiency_trend import render_efficiency_trend

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Autonomy Trust Score")
        render_trust_score("felix_system_actions.db", days=7)

    with col2:
        st.subheader("Workflow Efficiency Trend")
        render_efficiency_trend("felix_workflow_history.db", limit=20)

except ImportError as e:
    st.warning("Strategic insights components not available. Ensure trust_score.py and efficiency_trend.py are in streamlit_gui/components/")
except Exception as e:
    st.error(f"Error loading strategic insights: {e}")

st.divider()

# Section 2: Performance Analytics
st.subheader("📈 Performance Analytics")
st.markdown("Deep-dive analysis of system performance and risk patterns")

st.divider()
st.markdown("**Helix Phase Performance Breakdown**")
st.markdown("*Identify bottlenecks in workflow progression*")

try:
    from streamlit_gui.components.phase_performance import render_phase_performance
    render_phase_performance("felix_agent_performance.db")
except ImportError:
    st.warning("Phase performance component not available. Ensure phase_performance.py is in streamlit_gui/components/")
except Exception as e:
    st.error(f"Error loading phase performance: {e}")

st.divider()
st.markdown("**Command Risk Heatmap**")
st.markdown("*Analyze risk distribution and identify patterns for trust rule updates*")

try:
    from streamlit_gui.components.risk_heatmap import render_risk_heatmap
    render_risk_heatmap("felix_system_actions.db")
except ImportError:
    st.warning("Risk heatmap component not available. Ensure risk_heatmap.py is in streamlit_gui/components/")
except Exception as e:
    st.error(f"Error loading risk heatmap: {e}")

# Footer
st.markdown("---")
st.caption("Advanced Analytics Dashboard - Felix Framework | Comprehensive insights and agent monitoring (read-only)")