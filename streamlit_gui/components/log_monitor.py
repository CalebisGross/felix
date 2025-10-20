"""
Log monitoring component for Felix Framework.

Provides real-time log streaming and display with auto-refresh capability.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import json
from pathlib import Path


class LogMonitor:
    """
    Monitor and display log entries with auto-refresh capability.
    """

    def __init__(self, log_file_path: Optional[str] = None, max_lines: int = 100):
        """
        Initialize the log monitor.

        Args:
            log_file_path: Path to log file to monitor
            max_lines: Maximum number of lines to display
        """
        self.log_file_path = log_file_path or "felix.log"
        self.max_lines = max_lines
        self.last_position = 0

    def read_new_logs(self) -> List[str]:
        """
        Read new log entries since last check.

        Returns:
            List of new log lines
        """
        if not Path(self.log_file_path).exists():
            return []

        new_lines = []
        try:
            with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to last position
                f.seek(self.last_position)

                # Read new lines
                for line in f:
                    new_lines.append(line.strip())

                # Update position
                self.last_position = f.tell()

        except Exception as e:
            st.error(f"Error reading log file: {e}")

        return new_lines[-self.max_lines:] if len(new_lines) > self.max_lines else new_lines

    def display_logs(self, container=None):
        """
        Display log entries in Streamlit.

        Args:
            container: Streamlit container to display logs in
        """
        logs = self.read_new_logs()

        if not logs:
            if container:
                container.info("No new log entries")
            else:
                st.info("No new log entries")
            return

        # Parse and format logs
        formatted_logs = []
        for log in logs:
            formatted = self._format_log_entry(log)
            formatted_logs.append(formatted)

        # Display in container or main area
        target = container if container else st

        for entry in formatted_logs:
            if entry['level'] == 'ERROR':
                target.error(entry['message'])
            elif entry['level'] == 'WARNING':
                target.warning(entry['message'])
            elif entry['level'] == 'INFO':
                target.info(entry['message'])
            else:
                target.text(entry['message'])

    def _format_log_entry(self, log_line: str) -> Dict[str, str]:
        """
        Format a log entry for display.

        Args:
            log_line: Raw log line

        Returns:
            Formatted log entry dict
        """
        # Simple parsing - can be enhanced based on actual log format
        level = 'INFO'

        if 'ERROR' in log_line.upper():
            level = 'ERROR'
        elif 'WARNING' in log_line.upper() or 'WARN' in log_line.upper():
            level = 'WARNING'
        elif 'DEBUG' in log_line.upper():
            level = 'DEBUG'

        return {
            'level': level,
            'message': log_line,
            'timestamp': datetime.now().isoformat()
        }


def create_auto_refresh_component(refresh_interval: int = 5):
    """
    Create an auto-refresh component for the dashboard.

    Args:
        refresh_interval: Refresh interval in seconds

    Returns:
        Boolean indicating if auto-refresh is enabled
    """
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=False)

    with col2:
        if auto_refresh:
            interval = st.slider(
                "Refresh interval (seconds)",
                min_value=1,
                max_value=60,
                value=refresh_interval,
                key="refresh_interval"
            )
        else:
            interval = refresh_interval

    with col3:
        if auto_refresh:
            st.info(f"Dashboard refreshes every {interval} seconds")

            # Add countdown
            placeholder = st.empty()
            for remaining in range(interval, 0, -1):
                placeholder.text(f"Refreshing in {remaining}...")
                time.sleep(1)

            placeholder.empty()
            st.rerun()

    return auto_refresh


def display_activity_feed(activities: List[Dict[str, Any]], max_items: int = 20):
    """
    Display a real-time activity feed.

    Args:
        activities: List of activity entries
        max_items: Maximum items to display
    """
    st.subheader("📢 Activity Feed")

    if not activities:
        st.info("No recent activities")
        return

    # Limit to max items
    recent_activities = activities[:max_items]

    for activity in recent_activities:
        # Create activity card
        timestamp = activity.get('timestamp', '')
        agent = activity.get('agent_id', 'System')
        action = activity.get('action', 'Unknown action')
        confidence = activity.get('confidence', 0)

        # Format timestamp
        try:
            ts = pd.to_datetime(timestamp, unit='s')
            time_str = ts.strftime('%H:%M:%S')
        except:
            time_str = str(timestamp)

        # Color based on confidence
        if confidence > 0.8:
            color = "🟢"
        elif confidence > 0.5:
            color = "🟡"
        else:
            color = "🔴"

        # Display activity
        st.markdown(
            f"{color} **{time_str}** - Agent `{agent}`: {action} "
            f"(confidence: {confidence:.2f})"
        )


def create_status_indicator(status: str, label: str = "Status"):
    """
    Create a visual status indicator.

    Args:
        status: Status value ('running', 'stopped', 'error', 'warning')
        label: Label for the indicator
    """
    status_config = {
        'running': {'icon': '🟢', 'color': 'green', 'text': 'Running'},
        'stopped': {'icon': '🔴', 'color': 'red', 'text': 'Stopped'},
        'error': {'icon': '❌', 'color': 'red', 'text': 'Error'},
        'warning': {'icon': '⚠️', 'color': 'orange', 'text': 'Warning'},
        'idle': {'icon': '💤', 'color': 'gray', 'text': 'Idle'}
    }

    config = status_config.get(status.lower(), status_config['stopped'])

    st.markdown(
        f"""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <span style='font-size: 24px;'>{config['icon']}</span>
            <div>
                <div style='font-weight: bold;'>{label}</div>
                <div style='color: {config['color']};'>{config['text']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def monitor_system_resources():
    """
    Monitor and display system resource usage.
    """
    try:
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        mem_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "CPU Usage",
                f"{cpu_percent}%",
                delta=None,
                help="Current CPU utilization"
            )
            if cpu_percent > 80:
                st.warning("High CPU usage detected")

        with col2:
            st.metric(
                "Memory Usage",
                f"{mem_percent}%",
                delta=None,
                help="Current RAM utilization"
            )
            if mem_percent > 80:
                st.warning("High memory usage detected")

        with col3:
            st.metric(
                "Disk Usage",
                f"{disk_percent}%",
                delta=None,
                help="Disk space utilization"
            )
            if disk_percent > 90:
                st.error("Low disk space!")

    except ImportError:
        st.info("Install 'psutil' for system resource monitoring")
    except Exception as e:
        st.error(f"Error monitoring system resources: {e}")


def create_alert_system(alerts: List[Dict[str, Any]]):
    """
    Display system alerts and notifications.

    Args:
        alerts: List of alert dictionaries
    """
    if not alerts:
        return

    # Group alerts by severity
    errors = [a for a in alerts if a.get('severity') == 'error']
    warnings = [a for a in alerts if a.get('severity') == 'warning']
    info = [a for a in alerts if a.get('severity') == 'info']

    # Display alerts
    if errors:
        with st.expander(f"❌ Errors ({len(errors)})", expanded=True):
            for alert in errors:
                st.error(alert.get('message', 'Unknown error'))

    if warnings:
        with st.expander(f"⚠️ Warnings ({len(warnings)})"):
            for alert in warnings:
                st.warning(alert.get('message', 'Unknown warning'))

    if info:
        with st.expander(f"ℹ️ Information ({len(info)})"):
            for alert in info:
                st.info(alert.get('message', 'Unknown information'))


def save_dashboard_state(state: Dict[str, Any]):
    """
    Save dashboard state for persistence.

    Args:
        state: Dashboard state dictionary
    """
    state_file = Path(".streamlit_dashboard_state.json")

    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Could not save dashboard state: {e}")


def load_dashboard_state() -> Dict[str, Any]:
    """
    Load saved dashboard state.

    Returns:
        Dashboard state dictionary
    """
    state_file = Path(".streamlit_dashboard_state.json")

    if not state_file.exists():
        return {}

    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load dashboard state: {e}")
        return {}