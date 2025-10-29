"""
Detailed Execution Viewer Component

Provides comprehensive view of command execution details:
- Execution metadata (ID, agent, workflow, timestamps)
- Command and context
- Execution results (exit code, duration, success)
- Full stdout/stderr output
- Environment info (cwd, venv status, approval details)

Based on tkinter GUI implementation (src/gui/terminal.py - CommandDetailsDialog)
"""

import streamlit as st
from typing import Dict, Optional, Any
from datetime import datetime
import sqlite3
from pathlib import Path


class ExecutionViewer:
    """Detailed execution viewer for command history."""

    def __init__(self, db_path: str = "felix_system_actions.db"):
        """
        Initialize the execution viewer.

        Args:
            db_path: Path to system actions database
        """
        self.db_path = Path(db_path)

    def get_execution_details(self, execution_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full details for a specific execution.

        Args:
            execution_id: The execution ID

        Returns:
            Dict with full execution details or None
        """
        if not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = """
            SELECT
                execution_id,
                command,
                agent_id,
                workflow_id,
                context,
                trust_level,
                risk_score,
                executed,
                success,
                exit_code,
                duration,
                stdout_preview,
                stderr_preview,
                error_category,
                timestamp,
                approved_at,
                approved_by,
                approval_id,
                cwd,
                venv_active
            FROM command_executions
            WHERE execution_id = ?
            """

            cursor.execute(query, (execution_id,))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return None

            details = {
                "execution_id": row[0],
                "command": row[1],
                "agent_id": row[2],
                "workflow_id": row[3],
                "context": row[4],
                "trust_level": row[5],
                "risk_score": row[6],
                "executed": bool(row[7]),
                "success": bool(row[8]) if row[8] is not None else None,
                "exit_code": row[9],
                "duration": row[10],
                "stdout_preview": row[11] or "",
                "stderr_preview": row[12] or "",
                "error_category": row[13],
                "timestamp": row[14],
                "approved_at": row[15],
                "approved_by": row[16],
                "approval_id": row[17],
                "cwd": row[18],
                "venv_active": bool(row[19]) if row[19] is not None else None
            }

            conn.close()
            return details

        except Exception as e:
            st.error(f"Error fetching execution details: {e}")
            return None

    def render_execution_details(
        self,
        execution_id: int,
        container_key: str = "execution_details"
    ) -> None:
        """
        Render detailed view of command execution.

        Args:
            execution_id: The execution ID to display
            container_key: Unique key for this viewer instance
        """
        details = self.get_execution_details(execution_id)

        if not details:
            st.error(f"Execution ID {execution_id} not found")
            return

        # Header with status
        status_icon = "✅" if details.get("success") else ("❌" if details.get("executed") else "⏳")
        status_text = "Success" if details.get("success") else ("Failed" if details.get("executed") else "Pending")

        st.markdown(f"## {status_icon} Execution Details - {status_text}")
        st.markdown(f"**Execution ID:** `{execution_id}`")

        st.markdown("---")

        # Metadata section
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📋 Metadata")
            st.markdown(f"**Agent:** {details.get('agent_id', 'N/A')}")
            st.markdown(f"**Workflow:** {details.get('workflow_id', 'N/A')}")

            # Trust level badge
            trust_level = details.get('trust_level', 'UNKNOWN')
            if trust_level == "SAFE":
                trust_badge = "🟢 SAFE"
            elif trust_level == "REVIEW":
                trust_badge = "🟡 REVIEW"
            else:
                trust_badge = "🔴 BLOCKED"
            st.markdown(f"**Trust Level:** {trust_badge}")

        with col2:
            st.markdown("### ⏱️ Timestamps")

            # Request timestamp
            if details.get('timestamp'):
                request_dt = datetime.fromtimestamp(details['timestamp'])
                st.markdown(f"**Requested:** {request_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            # Approval timestamp
            if details.get('approved_at'):
                approval_dt = datetime.fromtimestamp(details['approved_at'])
                st.markdown(f"**Approved:** {approval_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Approved By:** {details.get('approved_by', 'Unknown')}")

            # Duration
            if details.get('duration'):
                st.markdown(f"**Duration:** {details['duration']:.2f}s")

        with col3:
            st.markdown("### 📊 Results")

            if details.get('executed'):
                st.markdown(f"**Exit Code:** {details.get('exit_code', 'N/A')}")
                st.markdown(f"**Success:** {'✅ Yes' if details.get('success') else '❌ No'}")

                if details.get('error_category'):
                    st.markdown(f"**Error Category:** {details['error_category']}")
            else:
                st.markdown("*Execution pending or in progress*")

            # Risk score
            if details.get('risk_score') is not None:
                risk_score = details['risk_score']
                risk_pct = risk_score * 100
                st.markdown(f"**Risk Score:** {risk_pct:.0f}%")
                st.progress(risk_score)

        st.markdown("---")

        # Command section
        st.markdown("### 💻 Command")
        st.code(details.get('command', 'N/A'), language='bash')

        # Context
        if details.get('context'):
            st.markdown("**Context:**")
            st.info(details['context'])

        # Environment info
        st.markdown("### 🌍 Environment")
        env_col1, env_col2 = st.columns(2)

        with env_col1:
            if details.get('cwd'):
                st.markdown(f"**Working Directory:** `{details['cwd']}`")

        with env_col2:
            venv_status = details.get('venv_active')
            if venv_status is not None:
                venv_icon = "✅" if venv_status else "❌"
                st.markdown(f"**Virtual Environment:** {venv_icon} {'Active' if venv_status else 'Inactive'}")

        st.markdown("---")

        # Output section
        st.markdown("### 📤 Output")

        if details.get('executed'):
            # Tabbed output view
            tab1, tab2 = st.tabs(["Standard Output", "Standard Error"])

            with tab1:
                stdout = details.get('stdout_preview', '')
                if stdout:
                    st.markdown("**stdout:**")
                    # Terminal-style green text
                    st.markdown(
                        f"""
                        <div style="background-color: #1e1e1e; color: #00ff00;
                             padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace;
                             max-height: 500px; overflow-y: auto; white-space: pre-wrap;">
                            {stdout}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Copy button (would need clipboard support)
                    if st.button("📋 Copy stdout", key=f"{container_key}_copy_stdout"):
                        st.toast("Output copied to clipboard!")
                else:
                    st.info("No standard output")

            with tab2:
                stderr = details.get('stderr_preview', '')
                if stderr:
                    st.markdown("**stderr:**")
                    # Terminal-style red text
                    st.markdown(
                        f"""
                        <div style="background-color: #1e1e1e; color: #ff0000;
                             padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace;
                             max-height: 500px; overflow-y: auto; white-space: pre-wrap;">
                            {stderr}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if st.button("📋 Copy stderr", key=f"{container_key}_copy_stderr"):
                        st.toast("Error output copied to clipboard!")
                else:
                    st.info("No error output")
        else:
            st.info("⏳ Command execution pending or in progress. Output will appear here when available.")

        # Approval details
        if details.get('approval_id'):
            with st.expander("🔐 Approval Details"):
                st.markdown(f"**Approval ID:** `{details['approval_id']}`")
                if details.get('approved_by'):
                    st.markdown(f"**Approved By:** {details['approved_by']}")
                if details.get('approved_at'):
                    approval_dt = datetime.fromtimestamp(details['approved_at'])
                    st.markdown(f"**Approval Time:** {approval_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    def render_execution_selector(
        self,
        on_select: Optional[callable] = None,
        limit: int = 50
    ) -> Optional[int]:
        """
        Render execution selector dropdown.

        Args:
            on_select: Callback when execution is selected
            limit: Maximum number of executions to show

        Returns:
            Selected execution ID or None
        """
        if not self.db_path.exists():
            st.warning("System actions database not found")
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = """
            SELECT
                execution_id,
                command,
                timestamp,
                success,
                executed
            FROM command_executions
            ORDER BY timestamp DESC
            LIMIT ?
            """

            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                st.info("No executions found")
                return None

            # Create selection options
            options = {}
            for row in rows:
                exec_id, command, timestamp, success, executed = row

                # Format display
                dt = datetime.fromtimestamp(timestamp)
                time_str = dt.strftime('%Y-%m-%d %H:%M')

                # Status icon
                if not executed:
                    status = "⏳"
                elif success:
                    status = "✅"
                else:
                    status = "❌"

                # Truncate command
                cmd_preview = command[:60] + "..." if len(command) > 60 else command

                label = f"{status} [{exec_id}] {cmd_preview} - {time_str}"
                options[label] = exec_id

            # Render selector
            selected_label = st.selectbox(
                "Select Execution:",
                list(options.keys()),
                key="execution_selector"
            )

            selected_id = options[selected_label]

            if on_select:
                on_select(selected_id)

            return selected_id

        except Exception as e:
            st.error(f"Error loading executions: {e}")
            return None


def render_execution_viewer(
    execution_id: Optional[int] = None,
    db_path: str = "felix_system_actions.db",
    show_selector: bool = True
) -> None:
    """
    Convenience function to render execution viewer.

    Args:
        execution_id: Specific execution to view (optional)
        db_path: Path to system actions database
        show_selector: Show execution selector dropdown
    """
    viewer = ExecutionViewer(db_path)

    if show_selector:
        selected_id = viewer.render_execution_selector()
        if selected_id:
            st.markdown("---")
            viewer.render_execution_details(selected_id)
    elif execution_id:
        viewer.render_execution_details(execution_id)
    else:
        st.warning("No execution ID provided")


# Example usage
if __name__ == "__main__":
    st.set_page_config(page_title="Execution Viewer Demo", layout="wide")

    st.title("Execution Viewer Component Demo")

    st.markdown("""
    This component provides detailed view of command execution history.
    Select an execution from the dropdown to see full details.
    """)

    render_execution_viewer()