"""
Live Terminal Component for System Control

Streams real-time command execution output with:
- Active command monitoring (100ms polling)
- Real-time stdout/stderr capture
- Command status indicators
- Fast-failing command detection (<100ms)

Based on tkinter GUI implementation (src/gui/terminal.py)
"""

import streamlit as st
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path


class LiveTerminal:
    """Live terminal output streaming component."""

    def __init__(self, db_path: str = "felix_system_actions.db"):
        """
        Initialize the live terminal.

        Args:
            db_path: Path to system actions database
        """
        self.db_path = Path(db_path)
        self.polling_interval = 0.1  # 100ms polling

    def get_active_commands(self) -> List[Dict[str, Any]]:
        """
        Get currently executing commands.

        Returns:
            List of active command dicts with execution details
        """
        if not self.db_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Query for commands that are approved but not yet completed
            # or completed very recently (last 30 seconds for fast failures)
            now = time.time()
            recent_threshold = now - 30  # Last 30 seconds

            query = """
            SELECT
                execution_id,
                command,
                agent_id,
                workflow_id,
                trust_level,
                executed,
                success,
                exit_code,
                duration,
                stdout_preview,
                stderr_preview,
                timestamp,
                approved_at,
                approved_by,
                cwd
            FROM command_executions
            WHERE (executed = 0 AND approved_at IS NOT NULL)
               OR (executed = 1 AND timestamp >= ?)
            ORDER BY timestamp DESC
            LIMIT 20
            """

            cursor.execute(query, (recent_threshold,))
            rows = cursor.fetchall()

            commands = []
            for row in rows:
                command_data = {
                    "execution_id": row[0],
                    "command": row[1],
                    "agent_id": row[2],
                    "workflow_id": row[3],
                    "trust_level": row[4],
                    "executed": bool(row[5]),
                    "success": bool(row[6]) if row[6] is not None else None,
                    "exit_code": row[7],
                    "duration": row[8],
                    "stdout_preview": row[9] or "",
                    "stderr_preview": row[10] or "",
                    "timestamp": row[11],
                    "approved_at": row[12],
                    "approved_by": row[13],
                    "cwd": row[14]
                }

                # Determine status
                if not command_data["executed"]:
                    command_data["status"] = "running"
                    command_data["status_icon"] = "⏳"
                    command_data["status_color"] = "blue"
                elif command_data["success"]:
                    command_data["status"] = "completed"
                    command_data["status_icon"] = "✅"
                    command_data["status_color"] = "green"
                else:
                    command_data["status"] = "failed"
                    command_data["status_icon"] = "❌"
                    command_data["status_color"] = "red"

                commands.append(command_data)

            conn.close()
            return commands

        except Exception as e:
            st.error(f"Error fetching active commands: {e}")
            return []

    def get_command_output(self, execution_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full output for a specific command.

        Args:
            execution_id: The execution ID

        Returns:
            Dict with full stdout/stderr or None
        """
        if not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = """
            SELECT
                stdout_preview,
                stderr_preview,
                executed,
                success,
                exit_code,
                duration,
                timestamp
            FROM command_executions
            WHERE execution_id = ?
            """

            cursor.execute(query, (execution_id,))
            row = cursor.fetchone()

            if row:
                output_data = {
                    "stdout": row[0] or "",
                    "stderr": row[1] or "",
                    "executed": bool(row[2]),
                    "success": bool(row[3]) if row[3] is not None else None,
                    "exit_code": row[4],
                    "duration": row[5],
                    "timestamp": row[6]
                }
                conn.close()
                return output_data

            conn.close()
            return None

        except Exception as e:
            st.error(f"Error fetching command output: {e}")
            return None

    def render_active_commands_panel(self) -> None:
        """Render the active commands panel with real-time updates."""
        st.markdown("### ⚡ Active Commands")

        # Get active commands
        active_commands = self.get_active_commands()

        if not active_commands:
            st.info("✅ No active commands. All executions completed.")
            return

        # Display active commands
        for cmd in active_commands:
            # Only show truly active (not executed) or very recent completions
            if not cmd["executed"] or (time.time() - cmd["timestamp"]) < 5:
                status_icon = cmd["status_icon"]
                status_label = cmd["status"].title()
                command_preview = cmd["command"][:80] + "..." if len(cmd["command"]) > 80 else cmd["command"]

                with st.expander(
                    f"{status_icon} **{status_label}** - `{command_preview}`",
                    expanded=(not cmd["executed"])  # Expand only running commands
                ):
                    # Command metadata
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"**Execution ID:** {cmd['execution_id']}")
                        st.markdown(f"**Agent:** {cmd['agent_id']}")

                    with col2:
                        st.markdown(f"**Trust Level:** {cmd['trust_level']}")
                        if cmd['duration']:
                            st.markdown(f"**Duration:** {cmd['duration']:.2f}s")

                    with col3:
                        if cmd['executed']:
                            st.markdown(f"**Exit Code:** {cmd.get('exit_code', 'N/A')}")
                            st.markdown(f"**Success:** {'Yes' if cmd['success'] else 'No'}")

                    # Full command
                    st.markdown("**Full Command:**")
                    st.code(cmd["command"], language='bash')

                    # Working directory
                    if cmd.get('cwd'):
                        st.markdown(f"**Working Directory:** `{cmd['cwd']}`")

                    # Output section
                    if cmd["stdout_preview"] or cmd["stderr_preview"]:
                        st.markdown("---")

                        if cmd["stdout_preview"]:
                            st.markdown("**📤 Standard Output:**")
                            # Terminal-style output (green on dark)
                            st.markdown(
                                f"""
                                <div style="background-color: #1e1e1e; color: #00ff00;
                                     padding: 10px; border-radius: 5px; font-family: monospace;
                                     max-height: 300px; overflow-y: auto;">
                                    <pre style="margin: 0;">{cmd["stdout_preview"]}</pre>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if cmd["stderr_preview"]:
                            st.markdown("**📥 Standard Error:**")
                            # Red output for errors
                            st.markdown(
                                f"""
                                <div style="background-color: #1e1e1e; color: #ff0000;
                                     padding: 10px; border-radius: 5px; font-family: monospace;
                                     max-height: 300px; overflow-y: auto;">
                                    <pre style="margin: 0;">{cmd["stderr_preview"]}</pre>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

    def render_terminal_viewer(
        self,
        auto_refresh: bool = True,
        refresh_interval: float = 2.0
    ) -> None:
        """
        Render the complete terminal viewer with auto-refresh.

        Args:
            auto_refresh: Enable automatic refresh
            refresh_interval: Seconds between refreshes
        """
        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("**Live Terminal Output Viewer**")

        with col2:
            auto_refresh_enabled = st.checkbox(
                "Auto-refresh",
                value=auto_refresh,
                key="terminal_auto_refresh"
            )

        with col3:
            if st.button("🔄 Refresh Now", key="terminal_refresh_now"):
                st.rerun()

        st.markdown("---")

        # Active commands panel
        self.render_active_commands_panel()

        # Auto-refresh logic
        if auto_refresh_enabled:
            # Show active commands and check if any are still running
            active_commands = self.get_active_commands()
            has_running = any(not cmd["executed"] for cmd in active_commands)

            if has_running:
                st.caption(f"Auto-refreshing every {refresh_interval}s while commands are running...")
                time.sleep(refresh_interval)
                st.rerun()


def render_live_terminal(
    db_path: str = "felix_system_actions.db",
    auto_refresh: bool = True,
    refresh_interval: float = 2.0
) -> None:
    """
    Convenience function to render live terminal.

    Args:
        db_path: Path to system actions database
        auto_refresh: Enable automatic refresh
        refresh_interval: Seconds between refreshes
    """
    terminal = LiveTerminal(db_path)
    terminal.render_terminal_viewer(auto_refresh, refresh_interval)


# Example usage
if __name__ == "__main__":
    st.set_page_config(page_title="Live Terminal Demo", layout="wide")

    st.title("Live Terminal Component Demo")

    st.markdown("""
    This component shows real-time command execution output from the Felix system.
    Commands are polled every 2 seconds and display live stdout/stderr.
    """)

    render_live_terminal()