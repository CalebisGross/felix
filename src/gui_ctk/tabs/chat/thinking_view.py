"""
Thinking View Component for Felix Chat Interface

A collapsible panel that shows real-time agent activity during workflow mode.
Displays agent thinking steps with color-coded indicators and progress tracking.

Features:
- Collapsible CTkFrame with expand/collapse toggle
- Real-time updates via streaming callbacks
- Agent icons/names with progress indicators
- Color-coded by agent type (research, analysis, synthesis, critic)
- Theme-aware styling
- Smooth expand/collapse animations
"""

import customtkinter as ctk
from typing import Optional, Dict, List, Callable
import logging

from ...theme_manager import get_theme_manager
from ...styles import (
    FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD,
    RADIUS_MD, RADIUS_LG
)

logger = logging.getLogger(__name__)


# Agent type color mapping
AGENT_COLORS = {
    "research": "#3B82F6",      # Blue - exploration and data gathering
    "analysis": "#8B5CF6",      # Purple - deep analysis and reasoning
    "synthesis": "#10B981",     # Green - combining and creating solutions
    "critic": "#F59E0B",        # Amber - evaluation and quality control
}


class ThinkingView(ctk.CTkFrame):
    """
    Collapsible panel showing agent thinking steps during workflow execution.

    This component displays real-time agent activity with color-coded indicators,
    progress tracking, and smooth expand/collapse behavior. Perfect for showing
    users what's happening "under the hood" during complex multi-agent workflows.

    Agent Color Coding:
        - Research (Blue): Data gathering and exploration
        - Analysis (Purple): Deep reasoning and pattern detection
        - Synthesis (Green): Solution creation and integration
        - Critic (Amber): Quality evaluation and refinement

    Usage:
        thinking_view = ThinkingView(parent, on_toggle=callback)
        thinking_view.pack(fill="x", padx=10, pady=5)

        # Add agent steps
        thinking_view.add_agent_step("research", "Analyzing user query...", progress=0.3)
        thinking_view.add_agent_step("analysis", "Processing context...")

        # Update existing steps
        thinking_view.update_agent_step("research", "Search complete", progress=1.0)

        # Clear all steps
        thinking_view.clear()
    """

    def __init__(
        self,
        master,
        on_toggle: Optional[Callable[[bool], None]] = None,
        **kwargs
    ):
        """
        Initialize ThinkingView component.

        Args:
            master: Parent widget
            on_toggle: Optional callback when expand/collapse is toggled.
                       Receives boolean (True=expanded, False=collapsed)
            **kwargs: Additional arguments passed to CTkFrame
        """
        self.theme_manager = get_theme_manager()
        self._on_toggle_callback = on_toggle
        self._expanded = False
        self._agent_steps: Dict[str, Dict] = {}  # Track active agent steps

        # Configure frame appearance
        kwargs.setdefault("corner_radius", RADIUS_LG)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("fg_color", self.theme_manager.get_color("bg_secondary"))
        kwargs.setdefault("border_color", self.theme_manager.get_color("border"))

        super().__init__(master, **kwargs)

        self._setup_ui()

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def _setup_ui(self):
        """Setup the UI components."""
        self.grid_columnconfigure(0, weight=1)

        # Header with toggle button
        self._header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_SM)
        self._header_frame.grid_columnconfigure(1, weight=1)

        # Toggle button (chevron icon)
        self._toggle_btn = ctk.CTkButton(
            self._header_frame,
            text="▶",
            width=24,
            height=24,
            font=ctk.CTkFont(size=FONT_SMALL),
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover"),
            text_color=self.theme_manager.get_color("fg_secondary"),
            command=self._toggle_expanded
        )
        self._toggle_btn.grid(row=0, column=0, sticky="w")

        # Title label
        self._title_label = ctk.CTkLabel(
            self._header_frame,
            text="Agent Activity",
            font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
            text_color=self.theme_manager.get_color("fg_primary"),
            anchor="w"
        )
        self._title_label.grid(row=0, column=1, sticky="w", padx=(SPACE_XS, 0))

        # Status badge (shows number of active agents)
        self._badge_label = ctk.CTkLabel(
            self._header_frame,
            text="0",
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color="#FFFFFF",
            fg_color=self.theme_manager.get_color("fg_muted"),
            corner_radius=RADIUS_MD,
            width=24,
            height=20
        )
        self._badge_label.grid(row=0, column=2, sticky="e", padx=(SPACE_SM, 0))

        # Content frame (hidden by default)
        self._content_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=self.theme_manager.get_color("bg_tertiary"),
            corner_radius=RADIUS_MD,
            height=200
        )
        self._content_frame.grid_columnconfigure(0, weight=1)
        # Don't grid content frame yet - it's collapsed by default

    def _toggle_expanded(self):
        """Toggle between expanded and collapsed states."""
        self._expanded = not self._expanded
        self.set_expanded(self._expanded)

        # Notify callback
        if self._on_toggle_callback:
            try:
                self._on_toggle_callback(self._expanded)
            except Exception as e:
                logger.error(f"Error in toggle callback: {e}")

    def set_expanded(self, expanded: bool):
        """
        Set the expanded state.

        Args:
            expanded: True to expand, False to collapse
        """
        self._expanded = expanded

        if expanded:
            # Show content and update toggle icon
            self._content_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
            self._toggle_btn.configure(text="▼")
        else:
            # Hide content and update toggle icon
            self._content_frame.grid_remove()
            self._toggle_btn.configure(text="▶")

    def add_agent_step(
        self,
        agent_type: str,
        content: str,
        progress: Optional[float] = None
    ):
        """
        Add a new agent thinking step or update existing one.

        Args:
            agent_type: Type of agent ('research', 'analysis', 'synthesis', 'critic')
            content: Description of what the agent is doing
            progress: Optional progress value (0.0 to 1.0)
        """
        # Get agent color
        agent_color = AGENT_COLORS.get(agent_type.lower(), self.theme_manager.get_color("fg_secondary"))

        # Check if this agent already has a step
        if agent_type in self._agent_steps:
            # Update existing step
            step_info = self._agent_steps[agent_type]
            step_info["content_label"].configure(text=content)

            if progress is not None and step_info.get("progress_bar"):
                step_info["progress_bar"].set(progress)
        else:
            # Create new step container
            step_frame = ctk.CTkFrame(self._content_frame, fg_color="transparent")
            step_frame.grid(
                row=len(self._agent_steps),
                column=0,
                sticky="ew",
                padx=SPACE_XS,
                pady=(0, SPACE_SM)
            )
            step_frame.grid_columnconfigure(1, weight=1)

            # Agent type indicator (colored dot)
            indicator = ctk.CTkFrame(
                step_frame,
                width=8,
                height=8,
                fg_color=agent_color,
                corner_radius=4
            )
            indicator.grid(row=0, column=0, sticky="w", padx=(0, SPACE_XS))

            # Agent name and content container
            text_container = ctk.CTkFrame(step_frame, fg_color="transparent")
            text_container.grid(row=0, column=1, sticky="ew")
            text_container.grid_columnconfigure(0, weight=1)

            # Agent name label
            name_label = ctk.CTkLabel(
                text_container,
                text=f"{agent_type.title()}:",
                font=ctk.CTkFont(size=FONT_SMALL, weight="bold"),
                text_color=agent_color,
                anchor="w"
            )
            name_label.grid(row=0, column=0, sticky="w")

            # Content label
            content_label = ctk.CTkLabel(
                text_container,
                text=content,
                font=ctk.CTkFont(size=FONT_SMALL),
                text_color=self.theme_manager.get_color("fg_primary"),
                anchor="w",
                justify="left",
                wraplength=400
            )
            content_label.grid(row=1, column=0, sticky="ew", pady=(SPACE_XS, 0))

            # Progress bar (if progress provided)
            progress_bar = None
            if progress is not None:
                progress_bar = ctk.CTkProgressBar(
                    text_container,
                    width=200,
                    height=4,
                    corner_radius=2,
                    progress_color=agent_color,
                    fg_color=self.theme_manager.get_color("bg_primary")
                )
                progress_bar.set(progress)
                progress_bar.grid(row=2, column=0, sticky="ew", pady=(SPACE_XS, 0))

            # Store step info for updates
            self._agent_steps[agent_type] = {
                "frame": step_frame,
                "indicator": indicator,
                "name_label": name_label,
                "content_label": content_label,
                "progress_bar": progress_bar,
                "color": agent_color
            }

        # Update badge count
        self._update_badge()

        # Auto-expand if not already expanded and there are steps
        if not self._expanded and self._agent_steps:
            self.set_expanded(True)

    def update_agent_step(
        self,
        agent_type: str,
        content: str,
        progress: Optional[float] = None
    ):
        """
        Update an existing agent step.

        Args:
            agent_type: Type of agent to update
            content: New content text
            progress: Optional new progress value (0.0 to 1.0)
        """
        if agent_type not in self._agent_steps:
            logger.warning(f"Attempted to update non-existent agent step: {agent_type}")
            return

        step_info = self._agent_steps[agent_type]
        step_info["content_label"].configure(text=content)

        if progress is not None and step_info.get("progress_bar"):
            step_info["progress_bar"].set(progress)

    def remove_agent_step(self, agent_type: str):
        """
        Remove an agent step.

        Args:
            agent_type: Type of agent to remove
        """
        if agent_type not in self._agent_steps:
            return

        # Destroy the step frame
        step_info = self._agent_steps[agent_type]
        step_info["frame"].destroy()

        # Remove from tracking
        del self._agent_steps[agent_type]

        # Update badge
        self._update_badge()

        # Reposition remaining steps
        for i, (agent, info) in enumerate(self._agent_steps.items()):
            info["frame"].grid(row=i, column=0, sticky="ew", padx=SPACE_XS, pady=(0, SPACE_SM))

        # Auto-collapse if no steps remain
        if not self._agent_steps and self._expanded:
            self.set_expanded(False)

    def clear(self):
        """Clear all agent steps."""
        # Destroy all step frames
        for agent_type in list(self._agent_steps.keys()):
            self.remove_agent_step(agent_type)

        # Collapse if expanded
        if self._expanded:
            self.set_expanded(False)

    def _update_badge(self):
        """Update the badge showing number of active agents."""
        count = len(self._agent_steps)
        self._badge_label.configure(text=str(count))

        # Change badge color based on count
        if count == 0:
            badge_color = self.theme_manager.get_color("fg_muted")
        elif count <= 2:
            badge_color = self.theme_manager.get_color("success")
        else:
            badge_color = self.theme_manager.get_color("accent")

        self._badge_label.configure(fg_color=badge_color)

    def _on_theme_change(self, mode: str):
        """
        Handle theme change and update all colors.

        Args:
            mode: New theme mode ('dark' or 'light')
        """
        # Update frame colors
        self.configure(
            fg_color=self.theme_manager.get_color("bg_secondary"),
            border_color=self.theme_manager.get_color("border")
        )

        # Update header components
        self._toggle_btn.configure(
            hover_color=self.theme_manager.get_color("bg_hover"),
            text_color=self.theme_manager.get_color("fg_secondary")
        )
        self._title_label.configure(text_color=self.theme_manager.get_color("fg_primary"))

        # Update content frame
        self._content_frame.configure(fg_color=self.theme_manager.get_color("bg_tertiary"))

        # Update badge
        self._update_badge()

        # Update all agent steps
        for agent_type, step_info in self._agent_steps.items():
            step_info["content_label"].configure(
                text_color=self.theme_manager.get_color("fg_primary")
            )

            if step_info.get("progress_bar"):
                step_info["progress_bar"].configure(
                    fg_color=self.theme_manager.get_color("bg_primary")
                )

    def get_expanded(self) -> bool:
        """
        Get current expanded state.

        Returns:
            True if expanded, False if collapsed
        """
        return self._expanded

    def get_agent_count(self) -> int:
        """
        Get the number of active agent steps.

        Returns:
            Number of agent steps currently displayed
        """
        return len(self._agent_steps)

    def has_agent_step(self, agent_type: str) -> bool:
        """
        Check if a specific agent has an active step.

        Args:
            agent_type: Agent type to check

        Returns:
            True if agent has an active step, False otherwise
        """
        return agent_type in self._agent_steps

    def destroy(self):
        """Clean up resources when destroyed."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass  # Ignore errors during cleanup
        super().destroy()


class CompactThinkingView(ctk.CTkFrame):
    """
    Compact variant of ThinkingView for smaller spaces.

    Shows only the most recent agent activity in a single line with an animated indicator.
    Useful for inline display in chat messages or tight layouts.

    Usage:
        compact_view = CompactThinkingView(parent)
        compact_view.pack(fill="x", padx=5, pady=2)
        compact_view.set_agent("research", "Analyzing query...")
    """

    def __init__(self, master, **kwargs):
        """
        Initialize CompactThinkingView.

        Args:
            master: Parent widget
            **kwargs: Additional arguments passed to CTkFrame
        """
        self.theme_manager = get_theme_manager()

        kwargs.setdefault("corner_radius", RADIUS_MD)
        kwargs.setdefault("fg_color", self.theme_manager.get_color("bg_tertiary"))
        kwargs.setdefault("height", 32)

        super().__init__(master, **kwargs)

        self._current_agent: Optional[str] = None
        self._animation_state = 0

        self._setup_ui()

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def _setup_ui(self):
        """Setup the UI components."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_propagate(False)

        # Animated indicator
        self._indicator = ctk.CTkLabel(
            self,
            text="●",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color=self.theme_manager.get_color("fg_muted"),
            width=16
        )
        self._indicator.grid(row=0, column=0, padx=(SPACE_SM, SPACE_XS))

        # Content label
        self._content_label = ctk.CTkLabel(
            self,
            text="Waiting for agent activity...",
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color=self.theme_manager.get_color("fg_secondary"),
            anchor="w"
        )
        self._content_label.grid(row=0, column=1, sticky="ew", padx=(0, SPACE_SM))

    def set_agent(self, agent_type: str, content: str):
        """
        Set the current agent and content.

        Args:
            agent_type: Type of agent ('research', 'analysis', 'synthesis', 'critic')
            content: What the agent is doing
        """
        self._current_agent = agent_type
        agent_color = AGENT_COLORS.get(agent_type.lower(), self.theme_manager.get_color("fg_secondary"))

        self._indicator.configure(text_color=agent_color)
        self._content_label.configure(
            text=f"{agent_type.title()}: {content}",
            text_color=self.theme_manager.get_color("fg_primary")
        )

    def clear(self):
        """Clear the current agent activity."""
        self._current_agent = None
        self._indicator.configure(text_color=self.theme_manager.get_color("fg_muted"))
        self._content_label.configure(
            text="Waiting for agent activity...",
            text_color=self.theme_manager.get_color("fg_secondary")
        )

    def _on_theme_change(self, mode: str):
        """Handle theme change."""
        self.configure(fg_color=self.theme_manager.get_color("bg_tertiary"))

        if self._current_agent:
            agent_color = AGENT_COLORS.get(
                self._current_agent.lower(),
                self.theme_manager.get_color("fg_secondary")
            )
            self._indicator.configure(text_color=agent_color)
            self._content_label.configure(text_color=self.theme_manager.get_color("fg_primary"))
        else:
            self._indicator.configure(text_color=self.theme_manager.get_color("fg_muted"))
            self._content_label.configure(text_color=self.theme_manager.get_color("fg_secondary"))

    def destroy(self):
        """Clean up resources."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass
        super().destroy()
