"""
Enhanced Progress Bar Component with Animation

Features:
- Standard mode: Filled progress with accent color
- Indeterminate mode: Animated back-and-forth movement
- Color variants: primary (Helix Blue), success (Elegant Teal), warning (amber)
- Size variants: thin (4px), standard (8px), thick (12px)
- Smooth corners and colors
"""

import customtkinter as ctk
from typing import Literal, Optional
from ..theme_manager import ThemeManager, get_theme_manager
from ..styles import RADIUS_SM


class EnhancedProgressBar(ctk.CTkProgressBar):
    """
    Enhanced progress bar with animation and color variants.

    Args:
        parent: Parent widget
        mode: "determinate" or "indeterminate"
        variant: "primary", "success", "warning"
        size: "thin", "standard", "thick"
        width: Progress bar width in pixels
        **kwargs: Additional CTkProgressBar arguments
    """

    def __init__(
        self,
        parent,
        mode: Literal["determinate", "indeterminate"] = "determinate",
        variant: Literal["primary", "success", "warning"] = "primary",
        size: Literal["thin", "standard", "thick"] = "standard",
        width: int = 300,
        **kwargs
    ):
        self.theme_manager = get_theme_manager()
        self.mode = mode
        self.variant = variant
        self.size = size

        # Get height based on size
        height = self._get_height()

        # Get progress color based on variant
        progress_color = self._get_progress_color()

        # Initialize base progress bar
        super().__init__(
            parent,
            width=width,
            height=height,
            corner_radius=RADIUS_SM,
            progress_color=progress_color,
            fg_color=self.theme_manager.colors["bg_tertiary"],
            mode=mode,
            **kwargs
        )

        # Animation state for indeterminate mode
        self._animation_value = 0.0
        self._animation_direction = 1  # 1 = forward, -1 = backward
        self._animation_job = None

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

        # Start animation if indeterminate
        if mode == "indeterminate":
            self.start_animation()

    def _get_height(self) -> int:
        """Get height based on size variant."""
        sizes = {
            "thin": 4,
            "standard": 8,
            "thick": 12,
        }
        return sizes.get(self.size, 8)

    def _get_progress_color(self) -> str:
        """Get progress color based on variant."""
        colors = self.theme_manager.colors

        if self.variant == "primary":
            return colors["accent"]  # Helix Blue
        elif self.variant == "success":
            return colors["success"]  # Elegant Teal
        elif self.variant == "warning":
            return colors["warning"]  # Amber
        else:
            return colors["accent"]

    def _update_appearance(self):
        """Update progress bar appearance."""
        progress_color = self._get_progress_color()
        colors = self.theme_manager.colors

        self.configure(
            progress_color=progress_color,
            fg_color=colors["bg_tertiary"]
        )

    def _animate_indeterminate(self):
        """Animate progress bar back and forth for indeterminate mode."""
        if not self.winfo_exists():
            return

        # Update animation value
        self._animation_value += 0.02 * self._animation_direction

        # Reverse direction at bounds
        if self._animation_value >= 1.0:
            self._animation_value = 1.0
            self._animation_direction = -1
        elif self._animation_value <= 0.0:
            self._animation_value = 0.0
            self._animation_direction = 1

        # Update progress bar
        self.set(self._animation_value)

        # Schedule next frame (50ms = 20 FPS)
        self._animation_job = self.after(50, self._animate_indeterminate)

    def start_animation(self):
        """Start indeterminate animation."""
        if self.mode == "indeterminate" and self._animation_job is None:
            self._animate_indeterminate()

    def stop_animation(self):
        """Stop indeterminate animation."""
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None

    def set_progress(self, value: float):
        """Set progress value (0.0 - 1.0) for determinate mode."""
        if self.mode == "determinate":
            self.set(max(0.0, min(1.0, value)))

    def set_variant(self, variant: Literal["primary", "success", "warning"]):
        """Change progress bar color variant."""
        self.variant = variant
        self._update_appearance()

    def set_mode(self, mode: Literal["determinate", "indeterminate"]):
        """Change progress bar mode."""
        self.mode = mode

        if mode == "indeterminate":
            self.start_animation()
        else:
            self.stop_animation()

    def _on_theme_change(self, mode: str):
        """Handle theme change."""
        self._update_appearance()

    def destroy(self):
        """Clean up on destruction."""
        self.stop_animation()
        self.theme_manager.unregister_callback(self._on_theme_change)
        super().destroy()
