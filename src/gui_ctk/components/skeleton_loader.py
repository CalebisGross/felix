"""
Skeleton Loader Component with Shimmer Animation

Features:
- Rectangular placeholder with animated gradient effect
- Smooth shimmer animation using after() loop
- Configurable dimensions and corner radius
- Theme-aware colors
- Automatically stops animation on destruction
"""

import customtkinter as ctk
from typing import Optional
from ..theme_manager import ThemeManager, get_theme_manager
from ..styles import RADIUS_MD


class SkeletonLoader(ctk.CTkFrame):
    """
    Loading placeholder with shimmer animation.

    Creates a gray rectangle that cycles through shades to indicate loading.
    Use in place of content while data is being fetched.

    Args:
        parent: Parent widget
        width: Skeleton width in pixels
        height: Skeleton height in pixels
        corner_radius: Corner radius (default: RADIUS_MD)
        **kwargs: Additional CTkFrame arguments
    """

    def __init__(
        self,
        parent,
        width: int = 200,
        height: int = 40,
        corner_radius: int = RADIUS_MD,
        **kwargs
    ):
        self.theme_manager = get_theme_manager()
        self._corner_radius = corner_radius

        # Animation state
        self._animation_step = 0
        self._animation_job = None
        self._is_animating = True

        # Color shades for shimmer effect
        self._color_steps = self._generate_color_gradient()

        # Initialize frame with initial color
        super().__init__(
            parent,
            width=width,
            height=height,
            corner_radius=corner_radius,
            fg_color=self._color_steps[0],
            **kwargs
        )

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

        # Start shimmer animation
        self.start_animation()

    def _generate_color_gradient(self) -> list:
        """Generate color gradient for shimmer effect."""
        colors = self.theme_manager.colors
        base_color = colors["bg_secondary"]
        light_color = colors["bg_hover"]

        # Parse hex colors
        base_rgb = self._hex_to_rgb(base_color)
        light_rgb = self._hex_to_rgb(light_color)

        # Generate 10 steps from base -> light -> base
        steps = []
        num_steps = 10

        # Forward gradient (base -> light)
        for i in range(num_steps // 2):
            factor = i / (num_steps // 2)
            r = int(base_rgb[0] + (light_rgb[0] - base_rgb[0]) * factor)
            g = int(base_rgb[1] + (light_rgb[1] - base_rgb[1]) * factor)
            b = int(base_rgb[2] + (light_rgb[2] - base_rgb[2]) * factor)
            steps.append(f"#{r:02x}{g:02x}{b:02x}")

        # Backward gradient (light -> base)
        for i in range(num_steps // 2):
            factor = i / (num_steps // 2)
            r = int(light_rgb[0] + (base_rgb[0] - light_rgb[0]) * factor)
            g = int(light_rgb[1] + (base_rgb[1] - light_rgb[1]) * factor)
            b = int(light_rgb[2] + (base_rgb[2] - light_rgb[2]) * factor)
            steps.append(f"#{r:02x}{g:02x}{b:02x}")

        return steps

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _animate_shimmer(self):
        """Animate shimmer effect by cycling through color gradient."""
        if not self.winfo_exists() or not self._is_animating:
            return

        # Update color
        current_color = self._color_steps[self._animation_step]
        self.configure(fg_color=current_color)

        # Advance to next step (loop back to 0)
        self._animation_step = (self._animation_step + 1) % len(self._color_steps)

        # Schedule next frame (100ms = 10 FPS for subtle shimmer)
        self._animation_job = self.after(100, self._animate_shimmer)

    def start_animation(self):
        """Start shimmer animation."""
        if self._animation_job is None and self._is_animating:
            self._animate_shimmer()

    def stop_animation(self):
        """Stop shimmer animation."""
        self._is_animating = False
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None

    def _on_theme_change(self, mode: str):
        """Handle theme change - regenerate color gradient."""
        self._color_steps = self._generate_color_gradient()
        # Reset animation to use new colors
        self._animation_step = 0

    def destroy(self):
        """Clean up on destruction."""
        self.stop_animation()
        self.theme_manager.unregister_callback(self._on_theme_change)
        super().destroy()


class SkeletonText(SkeletonLoader):
    """Skeleton loader specifically for text lines."""

    def __init__(self, parent, width: int = 200, **kwargs):
        super().__init__(parent, width=width, height=16, corner_radius=4, **kwargs)


class SkeletonCard(ctk.CTkFrame):
    """Skeleton loader for card-like content with multiple elements."""

    def __init__(self, parent, width: int = 300, **kwargs):
        self.theme_manager = get_theme_manager()

        super().__init__(
            parent,
            fg_color=self.theme_manager.colors["bg_secondary"],
            corner_radius=RADIUS_MD,
            width=width,
            **kwargs
        )

        # Title skeleton
        SkeletonLoader(self, width=width - 40, height=20).pack(padx=20, pady=(20, 10))

        # Description skeletons
        SkeletonText(self, width=width - 40).pack(padx=20, pady=5)
        SkeletonText(self, width=width - 60).pack(padx=20, pady=5)

        # Button skeleton
        SkeletonLoader(self, width=100, height=32, corner_radius=8).pack(padx=20, pady=(10, 20), anchor="w")
