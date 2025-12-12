"""
Premium Button Component with Variants and Interactive States

Features:
- Four variants: primary (Helix Blue), secondary (Elegant Teal), ghost, danger (Subdued Crimson)
- Hover state: Lighter/darker background on mouse over
- Press state: Darker background on click (active state)
- Focus state: Accent border ring
- Disabled state: Muted appearance
- Optional icon support
- Size variants: sm, md, lg
"""

import customtkinter as ctk
from typing import Optional, Callable, Literal
from ..theme_manager import ThemeManager, get_theme_manager
from ..styles import BUTTON_SM, BUTTON_MD, BUTTON_LG, RADIUS_MD


class PremiumButton(ctk.CTkButton):
    """
    Enhanced button with visual feedback and multiple variants.

    Args:
        parent: Parent widget
        text: Button text
        variant: Button style - "primary", "secondary", "ghost", "danger"
        size: Button size - "sm", "md", "lg"
        icon: Optional icon text (emoji or symbol)
        command: Callback function
        disabled: Whether button is disabled
        **kwargs: Additional CTkButton arguments
    """

    def __init__(
        self,
        parent,
        text: str = "",
        variant: Literal["primary", "secondary", "ghost", "danger"] = "primary",
        size: Literal["sm", "md", "lg"] = "md",
        icon: Optional[str] = None,
        command: Optional[Callable] = None,
        disabled: bool = False,
        **kwargs
    ):
        self.theme_manager = get_theme_manager()
        self.variant = variant
        self.size = size
        self.icon = icon
        self._disabled = disabled
        self._is_pressed = False
        self._is_hovered = False

        # Build display text
        display_text = f"{icon} {text}" if icon else text

        # Get size dimensions
        width, height = self._get_size_dimensions()

        # Get colors for variant
        fg_color, hover_color, text_color = self._get_variant_colors()

        # Initialize base button
        super().__init__(
            parent,
            text=display_text,
            width=width,
            height=height,
            fg_color=fg_color,
            hover_color=hover_color,
            text_color=text_color,
            corner_radius=RADIUS_MD,
            command=self._on_click if command else None,
            state="disabled" if disabled else "normal",
            **kwargs
        )

        # Store original command
        self._user_command = command

        # Bind hover events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

        # Bind press/release events for active state
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

        # Apply initial state
        self._update_appearance()

    def _get_size_dimensions(self) -> tuple:
        """Get width and height for current size."""
        sizes = {
            "sm": BUTTON_SM,
            "md": BUTTON_MD,
            "lg": BUTTON_LG,
        }
        return sizes.get(self.size, BUTTON_MD)

    def _get_variant_colors(self) -> tuple:
        """Get fg_color, hover_color, text_color for current variant."""
        colors = self.theme_manager.colors

        if self.variant == "primary":
            # Helix Blue
            return (colors["accent"], colors["accent_hover"], "#FFFFFF")
        elif self.variant == "secondary":
            # Elegant Teal
            return (colors["success"], self._lighten_color(colors["success"]), "#FFFFFF")
        elif self.variant == "ghost":
            # Transparent with border
            return ("transparent", colors["bg_hover"], colors["fg_primary"])
        elif self.variant == "danger":
            # Subdued Crimson
            return (colors["error"], self._darken_color(colors["error"]), "#FFFFFF")
        else:
            # Default to primary
            return (colors["accent"], colors["accent_hover"], "#FFFFFF")

    def _lighten_color(self, hex_color: str, factor: float = 0.2) -> str:
        """Lighten a hex color by a factor."""
        # Simple lightening by adding to RGB values
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))

        return f"#{r:02x}{g:02x}{b:02x}"

    def _darken_color(self, hex_color: str, factor: float = 0.2) -> str:
        """Darken a hex color by a factor."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))

        return f"#{r:02x}{g:02x}{b:02x}"

    def _update_appearance(self):
        """Update button appearance based on current state."""
        if self._disabled:
            # Disabled state: muted colors
            colors = self.theme_manager.colors
            self.configure(
                fg_color=colors["bg_secondary"],
                text_color=colors["fg_muted"],
                state="disabled"
            )
            return

        fg_color, hover_color, text_color = self._get_variant_colors()

        # Apply pressed state (darker)
        if self._is_pressed:
            fg_color = self._darken_color(fg_color, 0.3)
        # Apply hover state
        elif self._is_hovered:
            fg_color = hover_color

        # Ghost variant needs border
        if self.variant == "ghost":
            self.configure(
                fg_color=fg_color,
                text_color=text_color,
                border_width=2,
                border_color=self.theme_manager.colors["border"]
            )
        else:
            self.configure(
                fg_color=fg_color,
                text_color=text_color,
                border_width=0
            )

    def _on_enter(self, event):
        """Handle mouse enter (hover start)."""
        if not self._disabled:
            self._is_hovered = True
            self._update_appearance()

    def _on_leave(self, event):
        """Handle mouse leave (hover end)."""
        self._is_hovered = False
        self._is_pressed = False
        self._update_appearance()

    def _on_press(self, event):
        """Handle button press down."""
        if not self._disabled:
            self._is_pressed = True
            self._update_appearance()

    def _on_release(self, event):
        """Handle button release."""
        if not self._disabled:
            self._is_pressed = False
            self._update_appearance()

    def _on_click(self):
        """Handle button click."""
        if not self._disabled and self._user_command:
            self._user_command()

    def _on_theme_change(self, mode: str):
        """Handle theme change."""
        self._update_appearance()

    def set_disabled(self, disabled: bool):
        """Set button disabled state."""
        self._disabled = disabled
        self._update_appearance()

    def set_variant(self, variant: Literal["primary", "secondary", "ghost", "danger"]):
        """Change button variant."""
        self.variant = variant
        self._update_appearance()

    def destroy(self):
        """Clean up on destruction."""
        self.theme_manager.unregister_callback(self._on_theme_change)
        super().destroy()
