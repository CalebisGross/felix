"""
Felix GUI Design System Constants

Centralized styling constants for consistent UX across all CTK components.
Import these values instead of using hardcoded numbers.

Usage:
    from .styles import BUTTON_MD, FONT_BODY, SPACE_SM

    button = ctk.CTkButton(parent, width=BUTTON_MD[0], height=BUTTON_MD[1])
    label = ctk.CTkLabel(parent, font=ctk.CTkFont(size=FONT_BODY))
"""

# =============================================================================
# BUTTON SIZES (width, height)
# =============================================================================

BUTTON_XS = (60, 28)      # Icon-only buttons, close/X buttons
BUTTON_SM = (80, 32)      # Short actions: Clear, Cancel, Search, Refresh
BUTTON_MD = (120, 32)     # Standard actions: Save, Edit, Delete
BUTTON_LG = (150, 36)     # Primary CTAs: Start, Run, Submit, Save All

# Icon-only square button
BUTTON_ICON = (32, 32)

# =============================================================================
# FONT SIZES
# =============================================================================

FONT_TITLE = 18           # Page/tab titles
FONT_SECTION = 14         # Section headers, bold labels
FONT_BODY = 12            # Default body text, form labels
FONT_CAPTION = 11         # Muted text, secondary info, hints
FONT_SMALL = 10           # Tiny labels, timestamps

# Large display numbers (for status cards)
FONT_DISPLAY = 24

# =============================================================================
# SPACING SCALE
# =============================================================================

SPACE_XS = 5              # Tight spacing (between related elements)
SPACE_SM = 10             # Standard spacing (default padding)
SPACE_MD = 15             # Medium spacing (between sections)
SPACE_LG = 20             # Large spacing (page margins, major gaps)
SPACE_XL = 30             # Extra large (major section breaks)

# =============================================================================
# COMPONENT SIZES
# =============================================================================

# Status Cards
CARD_SM = 150
CARD_MD = 180
CARD_LG = 220

# TreeView
TREEVIEW_ROW_HEIGHT = 28

# Input Fields
INPUT_SM = 100
INPUT_MD = 200
INPUT_LG = 300
INPUT_XL = 400

# Sidebar/Panel widths
SIDEBAR_WIDTH = 300

# Textbox heights
TEXTBOX_SM = 60
TEXTBOX_MD = 150
TEXTBOX_LG = 250
TEXTBOX_XL = 400

# =============================================================================
# CORNER RADIUS SCALE
# =============================================================================

RADIUS_SM = 4      # inputs, small elements
RADIUS_MD = 8      # buttons, small cards
RADIUS_LG = 12     # cards, dialogs
RADIUS_XL = 16     # large panels
RADIUS_FULL = 9999 # pills, badges

# =============================================================================
# RESPONSIVE BREAKPOINTS
# =============================================================================

# Breakpoints for responsive layouts
BREAKPOINT_COMPACT = 1280    # < 1280px: single column
BREAKPOINT_STANDARD = 1920   # 1280-1920px: 2-column
BREAKPOINT_WIDE = 2560       # 1920-2560px: full 2-column
# 2560px+: ultrawide 3-column

# Responsive spacing multipliers
SPACE_MULT_COMPACT = 0.8
SPACE_MULT_STANDARD = 1.0
SPACE_MULT_WIDE = 1.2
SPACE_MULT_ULTRAWIDE = 1.4

# =============================================================================
# COLOR TOKENS
# =============================================================================

# These reference the theme_manager.py COLORS dictionary
# Import from theme_manager for dynamic dark/light mode support
COLOR_ACCENT = "#0C7BDC"            # Helix Blue - Interactive elements, buttons
COLOR_ACCENT_SECONDARY = "#45A29E"  # Elegant Teal - Highlights, success states
COLOR_WARNING = "#A94442"           # Subdued Crimson - Alerts, BLOCKED indicators

# =============================================================================
# SEPARATOR
# =============================================================================

SEPARATOR_HEIGHT = 2
SEPARATOR_WIDTH = 2

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def button_kwargs(size: str = "md") -> dict:
    """
    Get button keyword arguments for a given size.

    Args:
        size: One of 'xs', 'sm', 'md', 'lg', 'icon'

    Returns:
        Dict with 'width' and 'height' keys

    Usage:
        button = ctk.CTkButton(parent, text="Save", **button_kwargs("md"))
    """
    sizes = {
        "xs": BUTTON_XS,
        "sm": BUTTON_SM,
        "md": BUTTON_MD,
        "lg": BUTTON_LG,
        "icon": BUTTON_ICON,
    }
    w, h = sizes.get(size, BUTTON_MD)
    return {"width": w, "height": h}


def padding(size: str = "sm") -> tuple:
    """
    Get padding tuple for pack/grid.

    Args:
        size: One of 'xs', 'sm', 'md', 'lg', 'xl'

    Returns:
        Tuple (padx, pady) with same value for both

    Usage:
        frame.pack(fill="x", **padding("md"))  # padx=15, pady=15
    """
    sizes = {
        "xs": SPACE_XS,
        "sm": SPACE_SM,
        "md": SPACE_MD,
        "lg": SPACE_LG,
        "xl": SPACE_XL,
    }
    val = sizes.get(size, SPACE_SM)
    return {"padx": val, "pady": val}
