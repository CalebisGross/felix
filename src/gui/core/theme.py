"""Theme system for Felix GUI with dark/light mode support."""

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPalette, QColor, QFontMetrics
from PySide6.QtWidgets import QApplication, QPushButton


def create_button(text: str, min_width: int = None) -> QPushButton:
    """Create a button that auto-sizes to fit its text.

    Use this instead of manually calling setFixedWidth() on buttons.
    The button will use the global stylesheet padding and auto-size.

    Args:
        text: Button text
        min_width: Optional minimum width (use only for very short text like "OK")

    Returns:
        QPushButton configured for proper sizing
    """
    btn = QPushButton(text)
    if min_width:
        btn.setMinimumWidth(min_width)
    # Don't use setFixedWidth - let the button auto-size based on text
    return btn


def calculate_text_width(text: str, padding: int = 32) -> int:
    """Calculate the minimum width needed for text with padding.

    Useful when you need to know how wide a button should be.

    Args:
        text: The text to measure
        padding: Extra space for button borders/padding (default 32px)

    Returns:
        Width in pixels
    """
    app = QApplication.instance()
    if app:
        fm = QFontMetrics(app.font())
        return fm.horizontalAdvance(text) + padding
    return len(text) * 8 + padding  # Rough estimate if no app


class DarkColors:
    """Color constants for the Felix dark theme."""

    # Base colors
    BACKGROUND = "#1a1b26"
    BACKGROUND_LIGHT = "#1f2029"
    SURFACE = "#24283b"
    SURFACE_LIGHT = "#2a2e3f"

    # Text colors
    TEXT_PRIMARY = "#c0caf5"
    TEXT_SECONDARY = "#a9b1d6"
    TEXT_MUTED = "#565f89"

    # Accent colors
    ACCENT = "#7aa2f7"
    ACCENT_HOVER = "#89b4fa"
    SUCCESS = "#9ece6a"
    WARNING = "#e0af68"
    ERROR = "#f7768e"

    # Border colors
    BORDER = "#414868"
    BORDER_LIGHT = "#565f89"

    # Message bubbles
    USER_BUBBLE = "#364a82"
    ASSISTANT_BUBBLE = "#24283b"

    # Status colors
    STATUS_RUNNING = "#9ece6a"
    STATUS_STOPPED = "#f7768e"
    STATUS_STARTING = "#e0af68"


class LightColors:
    """Color constants for the Felix light theme."""

    # Base colors
    BACKGROUND = "#f5f5f5"
    BACKGROUND_LIGHT = "#ffffff"
    SURFACE = "#ffffff"
    SURFACE_LIGHT = "#fafafa"

    # Text colors
    TEXT_PRIMARY = "#1a1a2e"
    TEXT_SECONDARY = "#4a4a6a"
    TEXT_MUTED = "#8888a0"

    # Accent colors
    ACCENT = "#4a6cf7"
    ACCENT_HOVER = "#5a7cff"
    SUCCESS = "#2e8b57"
    WARNING = "#d68a00"
    ERROR = "#dc3545"

    # Border colors
    BORDER = "#d0d0d8"
    BORDER_LIGHT = "#e0e0e8"

    # Message bubbles
    USER_BUBBLE = "#e3e8f8"
    ASSISTANT_BUBBLE = "#f0f0f5"

    # Status colors
    STATUS_RUNNING = "#2e8b57"
    STATUS_STOPPED = "#dc3545"
    STATUS_STARTING = "#d68a00"


# Default to dark theme - this is the active color reference
Colors = DarkColors


class ThemeManager(QObject):
    """Manages theme switching between dark and light modes."""

    theme_changed = Signal(str)  # Emits "dark" or "light"

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._current_theme = "dark"
        self._initialized = True

    @property
    def current_theme(self) -> str:
        return self._current_theme

    @property
    def is_dark(self) -> bool:
        return self._current_theme == "dark"

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self.set_theme(new_theme)

    def set_theme(self, theme: str):
        """Set the theme to dark or light."""
        global Colors

        if theme not in ("dark", "light"):
            return

        self._current_theme = theme

        if theme == "dark":
            Colors = DarkColors
        else:
            Colors = LightColors

        # Apply to the application
        app = QApplication.instance()
        if app:
            apply_theme(app, theme)

        self.theme_changed.emit(theme)

    def get_colors(self):
        """Get the current color class."""
        return DarkColors if self._current_theme == "dark" else LightColors


def get_theme_manager() -> ThemeManager:
    """Get the singleton theme manager instance."""
    return ThemeManager()


def apply_theme(app: QApplication, theme: str = "dark") -> None:
    """Apply theme to the application using QPalette.

    Args:
        app: The QApplication instance
        theme: Either "dark" or "light"
    """
    colors = DarkColors if theme == "dark" else LightColors
    palette = QPalette()

    # Window and general backgrounds
    palette.setColor(QPalette.ColorRole.Window, QColor(colors.BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors.TEXT_PRIMARY))

    # Base (text input backgrounds)
    palette.setColor(QPalette.ColorRole.Base, QColor(colors.SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors.SURFACE_LIGHT))

    # Text
    palette.setColor(QPalette.ColorRole.Text, QColor(colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff" if theme == "dark" else "#000000"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors.TEXT_MUTED))

    # Buttons
    palette.setColor(QPalette.ColorRole.Button, QColor(colors.SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors.TEXT_PRIMARY))

    # Highlights (selection)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors.ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

    # Links
    palette.setColor(QPalette.ColorRole.Link, QColor(colors.ACCENT))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor("#bb9af7" if theme == "dark" else "#6a5acd"))

    # Tooltips
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors.SURFACE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colors.TEXT_PRIMARY))

    app.setPalette(palette)

    # Additional QSS for finer control
    app.setStyleSheet(get_stylesheet(colors))


def apply_dark_theme(app: QApplication) -> None:
    """Apply dark theme to the application. Convenience wrapper."""
    apply_theme(app, "dark")


def get_stylesheet(colors=None) -> str:
    """Return the QSS stylesheet for detailed styling.

    Args:
        colors: Color class to use (DarkColors or LightColors). Defaults to current Colors.
    """
    if colors is None:
        colors = Colors
    return f"""
        /* Global */
        QWidget {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
        }}

        /* Scrollbars */
        QScrollBar:vertical {{
            background: {Colors.BACKGROUND};
            width: 12px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {Colors.BORDER};
            min-height: 30px;
            border-radius: 6px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {Colors.BORDER_LIGHT};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background: {Colors.BACKGROUND};
            height: 12px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: {Colors.BORDER};
            min-width: 30px;
            border-radius: 6px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {Colors.BORDER_LIGHT};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}

        /* Buttons */
        QPushButton {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {Colors.SURFACE_LIGHT};
            border-color: {Colors.BORDER_LIGHT};
        }}
        QPushButton:pressed {{
            background-color: {Colors.BACKGROUND};
        }}
        QPushButton:disabled {{
            background-color: {Colors.BACKGROUND};
            color: {Colors.TEXT_MUTED};
            border-color: {Colors.BACKGROUND};
        }}

        /* Primary button */
        QPushButton[primary="true"] {{
            background-color: {Colors.ACCENT};
            color: #ffffff;
            border: none;
        }}
        QPushButton[primary="true"]:hover {{
            background-color: {Colors.ACCENT_HOVER};
        }}
        QPushButton[primary="true"]:pressed {{
            background-color: #6a8fd1;
        }}

        /* Success button */
        QPushButton[success="true"] {{
            background-color: {Colors.SUCCESS};
            color: #1a1b26;
            border: none;
        }}

        /* Danger button */
        QPushButton[danger="true"] {{
            background-color: {Colors.ERROR};
            color: #ffffff;
            border: none;
        }}

        /* Text inputs */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            padding: 8px;
            selection-background-color: {Colors.ACCENT};
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {Colors.ACCENT};
        }}

        /* Combo boxes */
        QComboBox {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            padding: 6px 12px;
        }}
        QComboBox:hover {{
            border-color: {Colors.BORDER_LIGHT};
        }}
        QComboBox::drop-down {{
            border: none;
            padding-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            border: 1px solid {Colors.BORDER};
            selection-background-color: {Colors.ACCENT};
        }}

        /* Labels */
        QLabel {{
            color: {Colors.TEXT_PRIMARY};
        }}
        QLabel[muted="true"] {{
            color: {Colors.TEXT_MUTED};
        }}
        QLabel[heading="true"] {{
            font-size: 16px;
            font-weight: 600;
        }}

        /* Group boxes */
        QGroupBox {{
            border: 1px solid {Colors.BORDER};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
            color: {Colors.TEXT_SECONDARY};
        }}

        /* Splitter */
        QSplitter::handle {{
            background-color: {Colors.BORDER};
        }}
        QSplitter::handle:horizontal {{
            width: 1px;
        }}
        QSplitter::handle:vertical {{
            height: 1px;
        }}

        /* Menu */
        QMenuBar {{
            background-color: {Colors.BACKGROUND};
            color: {Colors.TEXT_PRIMARY};
        }}
        QMenuBar::item:selected {{
            background-color: {Colors.SURFACE};
        }}
        QMenu {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            border: 1px solid {Colors.BORDER};
        }}
        QMenu::item:selected {{
            background-color: {Colors.ACCENT};
        }}

        /* Status bar */
        QStatusBar {{
            background-color: {Colors.BACKGROUND};
            color: {Colors.TEXT_SECONDARY};
        }}

        /* Tool tips */
        QToolTip {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
            padding: 4px 8px;
        }}
    """
