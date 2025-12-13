"""
Syntax Highlighter for Felix Chat Markdown Renderer

Uses Pygments to tokenize code and map tokens to Tkinter text tags
for syntax highlighting in CTkTextbox widgets.
"""

import logging
from typing import List, Tuple, Optional, TYPE_CHECKING

try:
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, TextLexer
    from pygments.token import Token
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

if TYPE_CHECKING:
    from ...theme_manager import ThemeManager

logger = logging.getLogger(__name__)


# Map Pygments token types to tag names
TOKEN_TAG_MAP = {
    Token.Keyword: "syn_keyword",
    Token.Keyword.Constant: "syn_keyword",
    Token.Keyword.Declaration: "syn_keyword",
    Token.Keyword.Namespace: "syn_keyword",
    Token.Keyword.Pseudo: "syn_keyword",
    Token.Keyword.Reserved: "syn_keyword",
    Token.Keyword.Type: "syn_type",

    Token.Name.Builtin: "syn_builtin",
    Token.Name.Builtin.Pseudo: "syn_builtin",
    Token.Name.Function: "syn_function",
    Token.Name.Function.Magic: "syn_function",
    Token.Name.Class: "syn_class",
    Token.Name.Decorator: "syn_decorator",
    Token.Name.Exception: "syn_class",
    Token.Name.Variable: "syn_variable",
    Token.Name.Variable.Magic: "syn_variable",

    Token.String: "syn_string",
    Token.String.Affix: "syn_string",
    Token.String.Backtick: "syn_string",
    Token.String.Char: "syn_string",
    Token.String.Doc: "syn_string",
    Token.String.Double: "syn_string",
    Token.String.Escape: "syn_escape",
    Token.String.Heredoc: "syn_string",
    Token.String.Interpol: "syn_escape",
    Token.String.Single: "syn_string",

    Token.Number: "syn_number",
    Token.Number.Bin: "syn_number",
    Token.Number.Float: "syn_number",
    Token.Number.Hex: "syn_number",
    Token.Number.Integer: "syn_number",
    Token.Number.Oct: "syn_number",

    Token.Operator: "syn_operator",
    Token.Operator.Word: "syn_keyword",
    Token.Punctuation: "syn_punctuation",

    Token.Comment: "syn_comment",
    Token.Comment.Hashbang: "syn_comment",
    Token.Comment.Multiline: "syn_comment",
    Token.Comment.Single: "syn_comment",
    Token.Comment.Special: "syn_comment",

    Token.Literal: "syn_string",
    Token.Generic.Heading: "syn_keyword",
    Token.Generic.Subheading: "syn_keyword",
    Token.Generic.Emph: "syn_italic",
    Token.Generic.Strong: "syn_bold",
}

# Dark mode syntax colors (One Dark inspired)
SYNTAX_COLORS_DARK = {
    "syn_keyword": "#C678DD",    # Purple - keywords
    "syn_type": "#E5C07B",       # Yellow - types
    "syn_builtin": "#E06C75",    # Red - builtins
    "syn_function": "#61AFEF",   # Blue - functions
    "syn_class": "#E5C07B",      # Yellow - classes
    "syn_decorator": "#C678DD",  # Purple - decorators
    "syn_variable": "#E06C75",   # Red - special variables
    "syn_string": "#98C379",     # Green - strings
    "syn_escape": "#56B6C2",     # Cyan - escape sequences
    "syn_number": "#D19A66",     # Orange - numbers
    "syn_operator": "#56B6C2",   # Cyan - operators
    "syn_punctuation": "#ABB2BF", # Light gray - punctuation
    "syn_comment": "#5C6370",    # Dark gray - comments
    "syn_italic": "#C678DD",     # Purple italic
    "syn_bold": "#E06C75",       # Red bold
}

# Light mode syntax colors (One Light inspired)
SYNTAX_COLORS_LIGHT = {
    "syn_keyword": "#A626A4",    # Purple
    "syn_type": "#C18401",       # Yellow-brown
    "syn_builtin": "#E45649",    # Red
    "syn_function": "#4078F2",   # Blue
    "syn_class": "#C18401",      # Yellow-brown
    "syn_decorator": "#A626A4",  # Purple
    "syn_variable": "#E45649",   # Red
    "syn_string": "#50A14F",     # Green
    "syn_escape": "#0184BC",     # Cyan
    "syn_number": "#986801",     # Orange-brown
    "syn_operator": "#0184BC",   # Cyan
    "syn_punctuation": "#383A42", # Dark gray
    "syn_comment": "#A0A1A7",    # Light gray
    "syn_italic": "#A626A4",     # Purple
    "syn_bold": "#E45649",       # Red
}

# Language aliases for common variations
LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "yml": "yaml",
    "md": "markdown",
}


class SyntaxHighlighter:
    """
    Syntax highlighter using Pygments for code tokenization.

    Provides theme-aware syntax highlighting that integrates with
    Felix's theme manager and CTkTextbox tag system.
    """

    def __init__(self):
        """Initialize the syntax highlighter."""
        self._tags_configured = False

        if not PYGMENTS_AVAILABLE:
            logger.warning("Pygments not available - syntax highlighting disabled")

    def setup_syntax_tags(self, text_widget, theme_manager: "ThemeManager"):
        """
        Configure syntax highlighting tags on a text widget.

        Args:
            text_widget: The underlying tk.Text widget (textbox._textbox)
            theme_manager: Felix theme manager for color access
        """
        if not PYGMENTS_AVAILABLE:
            return

        is_dark = theme_manager.is_dark_mode()
        colors = SYNTAX_COLORS_DARK if is_dark else SYNTAX_COLORS_LIGHT

        for tag_name, color in colors.items():
            try:
                config = {"foreground": color}

                # Add font styles for special tags
                if tag_name == "syn_italic":
                    config["font"] = ("TkDefaultFont", 12, "italic")
                elif tag_name == "syn_bold":
                    config["font"] = ("TkDefaultFont", 12, "bold")

                text_widget.tag_config(tag_name, **config)
            except Exception as e:
                logger.debug(f"Failed to configure tag {tag_name}: {e}")

        self._tags_configured = True
        logger.debug(f"Syntax tags configured for {'dark' if is_dark else 'light'} mode")

    def highlight_code(self, code: str, language: str) -> List[Tuple[str, Optional[str]]]:
        """
        Tokenize code and return (text, tag) pairs for rendering.

        Args:
            code: The source code to highlight
            language: Programming language (e.g., "python", "json")

        Returns:
            List of (text_segment, tag_name) tuples.
            tag_name is None for unstyled text.
        """
        if not PYGMENTS_AVAILABLE:
            return [(code, None)]

        # Normalize language name
        language = LANGUAGE_ALIASES.get(language.lower(), language.lower())

        try:
            lexer = get_lexer_by_name(language, stripall=False)
        except Exception:
            # Unknown language - return plain text
            logger.debug(f"No lexer for language: {language}")
            return [(code, None)]

        result = []

        try:
            for token_type, value in lex(code, lexer):
                if not value:
                    continue

                # Find the most specific matching tag
                tag = self._get_tag_for_token(token_type)
                result.append((value, tag))

        except Exception as e:
            logger.debug(f"Highlighting failed: {e}")
            return [(code, None)]

        return result

    def _get_tag_for_token(self, token_type) -> Optional[str]:
        """
        Get the tag name for a Pygments token type.

        Walks up the token type hierarchy to find a matching tag.
        """
        # Direct match
        if token_type in TOKEN_TAG_MAP:
            return TOKEN_TAG_MAP[token_type]

        # Walk up the hierarchy
        while token_type.parent:
            token_type = token_type.parent
            if token_type in TOKEN_TAG_MAP:
                return TOKEN_TAG_MAP[token_type]

        # No match - return None for default styling
        return None

    def get_supported_languages(self) -> List[str]:
        """Get list of commonly supported languages."""
        return [
            "python", "javascript", "typescript", "json", "yaml",
            "bash", "sql", "html", "css", "markdown", "rust", "go",
            "java", "c", "cpp", "ruby", "php", "swift", "kotlin"
        ]

    @property
    def available(self) -> bool:
        """Check if syntax highlighting is available."""
        return PYGMENTS_AVAILABLE
