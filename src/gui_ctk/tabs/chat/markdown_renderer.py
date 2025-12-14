"""
Markdown Renderer for Felix Chat Interface

Provides streaming-compatible markdown rendering to CTkTextbox widgets
using Tkinter's tag system for rich text styling.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
import customtkinter as ctk

from .syntax_highlighter import SyntaxHighlighter

if TYPE_CHECKING:
    from ...theme_manager import ThemeManager

logger = logging.getLogger(__name__)


# Maximum buffer size before force-flush (prevents memory issues)
MAX_BUFFER_SIZE = 50000


@dataclass
class ParsedSegment:
    """Represents a parsed chunk ready for rendering."""
    text: str
    tags: List[str] = field(default_factory=list)
    is_code_block: bool = False
    language: str = ""


class IncrementalParser:
    """
    Streaming-aware markdown parser.

    Handles incomplete markdown during streaming by buffering content
    and maintaining state across chunks.
    """

    # Regex patterns for markdown elements
    CODE_FENCE_START = re.compile(r'^```(\w*)\s*$', re.MULTILINE)
    CODE_FENCE_END = re.compile(r'^```\s*$', re.MULTILINE)
    INLINE_CODE = re.compile(r'`([^`\n]+)`')
    BOLD = re.compile(r'\*\*([^*]+)\*\*')
    ITALIC = re.compile(r'(?<!\*)\*([^*]+)\*(?!\*)')
    BOLD_ITALIC = re.compile(r'\*\*\*([^*]+)\*\*\*')
    HEADER = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    LINK = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    BLOCKQUOTE = re.compile(r'^>\s*(.*)$', re.MULTILINE)
    UNORDERED_LIST = re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE)
    ORDERED_LIST = re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE)
    HORIZONTAL_RULE = re.compile(r'^---+\s*$', re.MULTILINE)

    def __init__(self):
        """Initialize the parser state."""
        self.reset()

    def reset(self):
        """Reset parser to initial state."""
        self._buffer = ""
        self._in_code_block = False
        self._code_block_lang = ""
        self._code_block_content = ""

    def parse_chunk(self, chunk: str) -> List[ParsedSegment]:
        """
        Parse a streaming chunk incrementally.

        Returns segments that are safe to render (complete).
        Buffers potentially incomplete constructs.

        Args:
            chunk: New content to parse

        Returns:
            List of ParsedSegments ready for rendering
        """
        self._buffer += chunk
        segments = []

        # Force flush if buffer too large
        if len(self._buffer) > MAX_BUFFER_SIZE:
            logger.warning("Markdown buffer overflow, force flushing")
            return self._force_flush()

        if self._in_code_block:
            segments.extend(self._process_code_block_content())
        else:
            segments.extend(self._process_normal_content())

        return segments

    def finalize(self) -> List[ParsedSegment]:
        """
        Finalize parsing when stream ends.

        Renders any remaining buffered content.
        """
        segments = []

        if self._in_code_block:
            # Unclosed code block - render as code anyway
            if self._code_block_content:
                segments.append(ParsedSegment(
                    text=self._code_block_content,
                    is_code_block=True,
                    language=self._code_block_lang
                ))
            if self._buffer:
                segments.append(ParsedSegment(
                    text=self._buffer,
                    is_code_block=True,
                    language=self._code_block_lang
                ))
        elif self._buffer:
            # Parse any remaining content
            segments.extend(self._parse_inline_markdown(self._buffer))

        self.reset()
        return segments

    def _process_code_block_content(self) -> List[ParsedSegment]:
        """Process content while inside a code block."""
        segments = []

        # Look for closing fence
        match = self.CODE_FENCE_END.search(self._buffer)
        if match:
            # Found closing fence
            code_content = self._code_block_content + self._buffer[:match.start()]
            segments.append(ParsedSegment(
                text=code_content,
                is_code_block=True,
                language=self._code_block_lang
            ))

            # Continue parsing after the fence
            self._buffer = self._buffer[match.end():]
            self._in_code_block = False
            self._code_block_content = ""
            self._code_block_lang = ""

            # Process remaining content
            if self._buffer:
                # Add newline separator after code block
                segments.append(ParsedSegment(text="\n"))
                segments.extend(self._process_normal_content())
        else:
            # Still inside code block - accumulate content
            # Keep last line in buffer (might be partial fence)
            lines = self._buffer.split('\n')
            if len(lines) > 1:
                # Safe to render all but the last line
                complete_lines = '\n'.join(lines[:-1]) + '\n'
                self._code_block_content += complete_lines
                self._buffer = lines[-1]

        return segments

    def _process_normal_content(self) -> List[ParsedSegment]:
        """Process normal (non-code-block) content."""
        segments = []

        # Look for code fence start
        match = self.CODE_FENCE_START.search(self._buffer)
        if match:
            # Found opening fence
            # First, render content before the fence
            before_fence = self._buffer[:match.start()]
            if before_fence:
                segments.extend(self._parse_inline_markdown(before_fence))

            # Enter code block mode
            self._in_code_block = True
            self._code_block_lang = match.group(1) or ""
            self._code_block_content = ""
            self._buffer = self._buffer[match.end():]

            # Try to process code block content
            if self._buffer:
                segments.extend(self._process_code_block_content())
        else:
            # No code fence - but might have incomplete fence at end
            # Keep potential fence prefix in buffer
            safe_content, remaining = self._split_safe_content(self._buffer)
            if safe_content:
                segments.extend(self._parse_inline_markdown(safe_content))
            self._buffer = remaining

        return segments

    def _split_safe_content(self, content: str) -> Tuple[str, str]:
        """
        Split content into safe-to-render and potentially-incomplete parts.

        Keeps incomplete patterns (like partial ```) in the buffer.
        """
        # Check for potential incomplete code fence at end
        # A code fence must start at line beginning with ```
        lines = content.split('\n')
        if not lines:
            return "", ""

        last_line = lines[-1]

        # Check if last line could be start of code fence
        if last_line.startswith('`') and not last_line.startswith('```'):
            # Might be incomplete - keep in buffer
            safe = '\n'.join(lines[:-1])
            if safe:
                safe += '\n'
            return safe, last_line

        # Check for incomplete inline patterns at the very end
        # Keep last 10 chars to check for incomplete patterns
        if len(content) > 10:
            safe = content[:-10]
            remaining = content[-10:]

            # Find safe split point (end of complete line or after complete patterns)
            last_newline = safe.rfind('\n')
            if last_newline > len(safe) - 50:
                return safe[:last_newline + 1], safe[last_newline + 1:] + remaining

        return content, ""

    def _parse_inline_markdown(self, text: str) -> List[ParsedSegment]:
        """
        Parse inline markdown elements (bold, italic, code, etc.).

        Processes block elements first (headers, lists), then inline.
        """
        segments = []

        # Process line by line for block elements
        lines = text.split('\n')
        current_block = []

        for line in lines:
            # Check for block elements
            header_match = self.HEADER.match(line)
            blockquote_match = self.BLOCKQUOTE.match(line)
            hr_match = self.HORIZONTAL_RULE.match(line)
            ul_match = self.UNORDERED_LIST.match(line)
            ol_match = self.ORDERED_LIST.match(line)

            if header_match:
                # Flush current block
                if current_block:
                    segments.extend(self._parse_inline_text('\n'.join(current_block)))
                    current_block = []

                level = len(header_match.group(1))
                content = header_match.group(2)
                segments.append(ParsedSegment(
                    text=content + '\n',
                    tags=[f"header{level}"]
                ))

            elif blockquote_match:
                if current_block:
                    segments.extend(self._parse_inline_text('\n'.join(current_block)))
                    current_block = []

                content = blockquote_match.group(1)
                segments.append(ParsedSegment(
                    text=content + '\n',
                    tags=["blockquote"]
                ))

            elif hr_match:
                if current_block:
                    segments.extend(self._parse_inline_text('\n'.join(current_block)))
                    current_block = []
                segments.append(ParsedSegment(
                    text="---\n",
                    tags=["horizontal_rule"]
                ))

            elif ul_match:
                if current_block:
                    segments.extend(self._parse_inline_text('\n'.join(current_block)))
                    current_block = []

                content = ul_match.group(1)
                # Parse inline content of list item
                inline_segments = self._parse_inline_text(content)
                for seg in inline_segments:
                    seg.tags.append("list_item")
                    seg.text = "  " + seg.text  # Add bullet indent
                segments.extend(inline_segments)
                segments.append(ParsedSegment(text="\n"))

            elif ol_match:
                if current_block:
                    segments.extend(self._parse_inline_text('\n'.join(current_block)))
                    current_block = []

                content = ol_match.group(1)
                inline_segments = self._parse_inline_text(content)
                for seg in inline_segments:
                    seg.tags.append("list_item")
                    seg.text = "  " + seg.text
                segments.extend(inline_segments)
                segments.append(ParsedSegment(text="\n"))

            else:
                current_block.append(line)

        # Process remaining block
        if current_block:
            block_text = '\n'.join(current_block)
            if block_text:
                segments.extend(self._parse_inline_text(block_text))

        return segments

    def _parse_inline_text(self, text: str) -> List[ParsedSegment]:
        """Parse inline formatting (bold, italic, code, links)."""
        segments = []

        # Track position in text
        pos = 0

        while pos < len(text):
            # Find next pattern
            best_match = None
            best_start = len(text)
            best_type = None

            # Check for each pattern type
            patterns = [
                (self.BOLD_ITALIC, "bold_italic"),
                (self.BOLD, "bold"),
                (self.ITALIC, "italic"),
                (self.INLINE_CODE, "inline_code"),
                (self.LINK, "link"),
            ]

            for pattern, ptype in patterns:
                match = pattern.search(text, pos)
                if match and match.start() < best_start:
                    best_match = match
                    best_start = match.start()
                    best_type = ptype

            if best_match:
                # Add plain text before match
                if best_start > pos:
                    segments.append(ParsedSegment(text=text[pos:best_start]))

                # Add formatted segment
                if best_type == "link":
                    link_text = best_match.group(1)
                    link_url = best_match.group(2)
                    segments.append(ParsedSegment(
                        text=link_text,
                        tags=["link"]
                    ))
                elif best_type == "inline_code":
                    code_text = best_match.group(1)
                    segments.append(ParsedSegment(
                        text=code_text,
                        tags=["inline_code"]
                    ))
                else:
                    content = best_match.group(1)
                    segments.append(ParsedSegment(
                        text=content,
                        tags=[best_type]
                    ))

                pos = best_match.end()
            else:
                # No more patterns - add remaining text
                if pos < len(text):
                    segments.append(ParsedSegment(text=text[pos:]))
                break

        return segments if segments else [ParsedSegment(text=text)]

    def _force_flush(self) -> List[ParsedSegment]:
        """Force flush buffer when it gets too large."""
        content = self._buffer
        self._buffer = ""

        if self._in_code_block:
            return [ParsedSegment(
                text=self._code_block_content + content,
                is_code_block=True,
                language=self._code_block_lang
            )]
        else:
            return self._parse_inline_markdown(content)


class MarkdownRenderer:
    """
    Renders markdown to a CTkTextbox using Tkinter tags.

    Supports both streaming (incremental) and static rendering modes.
    Integrates with Felix's theme manager for consistent styling.
    """

    def __init__(self, textbox: ctk.CTkTextbox, theme_manager: "ThemeManager"):
        """
        Initialize the markdown renderer.

        Args:
            textbox: CTkTextbox widget to render into
            theme_manager: Felix theme manager for colors
        """
        self._textbox = textbox
        self._theme_manager = theme_manager
        self._parser = IncrementalParser()
        self._highlighter = SyntaxHighlighter()

        # Get underlying tk.Text widget for tag operations
        self._text_widget = self._textbox._textbox

        self._setup_tags()

        # Register for theme changes
        self._theme_manager.register_callback(self._on_theme_change)

    def _setup_tags(self):
        """Configure markdown tags based on current theme."""
        colors = self._theme_manager.colors
        is_dark = self._theme_manager.is_dark_mode()

        # Base font info
        base_size = 12

        # Tag configurations with improved spacing and colors
        tag_configs = {
            "bold": {
                "font": ("TkDefaultFont", base_size, "bold"),
            },
            "italic": {
                "font": ("TkDefaultFont", base_size, "italic"),
            },
            "bold_italic": {
                "font": ("TkDefaultFont", base_size, "bold italic"),
            },
            "inline_code": {
                "font": ("Courier", base_size),
                "background": colors.get("inline_code_bg", colors["bg_tertiary"]),
                "foreground": "#E06C75" if is_dark else "#E45649",
            },
            "code_block": {
                "font": ("Courier", base_size),
                "background": colors.get("code_bg", colors["bg_primary"]),
                "lmargin1": 10,          # Left margin for indentation
                "lmargin2": 10,          # Continued line margin
                "spacing1": 6,           # Space before code block
                "spacing3": 6,           # Space after code block
            },
            "header1": {
                "font": ("TkDefaultFont", 20, "bold"),
                "spacing1": 14,          # Space before header
                "spacing3": 6,           # Space after header
            },
            "header2": {
                "font": ("TkDefaultFont", 17, "bold"),
                "spacing1": 12,
                "spacing3": 5,
            },
            "header3": {
                "font": ("TkDefaultFont", 15, "bold"),
                "spacing1": 10,
                "spacing3": 4,
            },
            "header4": {
                "font": ("TkDefaultFont", 13, "bold"),
                "spacing1": 8,
                "spacing3": 3,
            },
            "link": {
                "foreground": colors["accent"],
                "underline": True,
            },
            "blockquote": {
                "foreground": colors["fg_secondary"],
                "lmargin1": 20,
                "lmargin2": 20,
                "background": colors.get("code_bg", colors["bg_tertiary"]),
                "spacing1": 4,
                "spacing3": 4,
            },
            "list_item": {
                "lmargin1": 20,
                "lmargin2": 35,
                "spacing1": 2,
                "spacing3": 2,
            },
            "list_bullet": {
                "lmargin1": 20,
                "foreground": colors["fg_secondary"],
            },
            "horizontal_rule": {
                "foreground": colors["border"],
                "spacing1": 8,
                "spacing3": 8,
            },
            "paragraph": {
                "spacing3": 6,           # Space after paragraphs
            },
        }

        # Apply tag configurations
        for tag_name, config in tag_configs.items():
            try:
                self._text_widget.tag_config(tag_name, **config)
            except Exception as e:
                logger.debug(f"Failed to configure tag {tag_name}: {e}")

        # Setup syntax highlighting tags
        self._highlighter.setup_syntax_tags(self._text_widget, self._theme_manager)

        logger.debug("Markdown tags configured")

    def _on_theme_change(self, mode: str):
        """Reconfigure tags when theme changes."""
        self._setup_tags()

    def append_markdown(self, chunk: str):
        """
        Append streaming markdown content.

        Parses incrementally and renders complete segments.

        Args:
            chunk: New markdown content to append
        """
        segments = self._parser.parse_chunk(chunk)
        self._render_segments(segments)

    def finalize(self):
        """Finalize rendering after streaming completes."""
        segments = self._parser.finalize()
        self._render_segments(segments)

    def set_content(self, content: str):
        """
        Set complete content (non-streaming mode).

        Clears existing content and renders the full markdown.

        Args:
            content: Full markdown content to render
        """
        self._parser.reset()

        # Parse entire content
        self._parser._buffer = content
        segments = self._parser.finalize()

        # Clear and render
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._render_segments(segments)
        self._textbox.configure(state="disabled")

    def _render_segments(self, segments: List[ParsedSegment]):
        """
        Render parsed segments to the textbox.

        Args:
            segments: List of ParsedSegments to render
        """
        if not segments:
            return

        # Use underlying tk.Text widget for state check (CTkTextbox doesn't support cget("state"))
        was_disabled = str(self._text_widget.cget("state")) == "disabled"
        if was_disabled:
            self._textbox.configure(state="normal")

        for segment in segments:
            if segment.is_code_block:
                self._render_code_block(segment)
            else:
                self._render_text_segment(segment)

        if was_disabled:
            self._textbox.configure(state="disabled")

    def _render_text_segment(self, segment: ParsedSegment):
        """Render a text segment with tags."""
        if not segment.text:
            return

        # Get current end position
        start_index = self._text_widget.index("end-1c")

        # Insert text
        self._text_widget.insert("end", segment.text)

        # Apply tags
        if segment.tags:
            end_index = self._text_widget.index("end-1c")
            for tag in segment.tags:
                try:
                    self._text_widget.tag_add(tag, start_index, end_index)
                except Exception as e:
                    logger.debug(f"Failed to apply tag {tag}: {e}")

    def _render_code_block(self, segment: ParsedSegment):
        """Render a code block with syntax highlighting."""
        if not segment.text:
            return

        code = segment.text
        language = segment.language

        # Apply syntax highlighting if available
        if language and self._highlighter.available:
            highlighted = self._highlighter.highlight_code(code, language)

            for text, tag in highlighted:
                start_index = self._text_widget.index("end-1c")
                self._text_widget.insert("end", text)
                end_index = self._text_widget.index("end-1c")

                # Apply code_block base style
                self._text_widget.tag_add("code_block", start_index, end_index)

                # Apply syntax tag on top
                if tag:
                    try:
                        self._text_widget.tag_add(tag, start_index, end_index)
                    except Exception as e:
                        logger.debug(f"Failed to apply syntax tag {tag}: {e}")
        else:
            # No highlighting - render as plain code block
            start_index = self._text_widget.index("end-1c")
            self._text_widget.insert("end", code)
            end_index = self._text_widget.index("end-1c")
            self._text_widget.tag_add("code_block", start_index, end_index)

    def reset(self):
        """Reset the renderer state."""
        self._parser.reset()

    def cleanup(self):
        """Cleanup resources when renderer is destroyed."""
        try:
            self._theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass
