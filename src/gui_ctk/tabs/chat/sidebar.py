"""
Conversation Sidebar Component for Felix Chat Interface

LM Studio-style sidebar with:
- Scrollable conversation list with folder support
- Right-click context menu for conversation management
- Search/filter functionality
- Pinned conversations section
- New chat and folder buttons
- Visual selection states and hover effects
- Nested folder support with expand/collapse
"""

import customtkinter as ctk
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Callable, Dict
import logging

from ...theme_manager import get_theme_manager
from ...styles import (
    BUTTON_SM, BUTTON_ICON,
    FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD,
    RADIUS_SM, RADIUS_MD, SIDEBAR_WIDTH
)
from ...components.search_entry import SearchEntry

logger = logging.getLogger(__name__)


@dataclass
class ConversationItem:
    """Represents a chat conversation/session."""
    session_id: str
    title: str
    last_active: datetime
    message_count: int
    pinned: bool = False
    folder_id: Optional[str] = None


@dataclass
class FolderItem:
    """Represents a folder for organizing conversations."""
    folder_id: str
    name: str
    parent_id: Optional[str] = None
    expanded: bool = True


class ConversationSidebar(ctk.CTkFrame):
    """
    LM Studio-style sidebar for managing chat conversations.

    Features:
    - Hierarchical folder organization with expand/collapse
    - Pinned conversations at the top
    - Search/filter functionality
    - Right-click context menus
    - Visual selection states
    - New chat and folder creation
    """

    def __init__(
        self,
        master,
        on_select: Optional[Callable[[str], None]] = None,
        on_new_chat: Optional[Callable[[], None]] = None,
        on_new_folder: Optional[Callable[[], None]] = None,
        on_rename: Optional[Callable[[str, str], None]] = None,
        on_delete: Optional[Callable[[str], None]] = None,
        on_duplicate: Optional[Callable[[str], None]] = None,
        on_move: Optional[Callable[[str, Optional[str]], None]] = None,
        on_pin: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        """
        Initialize the conversation sidebar.

        Args:
            master: Parent widget
            on_select: Callback when conversation is selected (session_id)
            on_new_chat: Callback when new chat button is clicked
            on_new_folder: Callback when new folder button is clicked
            on_rename: Callback for renaming (item_id, new_name)
            on_delete: Callback for deleting (item_id)
            on_duplicate: Callback for duplicating (session_id)
            on_move: Callback for moving (session_id, folder_id)
            on_pin: Callback for pinning/unpinning (session_id)
            **kwargs: Additional CTkFrame arguments
        """
        super().__init__(master, **kwargs)

        self.theme_manager = get_theme_manager()
        self.on_select = on_select
        self.on_new_chat = on_new_chat
        self.on_new_folder = on_new_folder
        self.on_rename = on_rename
        self.on_delete = on_delete
        self.on_duplicate = on_duplicate
        self.on_move = on_move
        self.on_pin = on_pin

        # Data storage
        self.conversations: List[ConversationItem] = []
        self.folders: List[FolderItem] = []
        self.selected_session_id: Optional[str] = None
        self.search_query: str = ""

        # UI element tracking
        self._conversation_widgets: Dict[str, ctk.CTkFrame] = {}
        self._folder_widgets: Dict[str, ctk.CTkFrame] = {}
        self._context_menu: Optional[ctk.CTkToplevel] = None

        self.configure(width=SIDEBAR_WIDTH)
        self._setup_ui()

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def _setup_ui(self):
        """Setup the sidebar UI components."""
        self.grid_rowconfigure(2, weight=1)  # Scrollable content gets all space
        self.grid_columnconfigure(0, weight=1)

        # Header with New Chat/Folder buttons
        self._setup_header()

        # Search bar
        self._setup_search()

        # Scrollable conversation list
        self._setup_conversation_list()

    def _setup_header(self):
        """Setup header with action buttons."""
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_SM)
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=1)

        # New Chat button
        self.new_chat_btn = ctk.CTkButton(
            header,
            text="+ Chat",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            font=ctk.CTkFont(size=FONT_CAPTION),
            corner_radius=RADIUS_SM,
            command=self._handle_new_chat
        )
        self.new_chat_btn.grid(row=0, column=0, sticky="w", padx=(0, SPACE_XS))

        # New Folder button
        self.new_folder_btn = ctk.CTkButton(
            header,
            text="+ Folder",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            font=ctk.CTkFont(size=FONT_CAPTION),
            corner_radius=RADIUS_SM,
            fg_color="transparent",
            border_width=1,
            border_color=self.theme_manager.get_color("border"),
            command=self._handle_new_folder
        )
        self.new_folder_btn.grid(row=0, column=1, sticky="e")

    def _setup_search(self):
        """Setup search bar."""
        search_container = ctk.CTkFrame(self, fg_color="transparent")
        search_container.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        search_container.grid_columnconfigure(0, weight=1)

        self.search_entry = SearchEntry(
            search_container,
            placeholder="Search conversations...",
            on_search=self._handle_search,
            on_clear=self._handle_search_clear,
            width=SIDEBAR_WIDTH - (SPACE_SM * 4)
        )
        self.search_entry.pack(fill="x", expand=True)

    def _setup_conversation_list(self):
        """Setup scrollable conversation list."""
        self.conversation_list = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0
        )
        self.conversation_list.grid(row=2, column=0, sticky="nsew", padx=SPACE_XS, pady=(0, SPACE_SM))
        self.conversation_list.grid_columnconfigure(0, weight=1)

    def load_conversations(
        self,
        conversations: List[ConversationItem],
        folders: Optional[List[FolderItem]] = None
    ):
        """
        Load conversations and folders into the sidebar.

        Args:
            conversations: List of conversation items
            folders: List of folder items (optional)
        """
        self.conversations = conversations
        self.folders = folders or []
        self.refresh()
        logger.debug(f"Loaded {len(conversations)} conversations and {len(self.folders)} folders")

    def add_conversation(self, conv: ConversationItem):
        """
        Add a single conversation to the sidebar.

        Args:
            conv: Conversation item to add
        """
        if conv.session_id not in [c.session_id for c in self.conversations]:
            self.conversations.append(conv)
            self.refresh()
            logger.debug(f"Added conversation: {conv.title}")

    def add_folder(self, folder: FolderItem):
        """
        Add a folder to the sidebar.

        Args:
            folder: Folder item to add
        """
        if folder.folder_id not in [f.folder_id for f in self.folders]:
            self.folders.append(folder)
            self.refresh()
            logger.debug(f"Added folder: {folder.name}")

    def select_conversation(self, session_id: str):
        """
        Select a conversation by ID.

        Args:
            session_id: Session ID to select
        """
        if self.selected_session_id == session_id:
            return

        # Deselect previous
        if self.selected_session_id and self.selected_session_id in self._conversation_widgets:
            self._update_conversation_widget_state(self.selected_session_id, selected=False)

        # Select new
        self.selected_session_id = session_id
        if session_id in self._conversation_widgets:
            self._update_conversation_widget_state(session_id, selected=True)

        logger.debug(f"Selected conversation: {session_id}")

    def refresh(self):
        """Refresh the conversation list display."""
        # Clear existing widgets
        for widget in self.conversation_list.winfo_children():
            widget.destroy()

        self._conversation_widgets.clear()
        self._folder_widgets.clear()

        # Filter conversations based on search
        filtered_convs = self._filter_conversations(self.search_query)

        # Separate pinned and regular conversations
        pinned_convs = [c for c in filtered_convs if c.pinned]
        regular_convs = [c for c in filtered_convs if not c.pinned]

        row = 0

        # Render pinned section
        if pinned_convs:
            row = self._render_section_header("PINNED", row)
            row = self._render_conversations(pinned_convs, row, indent=0)
            row = self._render_separator(row)

        # Render folders and conversations
        row = self._render_folder_tree(regular_convs, row)

        logger.debug(f"Refreshed sidebar with {len(filtered_convs)} conversations")

    def _filter_conversations(self, query: str) -> List[ConversationItem]:
        """
        Filter conversations based on search query.

        Args:
            query: Search query string

        Returns:
            Filtered list of conversations
        """
        if not query:
            return sorted(self.conversations, key=lambda c: c.last_active, reverse=True)

        query_lower = query.lower()
        filtered = [
            c for c in self.conversations
            if query_lower in c.title.lower()
        ]
        return sorted(filtered, key=lambda c: c.last_active, reverse=True)

    def _render_section_header(self, text: str, row: int) -> int:
        """
        Render a section header.

        Args:
            text: Header text
            row: Current grid row

        Returns:
            Next available row
        """
        header = ctk.CTkLabel(
            self.conversation_list,
            text=text,
            font=ctk.CTkFont(size=FONT_SMALL, weight="bold"),
            text_color=self.theme_manager.get_color("fg_muted"),
            anchor="w"
        )
        header.grid(row=row, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        return row + 1

    def _render_separator(self, row: int) -> int:
        """
        Render a visual separator.

        Args:
            row: Current grid row

        Returns:
            Next available row
        """
        separator = ctk.CTkFrame(
            self.conversation_list,
            height=1,
            fg_color=self.theme_manager.get_color("border")
        )
        separator.grid(row=row, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_SM)
        return row + 1

    def _render_folder_tree(self, conversations: List[ConversationItem], row: int) -> int:
        """
        Render the folder tree with nested conversations.

        Args:
            conversations: List of conversations to render
            row: Current grid row

        Returns:
            Next available row
        """
        # Get root folders (no parent)
        root_folders = [f for f in self.folders if f.parent_id is None]

        # Render root folders and their children
        for folder in root_folders:
            row = self._render_folder_recursive(folder, conversations, row, indent=0)

        # Render conversations not in any folder
        orphan_convs = [c for c in conversations if c.folder_id is None]
        row = self._render_conversations(orphan_convs, row, indent=0)

        return row

    def _render_folder_recursive(
        self,
        folder: FolderItem,
        conversations: List[ConversationItem],
        row: int,
        indent: int
    ) -> int:
        """
        Recursively render a folder and its contents.

        Args:
            folder: Folder to render
            conversations: All conversations (filtered)
            row: Current grid row
            indent: Indentation level

        Returns:
            Next available row
        """
        # Render folder header
        folder_widget = self._create_folder_widget(folder, indent)
        folder_widget.grid(row=row, column=0, sticky="ew", padx=(SPACE_SM + indent * 15, SPACE_SM), pady=(0, SPACE_XS))
        self._folder_widgets[folder.folder_id] = folder_widget
        row += 1

        # Render contents if expanded
        if folder.expanded:
            # Get conversations in this folder
            folder_convs = [c for c in conversations if c.folder_id == folder.folder_id]
            row = self._render_conversations(folder_convs, row, indent + 1)

            # Get child folders
            child_folders = [f for f in self.folders if f.parent_id == folder.folder_id]
            for child in child_folders:
                row = self._render_folder_recursive(child, conversations, row, indent + 1)

        return row

    def _render_conversations(
        self,
        conversations: List[ConversationItem],
        row: int,
        indent: int
    ) -> int:
        """
        Render a list of conversations.

        Args:
            conversations: Conversations to render
            row: Current grid row
            indent: Indentation level

        Returns:
            Next available row
        """
        for conv in conversations:
            conv_widget = self._create_conversation_widget(conv, indent)
            conv_widget.grid(row=row, column=0, sticky="ew", padx=(SPACE_SM + indent * 15, SPACE_SM), pady=(0, SPACE_XS))
            self._conversation_widgets[conv.session_id] = conv_widget
            row += 1

        return row

    def _create_folder_widget(self, folder: FolderItem, indent: int) -> ctk.CTkFrame:
        """
        Create a folder widget.

        Args:
            folder: Folder item
            indent: Indentation level

        Returns:
            CTkFrame containing folder UI
        """
        colors = self.theme_manager.colors

        widget = ctk.CTkFrame(
            self.conversation_list,
            fg_color="transparent",
            corner_radius=RADIUS_SM
        )
        widget.grid_columnconfigure(1, weight=1)

        # Expand/collapse button
        icon = "▼" if folder.expanded else "▶"
        toggle_btn = ctk.CTkButton(
            widget,
            text=icon,
            width=20,
            height=20,
            font=ctk.CTkFont(size=FONT_SMALL),
            fg_color="transparent",
            hover_color=colors["bg_hover"],
            command=lambda: self._toggle_folder(folder.folder_id)
        )
        toggle_btn.grid(row=0, column=0, padx=(SPACE_XS, 0))

        # Folder icon and name
        name_label = ctk.CTkLabel(
            widget,
            text=f"📁 {folder.name}",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color=colors["fg_primary"],
            anchor="w"
        )
        name_label.grid(row=0, column=1, sticky="w", padx=SPACE_XS)

        # Right-click context menu
        widget.bind("<Button-3>", lambda e: self._show_folder_context_menu(e, folder))
        name_label.bind("<Button-3>", lambda e: self._show_folder_context_menu(e, folder))

        # Hover effect
        widget.bind("<Enter>", lambda e: widget.configure(fg_color=colors["bg_hover"]))
        widget.bind("<Leave>", lambda e: widget.configure(fg_color="transparent"))

        return widget

    def _create_conversation_widget(self, conv: ConversationItem, indent: int) -> ctk.CTkFrame:
        """
        Create a conversation widget.

        Args:
            conv: Conversation item
            indent: Indentation level

        Returns:
            CTkFrame containing conversation UI
        """
        colors = self.theme_manager.colors
        is_selected = conv.session_id == self.selected_session_id

        widget = ctk.CTkFrame(
            self.conversation_list,
            fg_color=colors["selection"] if is_selected else "transparent",
            corner_radius=RADIUS_SM
        )
        widget.grid_columnconfigure(0, weight=1)

        # Main container with padding
        content_frame = ctk.CTkFrame(widget, fg_color="transparent")
        content_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_XS, pady=SPACE_XS)
        content_frame.grid_columnconfigure(0, weight=1)

        # Title with pin indicator
        title_text = f"📌 {conv.title}" if conv.pinned else conv.title
        title_label = ctk.CTkLabel(
            content_frame,
            text=title_text,
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color="#FFFFFF" if is_selected else colors["fg_primary"],
            anchor="w"
        )
        title_label.grid(row=0, column=0, sticky="w")

        # Metadata (message count, last active)
        time_str = self._format_time(conv.last_active)
        meta_text = f"{conv.message_count} msg · {time_str}"
        meta_label = ctk.CTkLabel(
            content_frame,
            text=meta_text,
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color="#FFFFFF" if is_selected else colors["fg_muted"],
            anchor="w"
        )
        meta_label.grid(row=1, column=0, sticky="w")

        # Click to select
        widget.bind("<Button-1>", lambda e: self._handle_conversation_click(conv.session_id))
        content_frame.bind("<Button-1>", lambda e: self._handle_conversation_click(conv.session_id))
        title_label.bind("<Button-1>", lambda e: self._handle_conversation_click(conv.session_id))
        meta_label.bind("<Button-1>", lambda e: self._handle_conversation_click(conv.session_id))

        # Right-click context menu
        widget.bind("<Button-3>", lambda e: self._show_conversation_context_menu(e, conv))
        content_frame.bind("<Button-3>", lambda e: self._show_conversation_context_menu(e, conv))
        title_label.bind("<Button-3>", lambda e: self._show_conversation_context_menu(e, conv))
        meta_label.bind("<Button-3>", lambda e: self._show_conversation_context_menu(e, conv))

        # Hover effect (only if not selected)
        if not is_selected:
            widget.bind("<Enter>", lambda e: widget.configure(fg_color=colors["bg_hover"]))
            widget.bind("<Leave>", lambda e: widget.configure(fg_color="transparent"))

        return widget

    def _update_conversation_widget_state(self, session_id: str, selected: bool):
        """
        Update the visual state of a conversation widget.

        Args:
            session_id: Session ID
            selected: Whether the conversation is selected
        """
        if session_id not in self._conversation_widgets:
            return

        widget = self._conversation_widgets[session_id]
        colors = self.theme_manager.colors

        if selected:
            widget.configure(fg_color=colors["selection"])
            # Update text colors to white
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkFrame):
                    for label in child.winfo_children():
                        if isinstance(label, ctk.CTkLabel):
                            label.configure(text_color="#FFFFFF")
        else:
            widget.configure(fg_color="transparent")
            # Restore original text colors
            for child in widget.winfo_children():
                if isinstance(child, ctk.CTkFrame):
                    labels = [w for w in child.winfo_children() if isinstance(w, ctk.CTkLabel)]
                    if len(labels) >= 2:
                        labels[0].configure(text_color=colors["fg_primary"])  # Title
                        labels[1].configure(text_color=colors["fg_muted"])    # Meta

    def _toggle_folder(self, folder_id: str):
        """
        Toggle folder expanded state.

        Args:
            folder_id: Folder ID to toggle
        """
        for folder in self.folders:
            if folder.folder_id == folder_id:
                folder.expanded = not folder.expanded
                self.refresh()
                logger.debug(f"Toggled folder {folder.name}: expanded={folder.expanded}")
                break

    def _format_time(self, dt: datetime) -> str:
        """
        Format datetime for display.

        Args:
            dt: Datetime to format

        Returns:
            Formatted string (e.g., "5m ago", "2h ago", "Jan 15")
        """
        now = datetime.now()
        diff = now - dt

        if diff.days == 0:
            hours = diff.seconds // 3600
            minutes = (diff.seconds % 3600) // 60
            if hours == 0:
                if minutes == 0:
                    return "just now"
                return f"{minutes}m ago"
            return f"{hours}h ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days}d ago"
        else:
            return dt.strftime("%b %d")

    def _handle_conversation_click(self, session_id: str):
        """
        Handle conversation click.

        Args:
            session_id: Session ID that was clicked
        """
        self.select_conversation(session_id)
        if self.on_select:
            self.on_select(session_id)

    def _handle_new_chat(self):
        """Handle new chat button click."""
        if self.on_new_chat:
            self.on_new_chat()
        logger.debug("New chat requested")

    def _handle_new_folder(self):
        """Handle new folder button click."""
        if self.on_new_folder:
            self.on_new_folder()
        logger.debug("New folder requested")

    def _handle_search(self, query: str):
        """
        Handle search query change.

        Args:
            query: Search query string
        """
        self.search_query = query
        self.refresh()
        logger.debug(f"Search query: {query}")

    def _handle_search_clear(self):
        """Handle search clear."""
        self.search_query = ""
        self.refresh()

    def _show_conversation_context_menu(self, event, conv: ConversationItem):
        """
        Show context menu for a conversation.

        Args:
            event: Click event
            conv: Conversation item
        """
        self._dismiss_context_menu()

        menu = ctk.CTkToplevel(self)
        menu.withdraw()
        menu.overrideredirect(True)
        menu.configure(fg_color=self.theme_manager.get_color("bg_secondary"))

        colors = self.theme_manager.colors

        # Menu items
        items = [
            ("Rename", lambda: self._handle_rename(conv.session_id)),
            ("Duplicate", lambda: self._handle_duplicate(conv.session_id)),
            ("Pin" if not conv.pinned else "Unpin", lambda: self._handle_pin(conv.session_id)),
            ("Move to folder", lambda: self._handle_move(conv.session_id)),
            ("separator", None),
            ("Delete", lambda: self._handle_delete(conv.session_id)),
        ]

        for item_text, item_command in items:
            if item_text == "separator":
                sep = ctk.CTkFrame(menu, height=1, fg_color=colors["border"])
                sep.pack(fill="x", padx=SPACE_XS, pady=SPACE_XS)
            else:
                btn = ctk.CTkButton(
                    menu,
                    text=item_text,
                    font=ctk.CTkFont(size=FONT_CAPTION),
                    fg_color="transparent",
                    hover_color=colors["bg_hover"],
                    anchor="w",
                    command=lambda cmd=item_command: self._execute_menu_command(cmd)
                )
                btn.pack(fill="x", padx=SPACE_XS, pady=SPACE_XS)

        # Position and show menu
        menu.update_idletasks()
        x = event.x_root
        y = event.y_root
        menu.geometry(f"+{x}+{y}")
        menu.deiconify()

        # Bind to dismiss on click outside
        menu.bind("<FocusOut>", lambda e: self._dismiss_context_menu())

        self._context_menu = menu

    def _show_folder_context_menu(self, event, folder: FolderItem):
        """
        Show context menu for a folder.

        Args:
            event: Click event
            folder: Folder item
        """
        self._dismiss_context_menu()

        menu = ctk.CTkToplevel(self)
        menu.withdraw()
        menu.overrideredirect(True)
        menu.configure(fg_color=self.theme_manager.get_color("bg_secondary"))

        colors = self.theme_manager.colors

        # Menu items
        items = [
            ("Rename", lambda: self._handle_rename(folder.folder_id)),
            ("separator", None),
            ("Delete", lambda: self._handle_delete(folder.folder_id)),
        ]

        for item_text, item_command in items:
            if item_text == "separator":
                sep = ctk.CTkFrame(menu, height=1, fg_color=colors["border"])
                sep.pack(fill="x", padx=SPACE_XS, pady=SPACE_XS)
            else:
                btn = ctk.CTkButton(
                    menu,
                    text=item_text,
                    font=ctk.CTkFont(size=FONT_CAPTION),
                    fg_color="transparent",
                    hover_color=colors["bg_hover"],
                    anchor="w",
                    command=lambda cmd=item_command: self._execute_menu_command(cmd)
                )
                btn.pack(fill="x", padx=SPACE_XS, pady=SPACE_XS)

        # Position and show menu
        menu.update_idletasks()
        x = event.x_root
        y = event.y_root
        menu.geometry(f"+{x}+{y}")
        menu.deiconify()

        # Bind to dismiss on click outside
        menu.bind("<FocusOut>", lambda e: self._dismiss_context_menu())

        self._context_menu = menu

    def _execute_menu_command(self, command: Callable):
        """
        Execute a context menu command and dismiss the menu.

        Args:
            command: Command function to execute
        """
        self._dismiss_context_menu()
        if command:
            command()

    def _dismiss_context_menu(self):
        """Dismiss the active context menu."""
        if self._context_menu:
            try:
                self._context_menu.destroy()
            except Exception:
                pass
            self._context_menu = None

    def _handle_rename(self, item_id: str):
        """
        Handle rename action.

        Args:
            item_id: Session or folder ID to rename
        """
        if self.on_rename:
            # In a real implementation, this would show a dialog
            # For now, just pass to callback
            self.on_rename(item_id, "New Name")
        logger.debug(f"Rename requested for: {item_id}")

    def _handle_delete(self, item_id: str):
        """
        Handle delete action.

        Args:
            item_id: Session or folder ID to delete
        """
        if self.on_delete:
            self.on_delete(item_id)
        logger.debug(f"Delete requested for: {item_id}")

    def _handle_duplicate(self, session_id: str):
        """
        Handle duplicate action.

        Args:
            session_id: Session ID to duplicate
        """
        if self.on_duplicate:
            self.on_duplicate(session_id)
        logger.debug(f"Duplicate requested for: {session_id}")

    def _handle_move(self, session_id: str):
        """
        Handle move to folder action.

        Args:
            session_id: Session ID to move
        """
        if self.on_move:
            # In a real implementation, this would show a folder picker dialog
            # For now, just pass to callback with None
            self.on_move(session_id, None)
        logger.debug(f"Move requested for: {session_id}")

    def _handle_pin(self, session_id: str):
        """
        Handle pin/unpin action.

        Args:
            session_id: Session ID to pin/unpin
        """
        if self.on_pin:
            self.on_pin(session_id)

        # Update local state
        for conv in self.conversations:
            if conv.session_id == session_id:
                conv.pinned = not conv.pinned
                self.refresh()
                logger.debug(f"Pin toggled for: {session_id}, pinned={conv.pinned}")
                break

    def _on_theme_change(self, mode: str):
        """
        Handle theme change.

        Args:
            mode: New theme mode
        """
        # Refresh to apply new theme colors
        self.refresh()
        logger.debug(f"Theme changed to {mode}, sidebar refreshed")

    def destroy(self):
        """Cleanup when destroyed."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
            self._dismiss_context_menu()
        except Exception:
            pass
        super().destroy()


# Alias for backward compatibility
Sidebar = ConversationSidebar
