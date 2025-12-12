"""
Knowledge Brain Tab for Felix GUI (CustomTkinter Edition).

Main tab that assembles all Knowledge Brain panels into a tabbed interface:
- Control & Processing: Daemon control, queue status, statistics, activity
- Knowledge Base: Documents and Concepts browser
- Relationships: Knowledge graph exploration
- Maintenance: Quality, Audit, and Cleanup tools
"""

import customtkinter as ctk
import logging

from .control_panel import ControlPanel
from .knowledge_base_panel import KnowledgeBasePanel
from .relationships_panel import RelationshipsPanel
from .maintenance_panel import MaintenancePanel
from ...styles import SPACE_XS

logger = logging.getLogger("felix_gui_ctk")


class KnowledgeBrainTab(ctk.CTkFrame):
    """
    Knowledge Brain monitoring and control interface.
    
    Provides comprehensive tools for managing Felix's autonomous
    knowledge brain system including document ingestion, concept
    extraction, relationship management, and maintenance.
    """

    def __init__(self, master, thread_manager, main_app=None, **kwargs):
        """
        Initialize Knowledge Brain Tab.
        
        Args:
            master: Parent widget
            thread_manager: ThreadManager for background operations
            main_app: Reference to main FelixApp
            **kwargs: Additional arguments for CTkFrame
        """
        super().__init__(master, fg_color="transparent", **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self._layout_manager = None

        # Knowledge brain component references
        self.knowledge_daemon = None
        self.knowledge_retriever = None
        self.knowledge_store = None

        # Set up UI
        self._setup_ui()

        logger.info("Knowledge Brain tab initialized")

    def set_layout_manager(self, layout_manager):
        """Set the layout manager (interface compliance)."""
        self._layout_manager = layout_manager

    def _setup_ui(self):
        """Set up the main UI layout with tabbed interface."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create tabview for sub-sections
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=SPACE_XS, pady=SPACE_XS)
        
        # Tab 1: Control & Processing
        self.tabview.add("⚙️ Control")
        self.control_panel = ControlPanel(
            self.tabview.tab("⚙️ Control"),
            self.thread_manager,
            main_app=self.main_app
        )
        self.control_panel.pack(fill="both", expand=True)
        
        # Tab 2: Knowledge Base (Documents + Concepts)
        self.tabview.add("📚 Knowledge Base")
        self.knowledge_base_panel = KnowledgeBasePanel(
            self.tabview.tab("📚 Knowledge Base"),
            self.thread_manager,
            main_app=self.main_app
        )
        self.knowledge_base_panel.pack(fill="both", expand=True)
        
        # Tab 3: Relationships
        self.tabview.add("🔗 Relationships")
        self.relationships_panel = RelationshipsPanel(
            self.tabview.tab("🔗 Relationships"),
            self.thread_manager,
            main_app=self.main_app
        )
        self.relationships_panel.pack(fill="both", expand=True)
        
        # Tab 4: Maintenance
        self.tabview.add("🛠️ Maintenance")
        self.maintenance_panel = MaintenancePanel(
            self.tabview.tab("🛠️ Maintenance"),
            self.thread_manager,
            main_app=self.main_app
        )
        self.maintenance_panel.pack(fill="both", expand=True)

    def _enable_features(self):
        """Enable features when Felix system starts."""
        if self.main_app and self.main_app.felix_system:
            # Wire up references to knowledge brain components
            self.knowledge_store = self.main_app.felix_system.knowledge_store
            self.knowledge_retriever = getattr(
                self.main_app.felix_system, 'knowledge_retriever', None
            )
            self.knowledge_daemon = getattr(
                self.main_app.felix_system, 'knowledge_daemon', None
            )
            
            # Pass references to all panels
            self._set_panel_references()
            
            # Enable each panel
            if hasattr(self.control_panel, '_enable_features'):
                self.control_panel._enable_features()
            if hasattr(self.knowledge_base_panel, '_enable_features'):
                self.knowledge_base_panel._enable_features()
            if hasattr(self.relationships_panel, '_enable_features'):
                self.relationships_panel._enable_features()
            if hasattr(self.maintenance_panel, '_enable_features'):
                self.maintenance_panel._enable_features()
            
            logger.info("Knowledge Brain tab features enabled")

    def _disable_features(self):
        """Disable features when Felix system stops."""
        # Disable each panel
        if hasattr(self.control_panel, '_disable_features'):
            self.control_panel._disable_features()
        if hasattr(self.knowledge_base_panel, '_disable_features'):
            self.knowledge_base_panel._disable_features()
        if hasattr(self.relationships_panel, '_disable_features'):
            self.relationships_panel._disable_features()
        if hasattr(self.maintenance_panel, '_disable_features'):
            self.maintenance_panel._disable_features()
        
        # Clear references
        self.knowledge_store = None
        self.knowledge_retriever = None
        self.knowledge_daemon = None
        
        logger.info("Knowledge Brain tab features disabled")

    def _set_panel_references(self):
        """Pass knowledge brain references to all panels."""
        # Control panel
        if hasattr(self.control_panel, 'set_knowledge_refs'):
            self.control_panel.set_knowledge_refs(
                knowledge_store=self.knowledge_store,
                knowledge_retriever=self.knowledge_retriever,
                knowledge_daemon=self.knowledge_daemon
            )
        
        # Knowledge base panel
        if hasattr(self.knowledge_base_panel, 'set_knowledge_refs'):
            self.knowledge_base_panel.set_knowledge_refs(
                knowledge_store=self.knowledge_store,
                knowledge_retriever=self.knowledge_retriever,
                knowledge_daemon=self.knowledge_daemon
            )
        
        # Relationships panel
        if hasattr(self.relationships_panel, 'set_knowledge_refs'):
            self.relationships_panel.set_knowledge_refs(
                knowledge_store=self.knowledge_store,
                knowledge_retriever=self.knowledge_retriever,
                knowledge_daemon=self.knowledge_daemon
            )
        
        # Maintenance panel
        if hasattr(self.maintenance_panel, 'set_knowledge_refs'):
            self.maintenance_panel.set_knowledge_refs(
                knowledge_store=self.knowledge_store,
                knowledge_retriever=self.knowledge_retriever,
                knowledge_daemon=self.knowledge_daemon
            )

    def refresh_all(self):
        """Refresh all panels."""
        if hasattr(self.control_panel, 'refresh'):
            self.control_panel.refresh()
        if hasattr(self.knowledge_base_panel, 'refresh_documents'):
            self.knowledge_base_panel.refresh_documents()
        if hasattr(self.knowledge_base_panel, 'refresh_concepts'):
            self.knowledge_base_panel.refresh_concepts()
        if hasattr(self.relationships_panel, 'refresh'):
            self.relationships_panel.refresh()
        if hasattr(self.maintenance_panel, 'refresh'):
            self.maintenance_panel.refresh()

    def cleanup(self):
        """Clean up resources when tab is destroyed."""
        # Clean up each panel if they have cleanup methods
        panels = [
            self.control_panel,
            self.knowledge_base_panel,
            self.relationships_panel,
            self.maintenance_panel
        ]
        
        for panel in panels:
            if hasattr(panel, 'cleanup'):
                try:
                    panel.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up panel: {e}")
        
        logger.info("Knowledge Brain tab cleaned up")
