"""Dialog windows for Felix GUI."""

from .settings_dialog import SettingsDialog
from .developer_views import AgentsDialog, MemoryDialog, PromptsDialog

__all__ = [
    "SettingsDialog",
    "AgentsDialog",
    "MemoryDialog",
    "PromptsDialog",
]
