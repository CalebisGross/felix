import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import json
from .utils import ThreadManager, DBHelper, logger

try:
    from src.memory import knowledge_store, task_memory
except ImportError as e:
    logger.error(f"Failed to import memory modules: {e}")
    knowledge_store = None
    task_memory = None

class MemoryFrame(ttk.Frame):
    def __init__(self, parent, thread_manager, db_helper):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.db_helper = db_helper

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Memory tab
        memory_frame = MemorySubFrame(self.notebook, self.thread_manager, self.db_helper, 'felix_memory.db', 'tasks')
        self.notebook.add(memory_frame, text="Memory")

        # Knowledge tab
        knowledge_frame = MemorySubFrame(self.notebook, self.thread_manager, self.db_helper, 'felix_knowledge.db', 'knowledge')
        self.notebook.add(knowledge_frame, text="Knowledge")

class MemorySubFrame(ttk.Frame):
    def __init__(self, parent, thread_manager, db_helper, db_name, table_name):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.db_helper = db_helper
        self.db_name = db_name
        self.table_name = table_name
        self.entries = []  # list of (id, content)

        # Listbox with scrollbar
        self.listbox = tk.Listbox(self, height=10)
        scrollbar = ttk.Scrollbar(self, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=scrollbar.set)
        self.listbox.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        scrollbar.grid(row=0, column=1, sticky='ns')

        self.listbox.bind('<<ListboxSelect>>', self.on_select)

        # Query entry and search button
        ttk.Label(self, text="Query:").grid(row=1, column=0, sticky='w', padx=5)
        self.query_entry = ttk.Entry(self)
        self.query_entry.grid(row=1, column=1, sticky='ew', padx=5)
        self.search_button = ttk.Button(self, text="Search", command=self.search)
        self.search_button.grid(row=1, column=2, padx=5)

        # View text with scrollbar
        ttk.Label(self, text="Details:").grid(row=2, column=0, sticky='w', padx=5)
        self.view_text = tk.Text(self, height=5, wrap=tk.WORD, state='disabled')
        view_scrollbar = ttk.Scrollbar(self, command=self.view_text.yview)
        self.view_text.config(yscrollcommand=view_scrollbar.set)
        self.view_text.grid(row=3, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        view_scrollbar.grid(row=3, column=2, sticky='ns')

        # Edit entry
        ttk.Label(self, text="Edit:").grid(row=4, column=0, sticky='w', padx=5)
        self.edit_entry = ttk.Entry(self)
        self.edit_entry.grid(row=4, column=1, columnspan=2, sticky='ew', padx=5)

        # Update and Delete buttons
        self.update_button = ttk.Button(self, text="Update", command=self.update_entry)
        self.update_button.grid(row=5, column=0, padx=5, pady=5)
        self.delete_button = ttk.Button(self, text="Delete", command=self.delete_entry)
        self.delete_button.grid(row=5, column=1, padx=5, pady=5)

        # Grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Load initial entries
        self.load_entries()

    def load_entries(self):
        self.thread_manager.start_thread(self._load_entries_thread)

    def _load_entries_thread(self):
        try:
            if self.table_name == 'knowledge' and knowledge_store:
                # Use KnowledgeStore API properly
                ks = knowledge_store.KnowledgeStore(self.db_name)
                query = knowledge_store.KnowledgeQuery(limit=100)
                entries = ks.retrieve_knowledge(query)
                # Format: (id, display_string)
                self.entries = []
                for e in entries:
                    # Create readable display from knowledge entry
                    content_str = json.dumps(e.content) if isinstance(e.content, dict) else str(e.content)
                    display = f"{e.domain}: {content_str[:100]}..."
                    self.entries.append((e.knowledge_id, content_str, display))
            elif self.table_name == 'tasks' and task_memory:
                # Use TaskMemory API properly
                tm = task_memory.TaskMemory(self.db_name)
                patterns = tm.get_patterns(task_memory.TaskMemoryQuery(limit=100))
                # Format: (id, full_content, display_string)
                self.entries = []
                for p in patterns:
                    keywords_str = ", ".join(p.keywords) if p.keywords else "No keywords"
                    full_content = json.dumps({
                        "keywords": p.keywords,
                        "task_type": p.task_type,
                        "success_count": p.success_count,
                        "avg_duration": p.avg_duration
                    })
                    display = f"{p.task_type}: {keywords_str}"
                    self.entries.append((p.pattern_id, full_content, display))
            else:
                # Fallback: try generic SQL query
                self.entries = []
                logger.warning(f"Memory modules not available, trying direct SQL query")

            self.after(0, self._update_listbox)
            # Show message if no entries
            if not self.entries:
                self.after(0, lambda: self._show_no_entries_message())
        except Exception as e:
            logger.error(f"Error loading entries: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to load entries: {error_msg}"))

    def _update_listbox(self):
        self.listbox.delete(0, tk.END)
        for entry in self.entries:
            if len(entry) == 3:
                id_, content, display = entry
                self.listbox.insert(tk.END, f"{id_}: {display}")
            elif len(entry) == 2:
                # Fallback for old 2-tuple format
                id_, content = entry
                snippet = content[:50] + '...' if len(content) > 50 else content
                self.listbox.insert(tk.END, f"{id_}: {snippet}")

    def _show_no_entries_message(self):
        self.view_text.config(state='normal')
        self.view_text.delete(1.0, tk.END)
        self.view_text.insert(tk.END, "No entries found. Add data via workflows or manually.")
        self.view_text.config(state='disabled')

    def search(self):
        query = self.query_entry.get().strip()
        if not query:
            self.load_entries()
            return
        self.thread_manager.start_thread(self._search_thread, args=(query,))

    def _search_thread(self, query):
        try:
            if self.table_name == 'knowledge' and knowledge_store:
                ks = knowledge_store.KnowledgeStore(self.db_name)
                query_obj = knowledge_store.KnowledgeQuery(content_keywords=[query], limit=100)
                entries = ks.retrieve_knowledge(query_obj)
                self.entries = []
                for e in entries:
                    content_str = json.dumps(e.content) if isinstance(e.content, dict) else str(e.content)
                    display = f"{e.domain}: {content_str[:100]}..."
                    self.entries.append((e.knowledge_id, content_str, display))
            elif self.table_name == 'tasks' and task_memory:
                tm = task_memory.TaskMemory(self.db_name)
                query_obj = task_memory.TaskMemoryQuery(keywords=[query], limit=100)
                patterns = tm.get_patterns(query_obj)
                self.entries = []
                for p in patterns:
                    keywords_str = ", ".join(p.keywords) if p.keywords else "No keywords"
                    full_content = json.dumps({
                        "keywords": p.keywords,
                        "task_type": p.task_type,
                        "success_count": p.success_count,
                        "avg_duration": p.avg_duration
                    })
                    display = f"{p.task_type}: {keywords_str}"
                    self.entries.append((p.pattern_id, full_content, display))
            else:
                self.entries = []
                logger.warning("Search not available without memory modules")

            self.after(0, self._update_listbox)
        except Exception as e:
            logger.error(f"Error searching: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to search: {error_msg}"))

    def on_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        index = selection[0]
        entry = self.entries[index]

        # Handle both 3-tuple and 2-tuple formats
        if len(entry) == 3:
            id_, content, display = entry
        elif len(entry) == 2:
            id_, content = entry
        else:
            return

        # Display full content in view text
        self.view_text.config(state='normal')
        self.view_text.delete(1.0, tk.END)
        # Pretty print JSON if possible
        try:
            parsed = json.loads(content)
            pretty_content = json.dumps(parsed, indent=2)
            self.view_text.insert(tk.END, pretty_content)
        except:
            self.view_text.insert(tk.END, content)
        self.view_text.config(state='disabled')

        # For editing, use the content
        self.edit_entry.delete(0, tk.END)
        self.edit_entry.insert(0, content)

    def update_entry(self):
        messagebox.showinfo("Not Supported",
                          "Direct editing of memory entries is not currently supported.\n\n"
                          "Memory entries are managed by the Felix memory systems and should be "
                          "modified through the appropriate APIs (KnowledgeStore or TaskMemory).")

    def delete_entry(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select an entry to delete.")
            return

        # Warn about deletion
        result = messagebox.askyesno("Confirm Deletion",
                                    "Are you sure you want to delete this entry?\n\n"
                                    "This will remove it from the Felix memory system.",
                                    icon='warning')
        if not result:
            return

        index = selection[0]
        entry = self.entries[index]

        # Get ID from entry tuple
        if len(entry) >= 2:
            id_ = entry[0]
        else:
            messagebox.showerror("Error", "Invalid entry format")
            return

        self.thread_manager.start_thread(self._delete_thread, args=(id_,))

    def _delete_thread(self, id_):
        try:
            if self.table_name == 'knowledge' and knowledge_store:
                ks = knowledge_store.KnowledgeStore(self.db_name)
                # KnowledgeStore doesn't have direct delete, use SQL
                import sqlite3
                conn = sqlite3.connect(self.db_name)
                conn.execute("DELETE FROM knowledge WHERE knowledge_id = ?", (id_,))
                conn.commit()
                conn.close()
                self.after(0, lambda: messagebox.showinfo("Success", "Knowledge entry deleted."))
            elif self.table_name == 'tasks' and task_memory:
                # TaskMemory doesn't have direct delete, use SQL
                import sqlite3
                conn = sqlite3.connect(self.db_name)
                conn.execute("DELETE FROM task_patterns WHERE pattern_id = ?", (id_,))
                conn.commit()
                conn.close()
                self.after(0, lambda: messagebox.showinfo("Success", "Task pattern deleted."))
            else:
                self.after(0, lambda: messagebox.showerror("Error", "Memory modules not available"))

            self.after(0, self.load_entries)
        except Exception as e:
            logger.error(f"Error deleting: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to delete: {error_msg}"))