"""
System Tool for Felix system status and health checks in conversational CLI.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_tool import BaseTool, ToolResult


class SystemTool(BaseTool):
    """Tool for system status and health monitoring."""

    @property
    def name(self) -> str:
        return "system"

    @property
    def description(self) -> str:
        return "Check Felix system status and health"

    @property
    def usage(self) -> str:
        return "/system [status|providers|databases|knowledge|config]"

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Execute system command.

        Commands:
            /system status          - Show overall system status
            /system providers       - Check LLM provider status
            /system databases       - Check database status
            /system knowledge       - Show knowledge statistics
            /system config          - Show system configuration
        """
        if not args:
            # Default to status
            return self._show_status(kwargs)

        command = args[0]

        if command == "status":
            return self._show_status(kwargs)
        elif command == "providers":
            return self._check_providers(kwargs)
        elif command == "databases":
            return self._check_databases(kwargs)
        elif command == "knowledge":
            return self._show_knowledge_stats(kwargs)
        elif command == "config":
            return self._show_config(kwargs)
        else:
            return self.format_error(f"Unknown system command: {command}")

    def _show_status(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Show overall system status."""
        try:
            output_lines = ["Felix System Status", "=" * 60, ""]

            # Felix system status
            felix_system = self.felix_context.get('felix_system')
            if felix_system:
                if felix_system.running:
                    output_lines.append("✓ Felix system is running")
                else:
                    output_lines.append("✗ Felix system is not running")
            else:
                output_lines.append("✗ Felix system not initialized")

            # LLM providers
            output_lines.append("\nLLM Providers:")
            try:
                from src.llm.router_adapter import create_router_adapter

                adapter = self.felix_context.get('llm_adapter')
                if not adapter:
                    adapter = create_router_adapter('config/llm.yaml')

                router = adapter.router
                results = router.test_all_connections()

                for provider, status in results.items():
                    icon = "✓" if status else "✗"
                    output_lines.append(f"  {icon} {provider}")

                # Show router statistics
                stats = router.get_statistics()
                if stats['total_requests'] > 0:
                    output_lines.append(f"\n  Total requests: {stats['total_requests']}")
                    output_lines.append(f"  Success rate: {stats['overall_success_rate']:.1%}")

            except Exception as e:
                output_lines.append(f"  ✗ Could not check providers: {e}")

            # Databases
            output_lines.append("\nDatabases:")
            db_files = [
                "felix_knowledge.db",
                "felix_workflow_history.db",
                "felix_memory.db",
                "felix_task_memory.db",
                "felix_cli_sessions.db"
            ]

            for db_file in db_files:
                if Path(db_file).exists():
                    size = Path(db_file).stat().st_size
                    size_mb = size / (1024 * 1024)
                    output_lines.append(f"  ✓ {db_file} ({size_mb:.1f} MB)")
                else:
                    output_lines.append(f"  ✗ {db_file} (not found)")

            # Knowledge stats
            output_lines.append("\nKnowledge:")
            try:
                import sqlite3
                conn = sqlite3.connect("felix_knowledge.db")
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
                entries = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM document_sources")
                docs = cursor.fetchone()[0]

                conn.close()

                output_lines.append(f"  Entries: {entries}")
                output_lines.append(f"  Documents: {docs}")

            except Exception:
                output_lines.append(f"  Could not read knowledge database")

            output_lines.append("")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to get system status: {str(e)}")

    def _check_providers(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Check LLM provider status."""
        try:
            output_lines = ["LLM Provider Status", "=" * 60, ""]

            from src.llm.router_adapter import create_router_adapter

            adapter = self.felix_context.get('llm_adapter')
            if not adapter:
                adapter = create_router_adapter('config/llm.yaml')

            router = adapter.router

            # Primary provider
            primary = router.get_primary_provider()
            output_lines.append(f"Primary provider: {primary.get_provider_name()}")
            output_lines.append("")

            # Test all providers
            output_lines.append("Testing providers...")
            results = router.test_all_connections()

            for provider, status in results.items():
                icon = "✓" if status else "✗"
                output_lines.append(f"  {icon} {provider}")

            # Router statistics
            output_lines.append("")
            stats = router.get_statistics()

            output_lines.append("Router Statistics:")
            output_lines.append(f"  Total requests: {stats['total_requests']}")
            output_lines.append(f"  Successful: {stats['successful_requests']}")
            output_lines.append(f"  Failed: {stats['failed_requests']}")
            output_lines.append(f"  Success rate: {stats['overall_success_rate']:.1%}")

            # Per-provider stats
            if stats['provider_stats']:
                output_lines.append("\nPer-provider statistics:")
                for provider, pstats in stats['provider_stats'].items():
                    output_lines.append(f"  {provider}:")
                    output_lines.append(f"    Requests: {pstats.get('requests', 0)}")
                    output_lines.append(f"    Success rate: {pstats.get('success_rate', 0):.1%}")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to check providers: {str(e)}")

    def _check_databases(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Check database status."""
        try:
            output_lines = ["Database Status", "=" * 60, ""]

            db_files = [
                ("felix_knowledge.db", "Knowledge store"),
                ("felix_workflow_history.db", "Workflow history"),
                ("felix_memory.db", "Memory store"),
                ("felix_task_memory.db", "Task memory"),
                ("felix_system_actions.db", "System actions"),
                ("felix_cli_sessions.db", "CLI sessions")
            ]

            total_size = 0
            found_count = 0

            for db_file, description in db_files:
                path = Path(db_file)
                if path.exists():
                    size = path.stat().st_size
                    size_mb = size / (1024 * 1024)
                    total_size += size
                    found_count += 1

                    output_lines.append(f"✓ {db_file}")
                    output_lines.append(f"  {description}")
                    output_lines.append(f"  Size: {size_mb:.2f} MB")
                    output_lines.append("")
                else:
                    output_lines.append(f"✗ {db_file}")
                    output_lines.append(f"  {description} (not found)")
                    output_lines.append("")

            total_mb = total_size / (1024 * 1024)
            output_lines.append(f"Total: {found_count}/{len(db_files)} databases, {total_mb:.2f} MB")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to check databases: {str(e)}")

    def _show_knowledge_stats(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Show knowledge system statistics."""
        try:
            output_lines = ["Knowledge System Statistics", "=" * 60, ""]

            import sqlite3
            conn = sqlite3.connect("felix_knowledge.db")
            cursor = conn.cursor()

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            total_entries = cursor.fetchone()[0]

            # Entries by domain
            cursor.execute("""
                SELECT domain, COUNT(*) as count
                FROM knowledge_entries
                GROUP BY domain
                ORDER BY count DESC
                LIMIT 10
            """)
            domain_counts = cursor.fetchall()

            # Entries by confidence
            cursor.execute("""
                SELECT confidence_level, COUNT(*) as count
                FROM knowledge_entries
                GROUP BY confidence_level
            """)
            confidence_counts = cursor.fetchall()

            # Documents
            cursor.execute("SELECT COUNT(*) FROM document_sources")
            total_docs = cursor.fetchone()[0]

            cursor.execute("""
                SELECT processing_status, COUNT(*) as count
                FROM document_sources
                GROUP BY processing_status
            """)
            doc_status = cursor.fetchall()

            # Relationships
            try:
                cursor.execute("SELECT COUNT(*) FROM knowledge_relationships")
                total_relationships = cursor.fetchone()[0]
            except:
                total_relationships = 0

            conn.close()

            # Format output
            output_lines.append(f"Total Entries: {total_entries}")
            output_lines.append(f"Total Documents: {total_docs}")
            if total_relationships:
                output_lines.append(f"Total Relationships: {total_relationships}")

            if domain_counts:
                output_lines.append("\nTop Domains:")
                for domain, count in domain_counts:
                    output_lines.append(f"  {domain}: {count}")

            if confidence_counts:
                output_lines.append("\nBy Confidence:")
                for confidence, count in confidence_counts:
                    output_lines.append(f"  {confidence}: {count}")

            if doc_status:
                output_lines.append("\nDocument Status:")
                for status, count in doc_status:
                    output_lines.append(f"  {status}: {count}")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to get knowledge stats: {str(e)}")

    def _show_config(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Show system configuration."""
        try:
            output_lines = ["Felix System Configuration", "=" * 60, ""]

            felix_system = self.felix_context.get('felix_system')
            if not felix_system:
                return self.format_error("Felix system not initialized")

            config = felix_system.config

            output_lines.append("General:")
            output_lines.append(f"  Max agents: {config.max_agents}")
            output_lines.append(f"  Base token budget: {config.base_token_budget}")
            output_lines.append(f"  Streaming enabled: {config.enable_streaming}")

            output_lines.append("\nWorkflow:")
            output_lines.append(f"  Max steps (simple): {config.workflow_max_steps_simple}")
            output_lines.append(f"  Max steps (medium): {config.workflow_max_steps_medium}")
            output_lines.append(f"  Max steps (complex): {config.workflow_max_steps_complex}")
            output_lines.append(f"  Simple threshold: {config.workflow_simple_threshold}")
            output_lines.append(f"  Medium threshold: {config.workflow_medium_threshold}")

            output_lines.append("\nFeatures:")
            output_lines.append(f"  Memory: {config.enable_memory}")
            output_lines.append(f"  Compression: {config.enable_compression}")
            output_lines.append(f"  Knowledge brain: {config.enable_knowledge_brain}")
            output_lines.append(f"  Web search: {config.web_search_enabled}")

            if config.enable_knowledge_brain:
                output_lines.append("\nKnowledge Brain:")
                output_lines.append(f"  Daemon enabled: {config.knowledge_daemon_enabled}")
                output_lines.append(f"  Auto augment: {config.knowledge_auto_augment}")
                output_lines.append(f"  Embedding mode: {config.knowledge_embedding_mode}")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to get config: {str(e)}")
