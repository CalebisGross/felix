# Streamlit GUI Development Notes

**Purpose**: Developer reference for implementation patterns and integration with Caleb's core development direction

## Implementation Guidelines

### Read-Only Operations
- Use SELECT queries only (no INSERT/UPDATE/DELETE)
- Use sqlite3 read-only mode: `sqlite3.connect("file:path?mode=ro", uri=True)`
- No file writes except user-initiated downloads

### Error Handling
- Return empty DataFrames for missing databases
- Display user-friendly info messages for missing data
- Show error details in expandable sections

### Performance Optimization
- Cache database readers with `@st.cache_resource`
- Cache query results with `@st.cache_data(ttl=60)`
- Use query limits (100-200 rows default)
- Lazy load detailed data in expandable sections

## Caleb's Development Direction & Integration Notes

**Note:** This Streamlit GUI provides monitoring and visualization support for Caleb's core development. The features below represent his current development direction on the awareness branch, which will eventually need visualization support in this GUI.

### Active Development Areas (Last 7 Commits on Awareness Branch)

#### 1. System Autonomy Infrastructure (commits 35fda46, 23abe06)
**What Caleb Built:**
- Agent-initiated system command execution with safety controls
- Three-tier trust system: SAFE/REVIEW/BLOCKED (137 regex patterns)
- SystemExecutor with timeouts, output limits, process cleanup
- TrustManager with approval workflow and risk scoring
- Command history tracking in `felix_system_actions.db`
- SystemAgent specialized for system operations (temperature 0.1-0.4)
- Virtual environment detection and activation support

**Performance:**
- Command execution overhead: ~2-5ms
- Trust classification: <1ms
- Safe commands auto-execute, risky commands require approval

**Potential GUI Support Needed:**
- System command history viewer (read-only)
- Trust classification statistics dashboard
- Approval workflow status monitor
- Virtual environment state display

#### 2. Database Migration Framework (commit c86af34)
**What Caleb Built:**
- Versioned migration system with rollback support
- Automatic timestamped backups with integrity verification
- Composite indexes (10-40x query performance improvements)
- Full-text search (FTS5) across knowledge, workflows, commands
- New databases: `agent_performance.db`, `felix_system_actions.db`

**Performance Impact:**
- Knowledge search: 500-1000ms → 10-50ms (10-20x faster)
- Pattern matching: 200-500ms → 5-20ms (10-40x faster)
- Workflow browsing: 50-100ms → 10-30ms (2-5x faster)

**Potential GUI Support Needed:**
- Database health dashboard
- Migration history viewer
- Backup status monitor
- Performance metrics comparison (before/after indexes)

#### 3. Prompt Management System (commit 31559b7)
**What Caleb Built:**
- Hybrid YAML+Database prompt storage
- Default prompts in version-controlled `config/prompts.yaml`
- Custom prompt overrides in SQLite database
- GUI editor for viewing, editing, versioning prompts
- Conversation continuity support (multi-turn threading)
- Runtime caching for performance

**Potential GUI Support Needed:**
- Prompt version history viewer
- Prompt effectiveness analytics (which prompts lead to best results)
- Custom vs default prompt comparison
- Conversation thread visualization

#### 4. Core System Enhancements (commits 7d96fc9, ec8b56b)
**What Caleb Built:**
- PromptManager integration for position-aware agent behavior
- Enhanced specialized agents with adaptive prompts
- Advanced truth assessment framework (364+ lines)
- Improved context building for collaborative agents
- Updated documentation (agent capacity: 50 agents instead of 133)

**Potential GUI Support Needed:**
- Position-aware prompt effectiveness dashboard
- Truth assessment results visualization
- Context building workflow display

### Integration Philosophy

**Our Role:** This Streamlit GUI is a **support tool** for Caleb's core development work. We provide:
- Read-only monitoring and visualization
- Performance analytics to inform his decisions
- Historical data views to understand patterns
- Hypothesis validation tracking

**Caleb's Role:** He steers the project direction, makes architectural decisions, and implements core features. We follow his lead and provide visualization support for his innovations.

### When to Add GUI Support

New visualization features should be added when:
1. Caleb requests specific monitoring/analytics
2. A new database table is created (add read-only viewer)
3. Performance metrics need tracking
4. Debugging or analysis would benefit from visual tools

### Integration with External Tools

- **Prometheus/Grafana:** Export metrics for external monitoring
- **Jupyter Notebooks:** Data export for deep analysis
- **GitHub Actions:** Automated testing and deployment
- **Docker:** Containerized deployment

## Code Quality Standards

### Docstring Guidelines
- Module headers: 1 line maximum
- Class docstrings: 1-2 lines maximum
- Function docstrings: Brief with essential Args/Returns
- No multi-paragraph explanations
- Keep inline comments (helpful for logic)

### File Structure
- Backend: Data access and business logic
- Components: Reusable UI elements
- Pages: Main interface screens
- Tests: Component and integration tests

### Best Practices
- Use type hints for all function parameters
- Implement graceful degradation for missing data
- Follow Streamlit's caching patterns
- Maintain separation from `src/` directory
- 100% Python (no CSS/JS modifications)

---

**Last Updated**: 2025-10-26
**For Questions**: See INTEGRATION_SUMMARY.md or CLAUDE.md
