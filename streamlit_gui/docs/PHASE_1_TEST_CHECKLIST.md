## Testing Checklist

### Manual Testing Required

**Benchmarking (1.1):**
- [ ] Navigate to Benchmarking page - no import errors
- [ ] Page displays "Test suite available and ready" message
- [ ] Can select hypothesis (all/H1/H2/H3)
- [ ] Can set iterations (1-20)
- [ ] Can toggle "Use Real LLM" checkbox
- [ ] "Run Validation Tests" button works
- [ ] Progress messages display during execution
- [ ] Results display with metrics for each hypothesis
- [ ] Box plots render when data available
- [ ] Can download JSON report

**Workflow History (1.2):**
- [ ] Navigate to Dashboard → Workflow History tab
- [ ] Summary metrics display (or "No workflows" message)
- [ ] Can filter by status (all/completed/failed)
- [ ] Can set days back (1-90)
- [ ] Can search by keyword
- [ ] Workflow list displays with output preview column
- [ ] Can enter workflow ID to view details
- [ ] Workflow detail shows synthesis FIRST (300px text area)
- [ ] Copy functionality works
- [ ] Metrics display below output
- [ ] Charts render (3 tabs)
- [ ] Can export workflow as JSON

**Configuration Display (1.4):**
- [ ] Navigate to Configuration page - no errors
- [ ] Helix parameters visible (top_radius, bottom_radius, height, turns)
- [ ] Agent parameters visible (max_agents, token_budget, temps)
- [ ] Memory & Compression section visible (ratio, strategy, target)
- [ ] LM Studio connection info visible
- [ ] Dynamic Spawning parameters visible (threshold, volatility)
- [ ] Web Search configuration visible (enabled, provider, max results)
- [ ] No broken file path errors
- [ ] Config loads from existing files or defaults

**General:**
- [ ] No Python import errors on startup
- [ ] All database operations are read-only
- [ ] Missing databases handled gracefully
- [ ] No broken links or file paths
- [ ] All pages load without crashes
