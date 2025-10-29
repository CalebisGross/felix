# Phase 2 & 3 Testing Checklist

## Phase 2: Web Search Monitoring
- [ ] `DatabaseReader.get_web_search_activity()` returns DataFrame with search data
- [ ] Parses JSON from knowledge_entries WHERE domain='web_search'
- [ ] `DatabaseReader.get_web_search_stats()` calculates all 5 metrics
- [ ] Handles empty database gracefully (returns empty DataFrame/zero stats)

## Phase 3: Advanced Features

### Truth Assessment Display
- [ ] Truth assessment component detects validation workflows via keywords
- [ ] Color-coded badges display based on confidence thresholds
- [ ] Badge shows: Validated (≥85%), Needs Review (70-84%), Failed (<70%)
- [ ] Source extraction and reasoning display work
- [ ] Integrated into Workflow History viewer detail view

### Web Search Configuration
- [ ] Configuration page displays web search settings section
- [ ] Shows: status, provider, max results/queries, blocked domains
- [ ] Displays confidence threshold and SearxNG URL (when applicable)
- [ ] Two-column metric layout with tooltips

### Advanced Analytics
- [ ] Dashboard Performance Trends tab loads enhanced analytics
- [ ] Hypothesis Performance Tracker shows target vs actual bar chart
- [ ] Agent Efficiency Matrix displays scatter plot
- [ ] Summary metrics display (total agents, max/avg efficiency)
- [ ] Charts handle missing data gracefully
