# Phase 1 Testing Checklist

## Benchmarking System
- [ ] Benchmarking page loads without import errors
- [ ] Can select hypothesis (all/H1/H2/H3) and set iterations (1-20)
- [ ] "Run Validation Tests" executes with progress messages
- [ ] Results display with metrics, box plots, and JSON download

## Workflow History Browser
- [ ] Dashboard → Workflow History tab displays summary metrics
- [ ] Filter by status, days back, and keyword search work
- [ ] Workflow list includes output preview column
- [ ] Workflow detail shows synthesis output first (300px text area)
- [ ] Copy functionality and JSON export work
- [ ] Analytics charts render (confidence, tokens, time)

## Configuration Display
- [ ] Configuration page loads all parameters without errors
- [ ] Helix geometry, agent config, memory/compression visible
- [ ] LM Studio connection, dynamic spawning, web search config displayed
- [ ] Loads from files or defaults gracefully

## General
- [ ] No Python import errors on startup
- [ ] All database operations read-only
- [ ] Missing databases handled gracefully
- [ ] All pages load without crashes
