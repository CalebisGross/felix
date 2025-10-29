# Technical Debt Tracking

## Streamlit Width Parameter Transition Issue

### Issue Summary
- **Date Identified**: 2025-10-26
- **Streamlit Version**: 1.50.0
- **Issue Type**: API Deprecation Paradox

### Problem Description
Streamlit 1.50.0 shows deprecation warnings for `use_container_width` parameter, announcing it will be removed after 2025-12-31. However, the replacement `width='stretch'` parameter is not yet implemented in version 1.50.0, creating a paradox where:
- Using `use_container_width=True` triggers deprecation warnings
- Using `width='stretch'` causes "keyword arguments deprecated" errors (worse)

### Temporary Solution Implemented
1. **Warning Suppression**: Added warning filters in `app.py` to suppress UI warnings:
   - Python warnings module filters (successfully hides warnings from Streamlit UI)
   - Multiple filter patterns to catch all warning variations
   - **Note**: Terminal warnings cannot be suppressed without causing performance degradation
   - Attempted logging level adjustment but caused significant slowdown - reverted
2. **Reverted Parameters**: Changed all instances back to `use_container_width=True` (32 changes across 8 files)

### Files Affected
- `pages/1_Dashboard.py` - 9 instances (7 plotly_chart + 2 dataframe)
- `pages/2_Configuration.py` - 1 instance (plotly_chart)
- `pages/3_Testing.py` - 10 instances (7 plotly_chart + 3 dataframe)
- `pages/4_Benchmarking.py` - 2 instances (1 plotly_chart + 1 dataframe)
- `components/agent_visualizer.py` - 1 instance (plotly_chart)
- `components/results_analyzer.py` - 1 instance (plotly_chart)
- `components/web_search_monitor.py` - 4 instances (3 plotly_chart + 1 dataframe)
- `components/workflow_history_viewer.py` - 4 instances (3 plotly_chart + 1 dataframe)

### Action Required
- **Monitor**: Check for Streamlit releases > 1.50.0 that properly implement the `width` parameter
- **Test Command**: `pip index versions streamlit`
- **Verification**: When new version available, test if `width='stretch'` works without errors
- **Migration**: Once verified, remove warning suppression from `app.py` and update all instances to use `width='stretch'`
- **Deadline**: Before 2025-12-31 (when `use_container_width` will be removed)

### Related GitHub Issues
- Streamlit is actively working on this transition as part of their Advanced Layouts initiative
- Multiple PRs have been merged adding `width` parameter to various components
- The deprecation timeline gives developers until end of 2025 to migrate

### Testing Checklist
When migrating to new version:
- [ ] Upgrade Streamlit: `pip install --upgrade streamlit`
- [ ] Remove warning suppression from `app.py`
- [ ] Update all `use_container_width=True` to `width='stretch'`
- [ ] Test all chart displays work correctly
- [ ] Verify no deprecation warnings appear
- [ ] Update this document with resolution date

### Code Locations for Future Update
Search pattern to find all instances:
```bash
grep -r "use_container_width=True" streamlit_gui/
```

### Notes
- This is a known issue in Streamlit's transition from boolean to string-based width parameters
- The warning suppression is a temporary measure and should be removed once resolved
- No functional impact on the application - purely a deprecation management issue
- **UI warnings are successfully suppressed** - users see a clean interface
- **Terminal warnings still appear** - this is acceptable as they don't affect end users, only developers
- Attempting to suppress terminal warnings via logging causes performance degradation - not recommended