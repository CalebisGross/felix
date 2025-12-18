# Felix Framework: Gap Analysis Report

**Date:** 2025-12-17
**Analysis Method:** Deep dive code tracing across 5 system areas
**Scope:** Communication, Knowledge/Memory, Agents, Prompting, System Integration

---

## Executive Summary

This analysis identified **46 actual gaps and disconnections** in the Felix codebase. Unlike the previous Gemini analysis (deleted), these are verified issues with exact file:line references and code evidence.

**Critical Issues (System Breaking):** 6
**High Priority (Feature Breaking):** 12
**Medium Priority (Degraded Behavior):** 28

---

## CRITICAL ISSUES (Fix Immediately)

### 1. API Import Path Broken
**Location:** `src/api/dependencies.py:12`, `src/api/routers/agents.py:22`, `src/api/routers/workflows.py:26`

**Problem:** Imports from `src.gui.felix_system` but module doesn't exist. Actual location is `src.core.felix_system`.

```python
# BROKEN
from src.gui.felix_system import FelixSystem, FelixConfig

# SHOULD BE
from src.core.felix_system import FelixSystem, FelixConfig
```

**Impact:** REST API fails to start with ImportError.

---

### 2. Unhandled Message Types (Meta-Learning Loop Broken)
**Location:** `src/communication/central_post.py:1472-1617`

**Problem:** Three message types are SENT but never HANDLED:
- `MessageType.SYNTHESIS_FEEDBACK` (sent line 1472)
- `MessageType.CONTRIBUTION_EVALUATION` (sent line 1485)
- `MessageType.IMPROVEMENT_REQUEST` (defined, never used)

The message handler (lines 1594-1617) has no case for these types.

**Impact:** Agents never receive feedback on their contributions. The meta-learning loop where agents improve based on synthesis results is completely broken.

---

### 3. Agent Personality Traits Never Injected Into Prompts
**Location:** `src/agents/specialized_agents.py:111, 315, 387` + `src/prompts/prompt_manager.py:494-512`

**Problem:** Agent traits are stored but never used:
```python
# Stored
self.research_domain = research_domain  # "technical"
self.analysis_type = analysis_type      # "deep"
self.review_focus = review_focus        # "security"

# Template has placeholders
"Research Domain: {research_domain}"

# But render_template() returns unsubstituted template on KeyError!
return template  # "{research_domain}" still in output
```

**Impact:** All agents behave identically regardless of specialization. The entire agent personality system is non-functional.

---

### 4. Web Search Coordinator State Never Initialized
**Location:** `src/communication/web_search_coordinator.py:94-108`

**Problem:** Three initialization methods are defined but NEVER CALLED:
- `set_task_context()` - task description always None
- `update_confidence()` - confidence window always empty
- `add_processed_message()` - message history always empty

**Impact:** Confidence-based web search triggering is completely broken. Searches only happen via explicit patterns, not automatic confidence monitoring.

---

### 5. Knowledge Relationships Missing CASCADE DELETE
**Location:** `src/memory/knowledge_store.py:242-251`

**Problem:** `knowledge_relationships` table has no FOREIGN KEY constraint:
```sql
CREATE TABLE IF NOT EXISTS knowledge_relationships (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    -- MISSING: FOREIGN KEY ... ON DELETE CASCADE
)
```

**Impact:** When knowledge entries are deleted, orphaned relationships accumulate indefinitely. Database grows with garbage data.

---

### 6. Knowledge Store Singleton Broken (Multiple Instances)
**Location:** `src/core/felix_system.py:346` vs `src/api/dependencies.py:268-288`

**Problem:** FelixSystem creates one KnowledgeStore instance. API dependencies create a SEPARATE instance:
```python
# FelixSystem
self.knowledge_store = KnowledgeStore(self.config.knowledge_db_path)

# API (different instance, possibly different path!)
def get_knowledge_store_memory():
    return KnowledgeStore()  # New instance!
```

**Impact:** GUI and API modifications don't see each other. Data isolation between components.

---

## HIGH PRIORITY (Feature Breaking)

### 7. Method Name Mismatch: unregister_agent vs deregister_agent
- **FelixSystem defines:** `deregister_agent()`
- **API calls:** `unregister_agent()`
- **Location:** `src/api/routers/agents.py:327`
- **Impact:** Agent deletion endpoint raises AttributeError

### 8. Approval Callback Not Forwarded to Synthesis Engine
- **Location:** `src/communication/central_post.py:1352-1414`
- **Problem:** `approval_callback` parameter accepted but not passed through
- **Impact:** Low-confidence synthesis auto-accepts without user review

### 9. Two Separate Confidence Windows (Never Synced)
- **CentralPost:** `self._recent_confidences` (updated)
- **WebSearchCoordinator:** `self._recent_confidences` (NEVER updated)
- **Location:** `central_post.py:988` vs `web_search_coordinator.py:99`
- **Impact:** Confidence monitoring uses wrong/empty data

### 10. Async Handlers Are Empty Stubs
- **Location:** `src/communication/central_post.py:1987-2032`
- **Problem:** `_handle_*_async()` methods are either `pass` or blocking sync calls
- **Impact:** Async message processing provides zero parallelism

### 11. Task Memory Recommendations Ignored During Spawning
- **Location:** `src/workflows/dynamic_spawning.py:832-900`
- **Problem:** `recommend_strategy()` called, cached, but never used in `analyze_and_spawn()`
- **Impact:** Pattern learning has no effect on agent spawning decisions

### 12. Agent Lifecycle Events Never Emitted
- **Location:** `src/agents/felix_agent.py:385-395, 742-745`
- **Problem:** No signals/events when agents spawn, complete, or fail
- **Impact:** GUI/API have no way to track agent lifecycle

### 13. Helix Position Ignored in Complexity Classification
- **Location:** `src/agents/felix_agent.py:124-168`
- **Problem:** `classify_complexity()` never checks helix position
- **Impact:** Top-of-helix (exploration) agents behave same as bottom (synthesis)

### 14. Direct Answer Mode Only in ResearchAgent
- **Location:** `src/agents/specialized_agents.py:113-254`
- **Problem:** Only ResearchAgent has direct answer optimization
- **Impact:** AnalysisAgent and CriticAgent take longer paths unnecessarily

### 15. Silent Recovery Failure in Tier Downgrade
- **Location:** `src/knowledge/embeddings.py:706-709`
- **Problem:** Tier downgrade doesn't properly notify recovery manager
- **Impact:** System may repeatedly attempt recovery to just-failed tier

### 16. API Dependency Injection Missing Startup Hook
- **Location:** `src/api/dependencies.py:218-227`
- **Problem:** `get_felix()` raises exception if Felix not initialized
- **Impact:** First API request always fails (503) until explicit init

### 17. Incomplete Component Initialization Order
- **Location:** `src/core/felix_system.py:267-498`
- **Problem:** If `enable_memory=False`, knowledge_store is None but passed to CentralPost
- **Impact:** NoneType errors in downstream components

### 18. Agent Deregistration Incomplete (DB Not Updated)
- **Location:** `src/communication/central_post.py:397-423`
- **Problem:** Removes from memory dicts but not from SQLite live database
- **Impact:** Database shows "active" agents that no longer exist

---

## MEDIUM PRIORITY (Degraded Behavior)

### 19. Streaming Coordinator No Cancellation Mechanism
- **Location:** `src/communication/streaming_coordinator.py:27-189`
- No `cancel_stream()` method exists

### 20. Streaming State Memory Leak on Agent Crash
- **Location:** `src/communication/streaming_coordinator.py:88-122`
- Crashed agents leave orphaned entries

### 21. Direct Knowledge Store Calls Bypass Facade
- **Location:** `src/communication/central_post.py:1509`
- `knowledge_store.record_knowledge_usage()` bypasses MemoryFacade

### 22. Synthesis Confidence Gating Incomplete
- **Location:** `src/communication/synthesis_engine.py:840-911`
- Approval only applies if callback provided; should always gate low confidence

### 23. Related Entries JSON Never Synced with Relationships Table
- **Location:** `src/memory/knowledge_store.py:560, 926`
- Two systems tracking relationships = constant divergence

### 24. Auto-Resolved Gaps Not Linked to Knowledge
- **Location:** `src/knowledge/gap_tracker.py:368-411`
- Gap marked resolved but knowledge entry doesn't track which gaps it resolved

### 25. Gap Severity Never Updated After Resolution
- **Location:** `src/knowledge/gap_tracker.py:165-197, 258-285`
- Severity adjusted for outcomes but not resolution

### 26. Task Execution Complexity Never Validated
- **Location:** `src/memory/task_memory.py:197-274`
- Complexity passed in but never cross-checked

### 27. Agent Performance Metrics Collected But Not Used Adaptively
- **Location:** `src/memory/agent_performance_tracker.py:105-191`
- Data collected but boost calculation uses hardcoded thresholds

### 28. Workflow Agent Summary Never Called (Dead Code)
- **Location:** `src/memory/agent_performance_tracker.py:222-266`
- Method exists but zero callers

### 29. knowledge_usage Table Missing from _init_database()
- **Location:** `src/memory/knowledge_store.py:155-293`
- Only created by migration; crashes if migration skipped

### 30. Meta-Learning Boost Fails Silently if Table Missing
- **Location:** `src/memory/knowledge_store.py:800-809`
- Returns original order without warning

### 31. Knowledge Usage Not Recorded After Retrieval
- **Location:** Only one call at `central_post.py:1509`
- Not systematic across all retrievals

### 32. Recovery Check Interval Too Long (60s default)
- **Location:** `src/knowledge/embeddings.py:46`
- Users get degraded results for up to 60s

### 33. Plugin Stats Not Reset on Reload
- **Location:** `src/agents/agent_plugin_registry.py:508-536`
- `external_count` becomes stale after reload

### 34. Plugin Load Errors Not Surfaced
- **Location:** `src/agents/agent_plugin_registry.py:174-199`
- Caller can't distinguish "no plugins" from "all failed"

### 35. Prompt Lookup Key Parsing Fragile
- **Location:** `src/prompts/prompt_manager.py:211-261`
- Assumes `{agent_type}_{sub_key}` format

### 36. Synthesis Prompts Missing Agent Type Metadata
- **Location:** `config/prompts.yaml:356-400`
- LLM doesn't know which agent said what

### 37. Three Entry Points, Three Prompt Loading Paths
- CLI, GUI PySide6, GUI CTK each load prompts differently

### 38. Chat System Prompt Bypasses PromptManager
- **Location:** `src/agents/felix_agent.py:92-122`
- Hardcoded file path, no DB overrides

### 39. Agent Cleanup Exception Swallowed
- **Location:** `src/agents/felix_agent.py:737-745`
- Deregistration failure logged but not handled

### 40. Signal-Adapter Disconnect (Weak Connections)
- **Location:** `src/gui/adapters/felix_adapter.py:40-48`
- No guarantee GUI components have connected

### 41. Synthesis Approval Blocking Thread Timeout
- **Location:** `src/gui/adapters/felix_adapter.py:264-304`
- 5-minute wait before auto-accept on timeout

### 42. Config Values Loaded But Never Used
- **Location:** `config/llm.yaml:64-73` vs `src/core/felix_system.py:377-382`
- YAML parsed separately from FelixConfig

### 43. Spoke Topology Silent Fallback Causes Double Registration
- **Location:** `src/core/felix_system.py:469-471`
- If spoke fails, falls back AND registers directly

### 44. Synthesis Engine Wiring Not Guaranteed
- **Location:** `src/core/felix_system.py:449-466`
- No check that synthesis_engine initialized successfully

### 45. Workflow Endpoint Hardcodes "unknown" Values
- **Location:** `src/api/routers/workflows.py:95-230`
- Agent type always "unknown" in result

### 46. SYSTEM_ACTION_REQUEST Sync in Async Context
- **Location:** `src/communication/central_post.py:1676`
- Blocking call in async handler

---

## Recommended Fix Order

### Phase 1: Unblock API (Day 1)
1. Fix import paths (`src.gui.felix_system` -> `src.core.felix_system`)
2. Fix method name (`unregister_agent` -> `deregister_agent`)
3. Add API startup initialization hook

### Phase 2: Fix Meta-Learning (Day 2-3)
4. Add handlers for SYNTHESIS_FEEDBACK, CONTRIBUTION_EVALUATION
5. Wire agent personality traits through prompt pipeline
6. Connect WebSearchCoordinator state initialization
7. Use task memory recommendations in agent spawning

### Phase 3: Data Integrity (Day 4-5)
8. Add CASCADE DELETE to knowledge_relationships
9. Fix knowledge store singleton pattern
10. Ensure knowledge_usage table in _init_database()
11. Sync related_entries JSON with relationships table

### Phase 4: Polish (Week 2)
12-46. Address remaining medium priority issues

---

## Notes

This analysis was conducted by deep code tracing, not surface-level pattern matching. Each gap was verified by:
1. Finding where functionality SHOULD work
2. Tracing the actual code path
3. Identifying where the connection breaks
4. Confirming with grep/search that no alternate path exists

The previous Gemini analysis has been deleted as it contained factual errors (e.g., claiming mock provider didn't exist when it does, claiming plugin discovery was hardcoded when external support exists).
