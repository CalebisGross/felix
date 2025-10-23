# Simple Query Testing Guide

## Changes Made to Fix Over-Engineering

### Problem
Simple queries like "what is the current date and time" resulted in:
- 1789 tokens of philosophical synthesis
- 11 LLM responses across 5 agents
- No web searches despite capability
- Agents discussing "temporal absence as a design principle"

### Solution
Implemented intelligent task complexity detection and optimization:

---

## Test Cases

### 1. Simple Factual Queries (SIMPLE_FACTUAL)

**Expected Behavior:**
- Classified as `SIMPLE_FACTUAL`
- Only 5 workflow steps (instead of 25+)
- Only 1 research agent spawned
- Agent will use `WEB_SEARCH_NEEDED:` to request web search
- Synthesis limited to 200 tokens
- Direct, concise answer

**Test Queries:**
```
"what is the current date and time"
"what time is it"
"who won the 2024 US presidential election"
"when is the next solar eclipse"
"how many people live in Tokyo today"
"latest news about AI"
```

**Expected Output:**
```
Query: "what is the current date and time"

Task complexity classification: SIMPLE_FACTUAL
Simple factual query detected - using minimal steps: 5

[Research Agent]
WEB_SEARCH_NEEDED: current date and time

[CentralPost performs web search]
Found results...

[Synthesis - 200 token limit]
The current date and time is October 23, 2025, 11:30 AM EDT.

Total: ~100-200 tokens, 1 agent, <5 seconds
```

---

### 2. Medium Complexity Queries (MEDIUM)

**Expected Behavior:**
- Classified as `MEDIUM`
- Moderate workflow steps
- 2-3 agents spawned
- Synthesis limited to 800-1200 tokens
- Focused answer without over-elaboration

**Test Queries:**
```
"explain how quantum computers work"
"compare Python and JavaScript"
"how to learn machine learning"
"list the benefits of exercise"
"summarize the French Revolution"
```

**Expected Output:**
```
Task complexity classification: MEDIUM
Medium complexity detected - using moderate steps: 15

[2-3 agents process the query]
[Focused synthesis 800-1200 tokens]

Total: ~1000 tokens, 2-3 agents, ~15 seconds
```

---

### 3. Complex Analytical Queries (COMPLEX)

**Expected Behavior:**
- Classified as `COMPLEX`
- Full workflow steps (25+)
- Multiple agents spawned dynamically
- Synthesis up to 3000 tokens
- Comprehensive, multi-faceted answer

**Test Queries:**
```
"analyze the benefits of helical agent progression"
"evaluate the economic impact of AI on employment"
"design a multi-agent system for healthcare"
"what are the philosophical implications of consciousness in AI"
```

**Expected Output:**
```
Task complexity classification: COMPLEX
Complex task detected - using maximum steps: 25

[5+ agents process the query]
[Comprehensive synthesis 1500-3000 tokens]

Total: ~2000-3000 tokens, 5+ agents, ~30 seconds
```

---

## How to Test

### Using the GUI:
1. Start Felix GUI: `python -m src.gui`
2. Go to Settings tab
3. **Enable Web Search** (if not already enabled)
4. Save settings
5. Start Felix system from Dashboard
6. Go to Workflows tab
7. Test each query type above
8. Observe:
   - Task complexity classification in logs
   - Number of agents spawned
   - Whether web search is triggered
   - Final synthesis token count

### Expected Improvements:

| Query Type | Before | After |
|------------|--------|-------|
| **Simple Factual** | 5 agents, 1789 tokens, no search | 1 agent, ~150 tokens, web search |
| **Medium** | 5 agents, 1789 tokens | 2-3 agents, ~1000 tokens |
| **Complex** | 5 agents, 1789 tokens | 5+ agents, ~2000 tokens (appropriate) |

---

## Key Features

### 1. Web Search Awareness
Agents now know they can request web searches:
```
🔍 AVAILABLE TOOL - Web Search:
If you need current information, include:
WEB_SEARCH_NEEDED: [your specific search query]
```

### 2. Task Complexity Classification
Automatic detection using regex patterns:
- **Simple**: Current time, who/what/when questions about recent events
- **Medium**: Explain, compare, how-to questions
- **Complex**: Analyze, evaluate, design questions

### 3. Optimized Synthesis
- Simple: 200 tokens, direct answer
- Medium: 800-1200 tokens, focused response
- Complex: 1500-3000 tokens, comprehensive synthesis

### 4. Early Exit
- Simple queries: 5 steps, 1 agent
- Medium queries: 15 steps, 2-3 agents
- Complex queries: 25+ steps, 5+ agents

---

## Debugging

### Check Logs For:

**Task Classification:**
```
Task complexity classification: SIMPLE_FACTUAL
Simple factual query detected - using minimal steps: 5
```

**Web Search Request:**
```
AGENT-REQUESTED WEB SEARCH
Requesting Agent: research_001
Query: "current date and time"
```

**Synthesis Parameters:**
```
Synthesis Parameters:
  Task complexity: SIMPLE_FACTUAL
  Agent messages: 1
  Adaptive token budget: 200
```

---

## Common Issues

### 1. Web Search Not Triggering
- **Check**: Is `enable_web_search` enabled in settings?
- **Check**: Is the agent including `WEB_SEARCH_NEEDED:` in response?
- **Check**: Is web search client available (ddgs package installed)?

### 2. Still Getting Verbose Responses
- **Check**: Is task being classified correctly? (Check logs)
- **Check**: Are synthesis parameters showing correct token limit?
- **Check**: Is the LLM following the instructions? (May need stronger model)

### 3. Task Misclassification
- **Solution**: Update patterns in `_classify_task_complexity()` in `felix_workflow.py`
- Add more patterns for simple queries
- Adjust regex patterns as needed

---

## Files Modified

1. **src/llm/lm_studio_client.py**
   - Added web search awareness to agent prompts

2. **src/communication/central_post.py**
   - Added `_handle_web_search_request()` method
   - Updated `_build_synthesis_prompt()` with complexity-specific instructions
   - Updated `_calculate_synthesis_tokens()` with complexity-based limits
   - Updated `synthesize_agent_outputs()` signature

3. **src/workflows/felix_workflow.py**
   - Added `_classify_task_complexity()` function
   - Updated workflow to adjust steps based on complexity
   - Skip dynamic spawning for simple queries
   - Pass `task_complexity` to synthesis

---

## Success Criteria

✅ **Simple Query Test:**
- Query: "what is the current date and time"
- Classified as SIMPLE_FACTUAL
- Web search triggered
- 1 agent only
- Response < 200 tokens
- Direct answer provided

✅ **Medium Query Test:**
- Query: "explain quantum computing"
- Classified as MEDIUM
- 2-3 agents
- Response 800-1200 tokens
- Focused explanation

✅ **Complex Query Test:**
- Query: "analyze helical agent progression"
- Classified as COMPLEX
- 5+ agents
- Response 1500-3000 tokens
- Comprehensive analysis

---

## Next Steps

1. Test all query types in the GUI
2. Monitor token usage improvements
3. Adjust classification patterns if needed
4. Report any issues with task classification
5. Consider adding user override for complexity (if needed)

---

**Questions or Issues?**
Check the logs for task complexity classification and synthesis parameters.
Adjust patterns in `_classify_task_complexity()` if queries are misclassified.
