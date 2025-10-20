# Felix Benchmarks

This directory contains comparison baseline implementations used for validating Felix framework hypotheses.

## Files

### linear_pipeline.py

**Purpose**: Comparison baseline for hypothesis validation

**NOT part of Felix framework** - this is a traditional sequential pipeline architecture used as a research baseline to validate the benefits of Felix's helix-based architecture.

**Key Differences from Felix**:
- ❌ Sequential stage-based processing (not helix-based)
- ❌ O(N²) mesh communication (not O(N) hub-spoke)
- ❌ Uniform workload distribution (not adaptive)
- ❌ Creates own LLM client and agents (doesn't use Felix system)
- ❌ No CentralPost or AgentFactory integration

**Used For**:
- Hypothesis H1 validation (workload distribution)
- Hypothesis H2 validation (communication efficiency)
- Performance comparison studies
- Statistical analysis of architecture differences

## Felix Framework Workflow

For **actual Felix workflow implementation**, see:
- `src/workflows/felix_workflow.py` - Proper Felix-integrated workflow
- `exp/example_workflow.py` - Complete demonstration of Felix architecture

The proper Felix workflow uses:
- ✅ CentralPost for O(N) hub-spoke communication
- ✅ AgentFactory for dynamic agent spawning
- ✅ Helix-based agent progression
- ✅ Shared LLM client across system
- ✅ Knowledge store and memory integration
- ✅ Confidence-based dynamic spawning

## Running Benchmarks

```bash
# Run Felix workflow (proper implementation)
python -m src.gui.main  # Use GUI Workflows tab

# Run benchmark pipeline (for comparison only)
cd exp/benchmarks
python -c "from linear_pipeline import run; run('test task')"
```

## Important Notes

1. **Do not use linear_pipeline.py for production workflows**
2. It's maintained solely for research comparison
3. GUI workflows now use `src/workflows/felix_workflow.py`
4. Benchmark results help validate Felix architecture benefits
