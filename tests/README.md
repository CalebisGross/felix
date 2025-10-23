# Felix Hypothesis Validation Test Suite

This test suite validates the three core hypotheses of the Felix framework:

- **H1**: Helical progression enhances agent adaptation (20% workload distribution improvement)
- **H2**: Hub-spoke communication optimizes resource allocation (15% efficiency gain)
- **H3**: Memory compression reduces latency (25% attention focus improvement)

## Test Structure

```
tests/
├── validation/           # Core validation tests
│   ├── test_h1_workload_distribution.py
│   ├── test_h1_adaptive_behavior.py
│   ├── test_h2_communication_efficiency.py
│   ├── test_h2_resource_allocation.py
│   ├── test_h3_memory_compression.py
│   ├── test_h3_attention_focus.py
│   └── validation_utils.py
├── baselines/           # Baseline implementations for comparison
│   ├── linear_progression.py    # Non-helical agent progression
│   └── mesh_communication.py    # O(N²) mesh topology
├── results/             # Test results (auto-generated)
└── run_hypothesis_validation.py  # Main test runner
```

## Running the Tests

### Full Validation Suite

Run all hypothesis tests with default settings (mock LLM, 5 iterations):
```bash
python tests/run_hypothesis_validation.py
```

### With Real LLM (LM Studio)

Ensure LM Studio is running on port 1234 with a loaded model:
```bash
python tests/run_hypothesis_validation.py --real-llm
```

### Individual Hypothesis Testing

Test specific hypotheses:
```bash
# Test only H1
python tests/run_hypothesis_validation.py --hypothesis H1

# Test only H2
python tests/run_hypothesis_validation.py --hypothesis H2

# Test only H3
python tests/run_hypothesis_validation.py --hypothesis H3
```

### Custom Iterations

Increase iterations for more statistical significance:
```bash
python tests/run_hypothesis_validation.py --iterations 10
```

### Command Line Options

```
--iterations, -i    Number of iterations per test (default: 5)
--real-llm          Use real LLM via LM Studio instead of mock
--hypothesis        Which hypothesis to validate: H1, H2, H3, or all (default: all)
--output, -o        Output file for validation report (default: tests/results/validation_report.json)
```

## Individual Test Execution

You can also run individual tests directly:

```bash
# H1 Tests
python tests/validation/test_h1_workload_distribution.py
python tests/validation/test_h1_adaptive_behavior.py

# H2 Tests
python tests/validation/test_h2_communication_efficiency.py
python tests/validation/test_h2_resource_allocation.py

# H3 Tests
python tests/validation/test_h3_memory_compression.py
python tests/validation/test_h3_attention_focus.py
```

## Understanding the Results

### Success Criteria

Each hypothesis must achieve its target improvement percentage:
- H1: ≥20% improvement
- H2: ≥15% improvement
- H3: ≥25% improvement

### Output Files

Results are saved to `tests/results/` with timestamps:
- `validation_report.json` - Complete validation report
- `h1_*.json` - Individual H1 test results
- `h2_*.json` - Individual H2 test results
- `h3_*.json` - Individual H3 test results

### Metrics Measured

**H1 - Workload Distribution & Adaptive Behavior:**
- Workload variance across agents
- Temperature gradient smoothness
- Token budget adaptation
- Agent type distribution

**H2 - Communication Efficiency & Resource Allocation:**
- Message count (O(N) vs O(N²))
- Routing time
- Memory usage
- Token allocation efficiency
- Resource waste reduction

**H3 - Memory Compression & Attention Focus:**
- Processing latency
- Context compression ratio
- Attention focus score
- Key concept retention
- Noise reduction

## Interpreting Test Results

### Successful Validation Example
```
📊 H1: Helical Progression Enhances Adaptation
   Target: 20% improvement
   Achieved: 23.4%
   Success Rate: 80.0%
   Status: ✅ PASSED
```

### Failed Validation Example
```
📊 H2: Hub-Spoke Communication Optimizes Resources
   Target: 15% improvement
   Achieved: 12.1%
   Success Rate: 40.0%
   Status: ❌ FAILED
```

## Troubleshooting

### Mock LLM vs Real LLM

- **Mock LLM** (default): Uses simulated responses for fast testing without external dependencies
- **Real LLM**: Requires LM Studio running on `http://localhost:1234` with a loaded model

### Common Issues

1. **Import errors**: Ensure you're running from the project root or tests directory
2. **LM Studio connection failed**: Check that LM Studio is running and has a model loaded
3. **Memory errors**: Reduce `--iterations` if running out of memory
4. **Slow tests**: Use mock LLM for faster iteration during development

## Test Development

### Adding New Tests

1. Create test file in `tests/validation/`
2. Inherit from `TestRunner` base class
3. Implement `run_test()` method
4. Add to main runner in `run_hypothesis_validation.py`

### Baseline Comparisons

Each hypothesis test compares Felix against a baseline:
- H1: Linear progression (no helix)
- H2: Mesh communication (O(N²))
- H3: No compression (full context)

## Continuous Integration

Exit codes for CI/CD:
- `0`: All tests passed
- `1`: One or more tests failed

Example GitHub Actions workflow:
```yaml
- name: Run Felix Validation
  run: python tests/run_hypothesis_validation.py --iterations 3
```

## Contributing

When modifying Felix core components, ensure tests still pass:
```bash
# Quick validation (3 iterations, mock LLM)
python tests/run_hypothesis_validation.py --iterations 3

# Thorough validation (10 iterations, real LLM)
python tests/run_hypothesis_validation.py --iterations 10 --real-llm
```