# Felix Framework Experimentation Directory

This directory contains detailed explanations and examples of how the Felix framework works, based on comprehensive analysis of the codebase and verification of components.

## Contents

### exp/README.md
This file - overview of the experimentation directory contents and usage guide.

### exp/workflow_steps.md
Detailed step-by-step breakdown of the Felix framework workflow, including:
- Initialization and setup
- Agent spawning and lifecycle management
- Communication flow between components
- Task processing with LLM integration
- Memory interactions and knowledge storage
- Dynamic features (confidence monitoring, spawning triggers)
- Pipeline integration for sequential processing
- Hypothesis validation metrics collection

### exp/example_workflow.py
Complete, runnable Python script demonstrating a minimal end-to-end Felix workflow. Features:
- Mock LLM responses (no external server required)
- Key component instantiation and interaction
- Agent progression simulation along helical geometry
- Message passing and result storage
- Comments explaining each step of the process

### exp/component_interactions.md
Visual diagram (using Markdown tables and ASCII art) showing how Felix components interact:
- Agent ↔ CentralPost ↔ Memory relationships
- HelixGeometry integration with LLMAgent positioning
- Communication flow patterns
- Data flow between modules

## Usage

1. **Read workflow_steps.md first** for conceptual understanding
2. **Run example_workflow.py** to see a working demonstration
3. **Refer to component_interactions.md** for architecture visualization
4. **Use these files** as reference for understanding Felix internals

## Prerequisites

- Python 3.8+
- Felix framework installed (see main README)
- Virtual environment activated (optional but recommended)

## Notes

- All examples use mock LLM responses to avoid external dependencies
- Components are instantiated from the verified src/ modules
- Examples demonstrate key Felix concepts: helical progression, multi-agent communication, memory persistence, and dynamic spawning