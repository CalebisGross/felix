#!/usr/bin/env python3
"""
Demo script for Engineering Agent Plugins (Frontend, Backend, QA).

This script demonstrates:
1. How custom plugins integrate with Felix's plugin registry
2. Task-based filtering (which agents spawn for which tasks)
3. Real-world engineering scenarios
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.agent_plugin_registry import get_global_registry
from examples.custom_agents.frontend_agent import FrontendAgentPlugin
from examples.custom_agents.backend_agent import BackendAgentPlugin
from examples.custom_agents.qa_agent import QAAgentPlugin


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_scenario(scenario_name, task_description, complexity):
    """Test a scenario and show which agents would spawn."""
    print(f"\n📋 Scenario: {scenario_name}")
    print(f"   Task: {task_description}")
    print(f"   Complexity: {complexity}")
    print(f"\n   Agents that would spawn:")

    # Test each plugin
    frontend = FrontendAgentPlugin()
    backend = BackendAgentPlugin()
    qa = QAAgentPlugin()

    metadata = {"complexity": complexity}

    spawning_agents = []

    if frontend.supports_task(task_description, metadata):
        spawning_agents.append("✅ Frontend Agent (UI/UX, components, responsive design)")

    if backend.supports_task(task_description, metadata):
        spawning_agents.append("✅ Backend Agent (APIs, databases, server architecture)")

    if qa.supports_task(task_description, metadata):
        spawning_agents.append("✅ QA Agent (test strategy, quality assurance)")

    if not spawning_agents:
        print("   ❌ No custom engineering agents (only built-in: research, analysis, critic)")
    else:
        for agent in spawning_agents:
            print(f"   {agent}")


def main():
    print_header("Felix Engineering Agent Plugins Demo")

    print("\n📦 Available Custom Plugins:")
    print("   1. Frontend Agent - UI/UX, React/Vue/Angular, responsive design, accessibility")
    print("   2. Backend Agent  - REST/GraphQL APIs, databases, authentication, microservices")
    print("   3. QA Agent       - Test strategy, quality assurance, edge cases, validation")

    print("\n🔍 Testing Task Filtering (which agents spawn for which tasks)...")

    # Scenario 1: Pure Backend
    test_scenario(
        "REST API Design",
        "Design REST API for user authentication with JWT tokens",
        "complex"
    )

    # Scenario 2: Pure Frontend
    test_scenario(
        "UI Component",
        "Create responsive navigation menu with dropdown for mobile",
        "medium"
    )

    # Scenario 3: Full Stack
    test_scenario(
        "Full-Stack Application",
        "Build todo application with React frontend and Node.js backend API",
        "complex"
    )

    # Scenario 4: Testing Focus
    test_scenario(
        "Testing Strategy",
        "Create comprehensive test suite for payment processing system",
        "complex"
    )

    # Scenario 5: Simple Frontend
    test_scenario(
        "CSS Bug Fix",
        "Fix CSS alignment issue on homepage button",
        "simple"
    )

    # Scenario 6: Non-Technical
    test_scenario(
        "Non-Technical Question",
        "Explain the history of quantum computing",
        "medium"
    )

    # Scenario 7: Database Design
    test_scenario(
        "Database Architecture",
        "Design database schema for e-commerce platform with products, orders, and users",
        "complex"
    )

    # Scenario 8: Accessibility
    test_scenario(
        "Accessibility Improvement",
        "Add ARIA labels and keyboard navigation to dashboard interface",
        "medium"
    )

    print_header("Demo Complete")
    print("\n✨ Key Takeaways:")
    print("   1. Custom plugins use STRICT task filtering to spawn only when needed")
    print("   2. Frontend/Backend agents spawn during analysis phase (0.3-0.6)")
    print("   3. QA agent spawns later (0.5-0.8) to review and test")
    print("   4. Complexity affects spawn timing (complex → spawn earlier)")
    print("   5. Built-in agents (research, analysis, critic) always available as baseline")
    print("\n🚀 To use these plugins in Felix:")
    print("   1. Place plugin files in examples/custom_agents/")
    print("   2. Register with: registry.add_plugin_directory('./examples/custom_agents')")
    print("   3. Plugins auto-spawn based on task description and complexity")
    print()


if __name__ == "__main__":
    main()
