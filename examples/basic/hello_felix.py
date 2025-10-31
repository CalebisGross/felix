"""
Hello Felix - Simplest possible workflow

This example shows the absolute minimum code needed to run a Felix workflow.
"""

from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory
from src.llm.router_adapter import create_router_adapter
from src.workflows.felix_workflow import execute_linear_workflow_optimized


def main():
    print("Felix Framework - Hello World Example")
    print("=" * 50)

    # Step 1: Initialize helix geometry (defines agent progression)
    helix = HelixGeometry(
        top_radius=3.0,      # Wide exploration at top
        bottom_radius=0.5,   # Narrow synthesis at bottom
        height=8.0,          # Total depth
        turns=2              # Spiral complexity
    )

    # Step 2: Initialize LLM client (uses config/llm.yaml)
    try:
        llm_client = create_router_adapter()
        print("✓ LLM client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize LLM: {e}")
        print("  Make sure LM Studio is running or cloud provider is configured")
        return

    # Step 3: Create communication hub
    central_post = CentralPost(helix)
    agent_factory = AgentFactory(central_post, helix, llm_client)
    print("✓ Communication hub ready")

    # Step 4: Create a simple system object
    class SimpleSystem:
        def __init__(self):
            self.helix = helix
            self.central_post = central_post
            self.agent_factory = agent_factory
            self.lm_client = llm_client

            # Minimal config
            class Config:
                workflow_max_steps = 5
                enable_web_search = False
                workflow_simple_threshold = 0.8
                workflow_medium_threshold = 0.6
                workflow_max_steps_simple = 3
                workflow_max_steps_medium = 5
                workflow_max_steps_complex = 10
                confidence_threshold = 0.80

            self.config = Config()
            self.task_memory = None

    system = SimpleSystem()

    # Step 5: Run a simple workflow
    print("\n" + "=" * 50)
    print("Running workflow...")
    print("=" * 50 + "\n")

    task = "Explain what makes a good software engineer in 3 sentences"

    result = execute_linear_workflow_optimized(
        task_input=task,
        felix_system=system,
        max_steps_override=5
    )

    # Step 6: Display results
    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50 + "\n")

    synthesis = result.get("centralpost_synthesis", {})
    content = synthesis.get("synthesis_content", "No result generated")
    confidence = synthesis.get("confidence", 0.0)
    agents_count = len(result.get("agents_spawned", []))

    print(content)
    print(f"\n📊 Confidence: {confidence:.2f}")
    print(f"🤖 Agents used: {agents_count}")


if __name__ == "__main__":
    main()
