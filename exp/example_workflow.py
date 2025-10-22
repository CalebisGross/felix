#!/usr/bin/env python3
"""
Felix Framework Example Workflow

Complete, runnable demonstration of a minimal Felix multi-agent workflow.
Uses mock LLM responses to avoid external server dependencies.

This script demonstrates:
1. Component initialization (HelixGeometry, CentralPost, AgentFactory)
2. Agent spawning and registration
3. Task processing with position-aware behavior
4. Message passing and result sharing
5. Memory storage and retrieval
6. Dynamic spawning based on confidence
7. Final result generation

Run with: python exp/example_workflow.py
"""

import sys
import time
from typing import Dict, Any
from dataclasses import dataclass

# Import Felix components
sys.path.append('.')
from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory, Message, MessageType
from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, CriticAgent
from src.memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel
from src.llm.token_budget import TokenBudgetManager


@dataclass
class MockLLMResponse:
    """Mock LLM response for demonstration."""
    content: str
    tokens_used: int = 150
    finish_reason: str = "stop"


class MockLMStudioClient:
    """Mock LLM client that returns predetermined responses."""

    def __init__(self):
        self.response_counter = 0
        self.responses = [
            # Research responses
            "Research findings: Python is a high-level programming language known for its simplicity and readability. Key features include dynamic typing, automatic memory management, and extensive standard library.",

            "Technical research: Python's GIL (Global Interpreter Lock) can limit multi-threading performance for CPU-bound tasks, but it's excellent for I/O operations and web development.",

            # Analysis responses
            "Analysis: Python's strengths lie in rapid development, extensive libraries (NumPy, Pandas, Django), and strong community support. Weaknesses include GIL limitations and relatively slower execution compared to compiled languages.",

            "Critical analysis: Python excels in data science, web development, and automation. However, for high-performance computing, languages like C++ or Rust may be more suitable.",

            # Synthesis responses
            "Final assessment: Python is an excellent choice for most software development projects due to its productivity benefits, extensive ecosystem, and ease of maintenance. Recommended for web applications, data analysis, and rapid prototyping.",

            "Executive summary: Python offers the best balance of developer productivity and performance for 80% of software projects. Its weaknesses are well-understood and can be mitigated through proper architecture."
        ]

    def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                temperature: float, max_tokens: int) -> MockLLMResponse:
        """Return mock response based on agent type."""
        response = self.responses[self.response_counter % len(self.responses)]
        self.response_counter += 1
        return MockLLMResponse(response)


def create_task(task_description: str) -> Dict[str, Any]:
    """Create a task dictionary for processing."""
    return {
        "id": f"task_{int(time.time())}",
        "description": task_description,
        "context": "Evaluate Python as a programming language for modern software development."
    }


def simulate_agent_progression(agent, task: Dict[str, Any], current_time: float,
                              central_post: CentralPost) -> None:
    """Simulate agent processing a task at current time."""
    if agent.state == "waiting" and agent.can_spawn(current_time):
        print(f"[{current_time:.1f}] Spawning {agent.agent_type} agent: {agent.agent_id}")

        # Create LLM task
        from src.agents.llm_agent import LLMTask
        llm_task = LLMTask(
            task_id=task["id"],
            description=task["description"],
            context=task["context"]
        )

        # Spawn and process
        agent.spawn(current_time, llm_task)
        result = agent.process_task_with_llm(llm_task, current_time)

        print(f"[{current_time:.1f}] {agent.agent_type.upper()} Result: {result.content[:100]}...")

        # Share result with central post
        message = agent.share_result_to_central(result)
        central_post.queue_message(message)

        # Store in knowledge base
        central_post.store_agent_result_as_knowledge(
            agent_id=agent.agent_id,
            content=result.content,
            confidence=result.confidence,
            domain="programming_languages"
        )


def demonstrate_confidence_monitoring(central_post: CentralPost) -> None:
    """Demonstrate confidence monitoring and dynamic spawning."""
    print("\n--- Confidence Monitoring Demo ---")

    # Simulate some messages with varying confidence
    messages = [
        Message("research_001", MessageType.STATUS_UPDATE,
               {"confidence": 0.8, "agent_type": "research", "position_info": {"depth_ratio": 0.2}},
               time.time()),
        Message("analysis_001", MessageType.STATUS_UPDATE,
               {"confidence": 0.6, "agent_type": "analysis", "position_info": {"depth_ratio": 0.5}},
               time.time()),
        Message("research_002", MessageType.STATUS_UPDATE,
               {"confidence": 0.4, "agent_type": "research", "position_info": {"depth_ratio": 0.3}},
               time.time())
    ]

    # Note: Confidence monitoring is part of DynamicSpawning, not directly on CentralPost
    # For demo purposes, we'll show basic confidence calculation
    confidences = [msg.content.get("confidence", 0.5) for msg in messages]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"Team confidence: {avg_confidence:.2f}")
    print(f"Individual confidences: {confidences}")

    # Check if confidence is low enough to trigger spawning
    should_spawn = avg_confidence < 0.7
    print(f"Should spawn additional agents: {should_spawn}")


def main():
    """Main demonstration workflow."""
    print("=== Felix Framework Example Workflow ===\n")

    # Step 1: Initialize core components
    print("1. Initializing Felix components...")

    # Create helix geometry (top_radius=2.0, bottom_radius=0.5, height=10.0, turns=3)
    helix = HelixGeometry(2.0, 0.5, 10.0, 3)
    print(f"   Helix geometry: {helix}")

    # Create mock LLM client
    llm_client = MockLMStudioClient()

    # Create central post with memory enabled
    central_post = CentralPost(
        max_agents=10,
        enable_metrics=True,
        enable_memory=True,
        memory_db_path="felix_memory.db",
        llm_client=llm_client  # For CentralPost synthesis
    )

    # Create token budget manager
    token_budget = TokenBudgetManager()

    # Create agent factory (disable dynamic spawning to avoid import issues)
    agent_factory = AgentFactory(
        helix=helix,
        llm_client=llm_client,
        token_budget_manager=token_budget,
        enable_dynamic_spawning=False
    )

    print("   Components initialized successfully\n")

    # Step 2: Create initial agent team
    print("2. Creating initial agent team...")

    # Create specialized agents with different spawn times
    research_agent = ResearchAgent(
        agent_id="research_001",
        spawn_time=0.1,  # Early spawn
        helix=helix,
        llm_client=llm_client,
        research_domain="technical",
        token_budget_manager=token_budget,
        max_tokens=800
    )

    analysis_agent = AnalysisAgent(
        agent_id="analysis_001",
        spawn_time=0.4,  # Mid spawn
        helix=helix,
        llm_client=llm_client,
        analysis_type="critical",
        token_budget_manager=token_budget,
        max_tokens=800
    )

    critic_agent = CriticAgent(
        agent_id="critic_001",
        spawn_time=0.5,  # Mid spawn for validation
        helix=helix,
        llm_client=llm_client,
        review_focus="accuracy",
        token_budget_manager=token_budget,
        max_tokens=800
    )

    agents = [research_agent, analysis_agent, critic_agent]

    # Register agents with central post
    for agent in agents:
        connection_id = central_post.register_agent(agent)
        print(f"   Registered {agent.agent_type} agent: {agent.agent_id} (conn: {connection_id})")

    print(f"   Team size: {len(agents)} agents\n")

    # Step 3: Create and process task
    print("3. Processing task through agent progression...")

    task = create_task("Evaluate Python programming language for modern software development projects")
    print(f"   Task: {task['description']}\n")

    # Simulate time progression and agent spawning
    current_time = 0.0
    time_step = 0.1

    while current_time <= 1.0:
        # Update each agent
        for agent in agents:
            simulate_agent_progression(agent, task, current_time, central_post)

        # Process any pending messages
        while central_post.has_pending_messages():
            message = central_post.process_next_message()
            if message:
                print(f"[{current_time:.1f}] Processed message from {message.sender_id}: {message.message_type.value}")

        current_time += time_step

    print()

    # Step 4: Demonstrate confidence monitoring
    demonstrate_confidence_monitoring(central_post)

    # Step 5: Show memory contents
    print("\n--- Memory System Demo ---")
    memory_summary = central_post.get_memory_summary()
    print(f"Knowledge entries: {memory_summary['knowledge_entries']}")
    print(f"Task executions: {memory_summary.get('task_executions', 0)}")

    # Retrieve stored knowledge
    if central_post.knowledge_store:
        from src.memory.knowledge_store import KnowledgeQuery
        query = central_post.knowledge_store.retrieve_knowledge(
            KnowledgeQuery(
                domains=["programming_languages"],
                limit=3
            )
        )

        print(f"Retrieved {len(query)} knowledge entries:")
        for entry in query:
            print(f"  - {entry.knowledge_type.value}: {entry.content.get('result', '')[:80]}...")
    else:
        print("Knowledge store not available")

    # Step 6: Show final results
    print("\n--- Final Results ---")
    print("Agent final positions and states:")
    for agent in agents:
        position = agent.get_position(1.0)  # Final position
        if position:
            print(f"  {agent.agent_id}: pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), state={agent.state}")
        else:
            print(f"  {agent.agent_id}: position unavailable, state={agent.state}")

    print("\nCentral post metrics:")
    metrics = central_post.get_performance_summary()
    print(f"  Messages processed: {metrics['total_messages_processed']}")
    print(f"  Active connections: {metrics['active_connections']}")
    print(f"  Uptime: {metrics['uptime']:.2f}s")
    # Step 7: Dynamic spawning demonstration
    print("\n--- Dynamic Spawning Demo ---")
    current_agents = [agent for agent in agents if agent.state == "active"]
    if agent_factory.dynamic_spawner:
        new_agents = agent_factory.assess_team_needs([], current_time, current_agents)
        print(f"Dynamic spawning recommended {len(new_agents)} new agents")

        if new_agents:
            print("New agents would be:")
            for agent in new_agents:
                print(f"  - {agent.agent_type} agent: {agent.agent_id}")
    else:
        print("Dynamic spawning not enabled in this demo")

    print("\n=== Workflow Complete ===")
    print("This example demonstrated:")
    print("- Component initialization and agent spawning")
    print("- Position-aware task processing along helical geometry")
    print("- Message passing and result sharing")
    print("- Memory storage and retrieval")
    print("- Confidence monitoring and dynamic spawning logic")
    print("- Performance metrics collection")


if __name__ == "__main__":
    main()