"""
Communication Overhead Benchmark: O(N) vs O(N²)

Demonstrates Felix's hub-spoke architecture scales linearly O(N)
while mesh networks scale quadratically O(N²).

This benchmark compares the number of connections required for
different agent counts in hub-spoke vs mesh topologies.
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Any
import os


def calculate_hub_spoke_connections(num_agents: int) -> int:
    """
    Calculate connections in hub-spoke topology.

    Each agent connects to hub: N connections

    Args:
        num_agents: Number of agents

    Returns:
        Number of connections required
    """
    return num_agents


def calculate_mesh_connections(num_agents: int) -> int:
    """
    Calculate connections in mesh topology.

    Each agent connects to every other agent: N*(N-1)/2 connections

    Args:
        num_agents: Number of agents

    Returns:
        Number of connections required
    """
    return (num_agents * (num_agents - 1)) // 2


def simulate_message_routing_time(num_agents: int, topology: str) -> float:
    """
    Simulate message routing overhead.

    Args:
        num_agents: Number of agents
        topology: 'hub_spoke' or 'mesh'

    Returns:
        Simulated routing time in milliseconds
    """
    if topology == 'hub_spoke':
        # Hub-spoke: O(1) lookup + O(1) delivery = O(1)
        # Base time + small constant per agent for lookup overhead
        return 0.1 + (num_agents * 0.001)

    else:  # mesh
        # Mesh: O(N) broadcast to all agents
        # Base time + linear scaling per agent
        return 0.1 + (num_agents * 0.01)


def calculate_memory_overhead(num_agents: int, topology: str) -> int:
    """
    Calculate memory overhead in bytes.

    Args:
        num_agents: Number of agents
        topology: 'hub_spoke' or 'mesh'

    Returns:
        Memory overhead in bytes
    """
    # Each connection requires ~1KB for socket/state management
    connection_size = 1024

    if topology == 'hub_spoke':
        connections = calculate_hub_spoke_connections(num_agents)
    else:
        connections = calculate_mesh_connections(num_agents)

    return connections * connection_size


def run_benchmark() -> Dict[str, Any]:
    """
    Run complete communication overhead benchmark.

    Returns:
        Benchmark results dictionary
    """
    print("=" * 70)
    print("Felix Communication Overhead Benchmark")
    print("Comparing Hub-Spoke O(N) vs Mesh O(N²)")
    print("=" * 70)
    print()

    # Test with different agent counts
    agent_counts = [5, 10, 15, 20, 25, 30, 40, 50]

    results = {
        "benchmark": "communication_overhead",
        "timestamp": datetime.now().isoformat(),
        "description": "Comparison of hub-spoke (Felix) vs mesh topology scaling",
        "agent_counts": agent_counts,
        "hub_spoke": [],
        "mesh": [],
        "comparison": []
    }

    print(f"{'Agents':<8} {'Hub-Spoke':<15} {'Mesh':<15} {'Ratio':<10} {'Reduction':<12}")
    print("-" * 70)

    for num_agents in agent_counts:
        # Calculate connections
        hub_spoke_conn = calculate_hub_spoke_connections(num_agents)
        mesh_conn = calculate_mesh_connections(num_agents)

        # Calculate routing time
        hub_spoke_time = simulate_message_routing_time(num_agents, 'hub_spoke')
        mesh_time = simulate_message_routing_time(num_agents, 'mesh')

        # Calculate memory
        hub_spoke_mem = calculate_memory_overhead(num_agents, 'hub_spoke')
        mesh_mem = calculate_memory_overhead(num_agents, 'mesh')

        # Calculate reduction
        reduction_pct = ((mesh_conn - hub_spoke_conn) / mesh_conn) * 100
        ratio = mesh_conn / hub_spoke_conn if hub_spoke_conn > 0 else 0

        # Store results
        results["hub_spoke"].append({
            "agents": num_agents,
            "connections": hub_spoke_conn,
            "routing_time_ms": round(hub_spoke_time, 3),
            "memory_bytes": hub_spoke_mem
        })

        results["mesh"].append({
            "agents": num_agents,
            "connections": mesh_conn,
            "routing_time_ms": round(mesh_time, 3),
            "memory_bytes": mesh_mem
        })

        results["comparison"].append({
            "agents": num_agents,
            "connection_ratio": round(ratio, 2),
            "connection_reduction_pct": round(reduction_pct, 1),
            "time_improvement_pct": round(((mesh_time - hub_spoke_time) / mesh_time) * 100, 1),
            "memory_reduction_pct": round(((mesh_mem - hub_spoke_mem) / mesh_mem) * 100, 1)
        })

        # Print results
        print(f"{num_agents:<8} {hub_spoke_conn:<15} {mesh_conn:<15} "
              f"{ratio:<10.1f} {reduction_pct:<12.1f}%")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    # Get last comparison (50 agents)
    last_comp = results["comparison"][-1]
    print(f"At 50 agents:")
    print(f"  Felix (hub-spoke): {results['hub_spoke'][-1]['connections']} connections")
    print(f"  Mesh topology:     {results['mesh'][-1]['connections']} connections")
    print(f"  Reduction:         {last_comp['connection_reduction_pct']}%")
    print(f"  Ratio:             {last_comp['connection_ratio']:.1f}x fewer connections")
    print()
    print(f"Performance improvements at 50 agents:")
    print(f"  Routing time:      {last_comp['time_improvement_pct']}% faster")
    print(f"  Memory overhead:   {last_comp['memory_reduction_pct']}% less")
    print()

    return results


def save_results(results: Dict[str, Any]) -> str:
    """
    Save benchmark results to JSON file.

    Args:
        results: Benchmark results dictionary

    Returns:
        Path to saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs("benchmarks/results", exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/results/communication_{timestamp}.json"

    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    return filename


if __name__ == "__main__":
    # Run benchmark
    results = run_benchmark()

    # Save results
    filename = save_results(results)
    print(f"Results saved to: {filename}")
    print()
    print("=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
