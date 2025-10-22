"""
Markdown Formatter for Felix Workflow Results

Provides utilities to format Felix workflow outputs into clean,
professional markdown documents suitable for documentation,
reports, and human-readable archives.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import textwrap


def format_synthesis_markdown(result: Dict[str, Any]) -> str:
    """
    Format Felix workflow results as professional markdown.

    Args:
        result: Workflow result dictionary containing:
            - task: Task description
            - status: Workflow status
            - agents_spawned: List of agent IDs
            - completed_agents: Number of completed agents
            - centralpost_synthesis: Synthesis output from CentralPost
            - llm_responses: List of agent outputs

    Returns:
        Formatted markdown string
    """
    lines = []

    # Title and timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Felix Workflow Synthesis Report")
    lines.append(f"\n**Generated:** {timestamp}\n")
    lines.append("---\n")

    # Metadata section
    lines.append("## Workflow Metadata\n")
    lines.append(f"**Task:** {result.get('task', 'N/A')}\n")
    lines.append(f"**Status:** {result.get('status', 'unknown').upper()}\n")
    lines.append(f"**Agents Spawned:** {len(result.get('agents_spawned', []))}\n")
    lines.append(f"**Completed Agents:** {result.get('completed_agents', 0)}\n")

    # List spawned agents
    agents_spawned = result.get('agents_spawned', [])
    if agents_spawned:
        lines.append(f"\n**Agent Team:**")
        for agent_id in agents_spawned:
            lines.append(f"- `{agent_id}`")

    lines.append("\n---\n")

    # Final Synthesis Section (CentralPost)
    final_synthesis = result.get("centralpost_synthesis")
    if final_synthesis:
        lines.append("## Executive Summary\n")
        lines.append("**Synthesis Method:** CentralPost (Smart Hub Synthesis)\n")

        # Synthesis metrics table
        lines.append("\n### Synthesis Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Confidence | {final_synthesis.get('confidence', 0.0):.2f} |")
        lines.append(f"| Temperature | {final_synthesis.get('temperature', 0.0):.2f} |")
        lines.append(f"| Synthesis Time | {final_synthesis.get('synthesis_time', 0):.2f}s |")
        lines.append(f"| Tokens Used | {final_synthesis.get('tokens_used', 0)} / {final_synthesis.get('max_tokens', 0)} |")
        lines.append(f"| Agents Synthesized | {final_synthesis.get('agents_synthesized', 0)} |")
        lines.append(f"| Avg Agent Confidence | {final_synthesis.get('avg_agent_confidence', 0):.2f} |")

        # Check if fallback was used
        if final_synthesis.get('fallback'):
            lines.append(f"| Fallback Mode | Yes (from `{final_synthesis.get('fallback_agent_id')}`) |")

        lines.append("\n### Synthesis Output\n")

        # Format synthesis content with proper line breaks
        synthesis_content = final_synthesis.get("synthesis_content", "")
        lines.append(f"{synthesis_content}\n")

        lines.append("\n---\n")

    # Agent Outputs Section
    llm_responses = result.get("llm_responses", [])
    if llm_responses:
        lines.append("## Agent Contributions\n")
        lines.append(f"Detailed outputs from {len(llm_responses)} agent processing cycles.\n")

        for i, resp in enumerate(llm_responses, 1):
            agent_id = resp.get('agent_id', 'unknown')
            agent_type = resp.get('agent_type', 'unknown')
            confidence = resp.get('confidence', 0.0)
            response_text = resp.get('response', '')
            checkpoint = resp.get('checkpoint', 'N/A')
            progress = resp.get('progress', 0.0)

            # Agent header
            lines.append(f"\n### Agent {i}: {agent_type.capitalize()}\n")
            lines.append(f"**ID:** `{agent_id}`  ")
            lines.append(f"**Confidence:** {confidence:.2f}  ")
            lines.append(f"**Checkpoint:** {checkpoint}  ")
            lines.append(f"**Progress:** {progress:.2f}\n")

            # Use collapsible section for agent output
            lines.append("<details>")
            lines.append(f"<summary>View {agent_type.capitalize()} Output</summary>\n")

            # Output content
            lines.append(f"{response_text}\n")

            lines.append("</details>\n")

    # Performance Summary
    lines.append("\n---\n")
    lines.append("## System Performance\n")

    # Calculate some basic stats
    if llm_responses:
        confidences = [r.get('confidence', 0.0) for r in llm_responses]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Agents | {len(result.get('agents_spawned', []))} |")
        lines.append(f"| Processing Cycles | {len(llm_responses)} |")
        lines.append(f"| Avg Confidence | {avg_confidence:.2f} |")
        lines.append(f"| Min Confidence | {min_confidence:.2f} |")
        lines.append(f"| Max Confidence | {max_confidence:.2f} |")

    # Footer
    lines.append("\n---\n")
    lines.append("*Report generated by Felix Multi-Agent Framework*  ")
    lines.append("*Framework: Helical Agent Progression with CentralPost Synthesis*\n")

    return "\n".join(lines)


def format_synthesis_markdown_detailed(
    result: Dict[str, Any],
    agent_manager=None,
    include_prompts: bool = False
) -> str:
    """
    Format Felix workflow results with comprehensive agent details.

    This extended version includes full agent metrics, prompts, and
    collaborative context information when available.

    Args:
        result: Workflow result dictionary
        agent_manager: Optional AgentManager to fetch detailed agent data
        include_prompts: Whether to include system/user prompts

    Returns:
        Formatted markdown string with detailed information
    """
    lines = []

    # Start with basic format
    basic_md = format_synthesis_markdown(result)
    lines.append(basic_md)

    # Add detailed agent information if agent_manager provided
    if agent_manager and include_prompts:
        lines.append("\n---\n")
        lines.append("## Detailed Agent Metrics\n")

        llm_responses = result.get("llm_responses", [])
        for i, resp in enumerate(llm_responses, 1):
            agent_id = resp.get('agent_id', 'unknown')
            agent_type = resp.get('agent_type', 'unknown')

            # Get comprehensive metrics from agent_manager
            agent_data = agent_manager.get_agent_output(agent_id)

            if agent_data:
                lines.append(f"\n### Agent {i} Details: `{agent_id}`\n")

                # Position and phase info
                position_info = agent_data.get('position_info', {})
                depth_ratio = position_info.get('depth_ratio', 0)
                phase = "Exploration" if depth_ratio < 0.3 else ("Analysis" if depth_ratio < 0.7 else "Synthesis")

                lines.append("#### Processing Context\n")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Agent Type | {agent_type.capitalize()} |")
                lines.append(f"| Depth Ratio | {depth_ratio:.2f} |")
                lines.append(f"| Phase | {phase} |")
                lines.append(f"| Processing Time | {agent_data.get('processing_time', 0):.2f}s |")
                lines.append(f"| Temperature | {agent_data.get('temperature', 0):.2f} |")
                lines.append(f"| Tokens Used | {agent_data.get('tokens_used', 0)} / {agent_data.get('token_budget', 0)} |")
                lines.append(f"| Model | {agent_data.get('model', 'unknown')} |")
                lines.append(f"| Collaborative Context | {agent_data.get('collaborative_count', 0)} prior outputs |")

                # Include prompts if requested
                if include_prompts:
                    system_prompt = agent_data.get('system_prompt', '')
                    user_prompt = agent_data.get('user_prompt', '')

                    if system_prompt:
                        lines.append("\n#### System Prompt\n")
                        lines.append("```")
                        lines.append(system_prompt)
                        lines.append("```\n")

                    if user_prompt:
                        lines.append("#### User Prompt\n")
                        lines.append("```")
                        lines.append(user_prompt)
                        lines.append("```\n")

    return "\n".join(lines)


def save_markdown_to_file(
    markdown_content: str,
    output_dir: str = "outputs",
    filename_prefix: str = "felix_synthesis",
    create_dir: bool = True
) -> str:
    """
    Save markdown content to a timestamped file.

    Args:
        markdown_content: Formatted markdown string
        output_dir: Directory to save file (default: "outputs")
        filename_prefix: Prefix for filename (default: "felix_synthesis")
        create_dir: Whether to create output directory if missing

    Returns:
        Path to saved file
    """
    import os

    # Create directory if needed
    if create_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    return filepath
