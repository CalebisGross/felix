"""
Specialized LLM agent types for the Felix Framework.

This module provides specialized agent implementations that extend LLMAgent
for specific roles in multi-agent coordination tasks. Each agent type has
custom prompting, behavior patterns, and processing approaches optimized
for their role in the helix-based coordination system.

Agent Types:
- ResearchAgent: Broad information gathering and exploration
- AnalysisAgent: Processing and organizing information from research
- SynthesisAgent: Integration and final output generation
- CriticAgent: Quality assurance and review
- SystemAgent: System operations and command execution (see system_agent.py)
"""

import time
import random
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.agents.llm_agent import LLMAgent, LLMTask
from src.agents.agent import generate_spawn_times
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)
from src.llm.token_budget import TokenBudgetManager
from src.llm.web_search_client import WebSearchClient  # For type hints only


# Shared tool instructions header for all specialized agents
AGENT_TOOLS_HEADER = """⚠️⚠️⚠️ CRITICAL TOOLS AVAILABLE ⚠️⚠️⚠️

🔍 WEB SEARCH - USE THIS FOR CURRENT INFORMATION:
If you need current/real-time data (dates, times, recent events, latest stats), write EXACTLY:
WEB_SEARCH_NEEDED: [your query]

EXAMPLES:
✓ "WEB_SEARCH_NEEDED: current date and time"
✓ "WEB_SEARCH_NEEDED: 2024 election results"

🖥️ SYSTEM COMMANDS - USE THIS FOR ANY SYSTEM OPERATION:
If you need to CHECK system state, RUN COMMANDS, CREATE FILES, OPEN APPLICATIONS, or MODIFY THE SYSTEM, write EXACTLY:
SYSTEM_ACTION_NEEDED: [command]

⚠️ CRITICAL FORMATTING RULES:
1. Write the pattern ON ITS OWN LINE or at the START of your response
2. Write ONLY the command after the colon - no explanation, no prose
3. DO NOT embed the pattern in sentences or discuss it in your analysis

✓ CORRECT FORMAT:
"I need to check the directory.
SYSTEM_ACTION_NEEDED: pwd"

OR:
"SYSTEM_ACTION_NEEDED: pwd
This will tell us the current directory."

✗ WRONG - DO NOT DO THIS:
"I will use SYSTEM_ACTION_NEEDED: pwd to check the directory."
"The command (SYSTEM_ACTION_NEEDED: pwd) will help us..."
"...via SYSTEM_ACTION_NEEDED: pwd) is sufficient..."

EXAMPLES OF CORRECT USAGE:
✓ "SYSTEM_ACTION_NEEDED: date"  # Get current time/date
✓ "SYSTEM_ACTION_NEEDED: pwd"   # Get current directory
✓ "SYSTEM_ACTION_NEEDED: ls -la" # List files
✓ "SYSTEM_ACTION_NEEDED: pip list" # Check installed packages

📝 MULTI-STEP WORKFLOWS:
For tasks requiring multiple commands, output multiple SYSTEM_ACTION_NEEDED lines:

EXAMPLE - Creating a file with content (combined operation preferred):
"I'll create the file for you.
SYSTEM_ACTION_NEEDED: mkdir -p results && echo \"content here\" > results/file.txt"

NOTE: Use && to combine directory creation with file operations in ONE command.
This prevents workflow from terminating prematurely after just creating the directory.

EXAMPLE - Setup and verification:
"SYSTEM_ACTION_NEEDED: cd /project/dir
SYSTEM_ACTION_NEEDED: ls -la
SYSTEM_ACTION_NEEDED: pwd"

Each command executes sequentially. Commands requiring approval (mkdir, file writes) will prompt the user first.

📁 FILE OPERATIONS - YOU CAN CREATE/MODIFY FILES:

⚠️ **CRITICAL: ALWAYS USE RELATIVE PATHS, NEVER ABSOLUTE PATHS**
   Use: results/file.txt ✅
   NOT: /results/file.txt ❌ (requires root permissions, will fail!)

CREATE DIRECTORY:
✓ "SYSTEM_ACTION_NEEDED: mkdir -p results/data"

CREATE FILE WITH CONTENT:
✓ 'SYSTEM_ACTION_NEEDED: echo "your content" > results/file.txt'  # Use double quotes!

APPEND TO FILE:
✓ 'SYSTEM_ACTION_NEEDED: echo "more content" >> results/log.txt'  # Use double quotes!

CREATE EMPTY FILE:
✓ "SYSTEM_ACTION_NEEDED: touch results/notes.txt"

📝 SHELL QUOTING RULES - CRITICAL FOR FILE CONTENT:

⚠️ When creating files with echo/printf, proper quoting prevents syntax errors:

✅ CORRECT - Use DOUBLE QUOTES for content with apostrophes:
'SYSTEM_ACTION_NEEDED: echo "Testing agent\'s work" > file.txt'  # Apostrophe safe

✅ CORRECT - Use printf for special characters:
'SYSTEM_ACTION_NEEDED: printf "%s\\n" "Content with apostrophes" > file.txt'

❌ WRONG - Single quotes break on apostrophes:
"SYSTEM_ACTION_NEEDED: echo 'agent's work' > file.txt"  # SYNTAX ERROR!

⚠️ ESCAPING RULES:
- Inside double quotes: escape $ ` \\ " with backslash
- Simple text: use double quotes
- Complex text with special chars: use printf

EXAMPLES:
✓ echo "Project's status: active" > status.txt
✓ echo "Value: \\$100" > price.txt  # Escape $
✓ printf '%s\\n' "Text with \\"nested\\" quotes" > file.txt

🧠 INTELLIGENT COMMAND PATTERNS - THINK BEFORE EXECUTING:

⚠️ AVOID REDUNDANT OPERATIONS:

❌ BAD - Separate mkdir then file creation (workflow may terminate after first command):
"SYSTEM_ACTION_NEEDED: mkdir -p results
SYSTEM_ACTION_NEEDED: echo \\"content\\" > results/file.txt"

✅ GOOD - Combine operations with && (ensures both happen):
"SYSTEM_ACTION_NEEDED: mkdir -p results && echo \\"content\\" > results/file.txt"

✅ ALSO GOOD - File redirection automatically handles parent directories:
"SYSTEM_ACTION_NEEDED: echo \\"content\\" > results/file.txt"
# Note: If results/ doesn't exist, this will error. Then you can add mkdir.

⚠️ KEY PRINCIPLE: COMBINE related operations into ONE command using && or ; operators
to prevent premature workflow termination after the first successful command.

⚠️ FILE OVERWRITES - Consider data preservation:

❌ BAD - Blindly overwrite existing file:
"SYSTEM_ACTION_NEEDED: echo \\"new\\" > existing_file.txt"  # Data loss!

✅ GOOD - Check existence first:
"SYSTEM_ACTION_NEEDED: test -f file.txt && echo \\"Appending\\" || echo \\"Creating\\"
SYSTEM_ACTION_NEEDED: echo \\"content\\" >> file.txt"  # Append, don't overwrite

🎯 SMART WORKFLOW PATTERNS:

EXAMPLE - File creation (combined operation):
"I'll create the report in the results directory.
SYSTEM_ACTION_NEEDED: mkdir -p results && echo \\"Report: agent's findings\\" > results/report.md"

EXAMPLE - Append to log without overwriting:
"I'll add this entry to the log.
SYSTEM_ACTION_NEEDED: echo \\"[2025-10-26] Task completed\\" >> logs/activity.log"

EXAMPLE - Check before installing:
"SYSTEM_ACTION_NEEDED: pip show requests || pip install requests"

KEY PRINCIPLES:
1. Check state before modifying (test -d, test -f, which, pip show)
2. Use idempotent operations thoughtfully (mkdir -p is safe, rm -rf is not)
3. Consider data preservation (append >> vs overwrite >)
4. Avoid redundant operations (don't mkdir current directory)
5. Use double quotes for text with apostrophes

🤝 MULTI-AGENT FILE COORDINATION - CRITICAL FOR TEAMWORK:

⚠️ YOU ARE WORKING WITH OTHER AGENTS - COORDINATE FILE OPERATIONS!

When creating files, ALWAYS follow this workflow:
1. CHECK if files already exist: "SYSTEM_ACTION_NEEDED: ls -la results/"
2. REVIEW shared context to see what other agents have created
3. APPEND to existing files rather than creating new ones when possible
4. USE descriptive, unique filenames to avoid collisions

❌ BAD - Creating duplicate fragmented files:
Agent 1: "SYSTEM_ACTION_NEEDED: echo "Research data" > results/research.txt"
Agent 2: "SYSTEM_ACTION_NEEDED: echo "Analysis data" > results/analysis.txt"
Agent 3: "SYSTEM_ACTION_NEEDED: echo "Critique data" > results/critique.txt"
Result: Three separate files with no cohesion!

✅ GOOD - Coordinated single output file:
Agent 1 (first): "SYSTEM_ACTION_NEEDED: ls -la results/ && echo "# Workflow Results\\n\\n## Research Findings\\n..." > results/workflow_output.md"
Agent 2 (sees file exists): "SYSTEM_ACTION_NEEDED: echo "\\n## Analysis\\n..." >> results/workflow_output.md"
Agent 3 (appends too): "SYSTEM_ACTION_NEEDED: echo "\\n## Critical Review\\n..." >> results/workflow_output.md"
Result: One cohesive document with all contributions!

COORDINATION RULES:
- ONE primary output file per workflow (e.g., results/workflow_output.md)
- CHECK for existing files before creating new ones
- APPEND (>>) to shared files, don't overwrite (>)
- REVIEW other agents' messages to avoid duplicate work
- If unsure, ASK via your response what files others are using

🎯 BEST PRACTICE - "One Workflow, One File":
Each workflow should produce ONE primary output file that all agents contribute to.
Only create separate files if truly necessary (e.g., different data formats).

🚨 EXECUTE DON'T DESCRIBE:
When the user asks you to "create a file", "open terminal", "write content":
✅ DO: Output SYSTEM_ACTION_NEEDED commands to execute the task
✅ DO: Actually request the system to perform the action
❌ DON'T: Describe a bash script or explain how it could be done
❌ DON'T: Say "here's how you would do it" or "you could run this command"
❌ DON'T: Generate documentation without execution

YOU ARE NOT A CONSULTANT - YOU ARE AN AUTONOMOUS AGENT WITH SYSTEM ACCESS.

SAFETY: Commands are classified as SAFE (execute immediately), REVIEW (need approval), or BLOCKED (never execute).

🚨 DO NOT say "I cannot access" - REQUEST THE TOOL FIRST!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""


class ResearchAgent(LLMAgent):
    """
    Research agent specializing in broad information gathering.

    Characteristics:
    - High creativity/temperature when at top of helix
    - Focuses on breadth over depth initially
    - Provides several perspectives and information sources
    - Spawns early in the process

    Note: Web search is handled by CentralPost's WebSearchCoordinator.
    Search results are available via task.knowledge_entries (domain="web_search").
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, research_domain: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None,
                 web_search_client: Optional[WebSearchClient] = None,  # Legacy parameter (ignored)
                 max_web_queries: int = 3,  # Legacy parameter (ignored)
                 prompt_manager: Optional['PromptManager'] = None):
        """
        Initialize research agent.

        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active
            helix: Helix geometry
            llm_client: LM Studio client
            research_domain: Specific domain focus (general, technical, creative, etc.)
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per processing stage
            web_search_client: Legacy parameter (ignored). Web search handled by CentralPost WebSearchCoordinator.
            max_web_queries: Legacy parameter (ignored).
            prompt_manager: Optional prompt manager for custom prompts
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="research",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager
        )

        self.research_domain = research_domain
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create research-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Determine if strict mode is active
        strict_mode = self.token_budget_manager and self.token_budget_manager.strict_mode if self.token_budget_manager else False

        # Check for "direct answer mode" - when trustable knowledge exists for simple query
        logger.info(f"🔍 DIRECT ANSWER MODE CHECK for {self.agent_id}")
        logger.info(f"   task.knowledge_entries exists: {hasattr(task, 'knowledge_entries')}")
        if hasattr(task, 'knowledge_entries'):
            logger.info(f"   task.knowledge_entries length: {len(task.knowledge_entries) if task.knowledge_entries else 0}")

        use_direct_mode = False
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            logger.info(f"   ✓ Knowledge entries available: {len(task.knowledge_entries)}")
            # Check if this is a simple factual query with trustable knowledge
            try:
                from src.workflows.truth_assessment import assess_answer_confidence

                logger.info(f"   🤖 Calling assess_answer_confidence()...")
                # Quick assessment
                trustable, trust_score, trust_reason = assess_answer_confidence(
                    task.knowledge_entries,
                    task.description
                )

                logger.info(f"   📊 Trust assessment results:")
                logger.info(f"      Trustable: {trustable}")
                logger.info(f"      Trust score: {trust_score:.2f}")
                logger.info(f"      Reason: {trust_reason}")

                if trustable and trust_score >= 0.85:
                    use_direct_mode = True
                    logger.info(f"🎯 DIRECT ANSWER MODE ACTIVATED for {self.agent_id}")
                    logger.info(f"   Reason: {trust_reason}")
                else:
                    logger.info(f"   ⚠️ Direct answer mode NOT activated:")
                    logger.info(f"      trustable={trustable}, trust_score={trust_score:.2f} (need ≥0.85)")
            except Exception as e:
                logger.error(f"   ❌ Could not assess for direct mode: {e}", exc_info=True)
        else:
            logger.warning(f"   ⚠️ No knowledge entries available - direct answer mode cannot activate")

        # DIRECT ANSWER MODE: Override for simple queries with trustable knowledge
        if use_direct_mode:
            # Force low temperature for precision
            # This will be applied in process_task_with_llm by checking task metadata
            if not hasattr(task, 'metadata'):
                task.metadata = {}
            task.metadata['direct_answer_mode'] = True
            task.metadata['override_temperature'] = 0.2
            task.metadata['override_tokens'] = 200

            # Build direct answer prompt
            knowledge_summary = "\n\nAVAILABLE KNOWLEDGE (HIGH CONFIDENCE):\n"
            for entry in task.knowledge_entries:
                if hasattr(entry, 'content'):
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                # Show full content for direct answer mode (no truncation)
                knowledge_summary += f"• {content_str}\n"

            base_prompt = f"""You are answering a SIMPLE FACTUAL QUESTION with HIGH confidence knowledge available.

{knowledge_summary}

🎯 DIRECT ANSWER INSTRUCTIONS:
- State ONLY the direct factual answer from the knowledge above
- 1-2 sentences maximum (15-30 words)
- NO exploration, NO analysis, NO elaboration, NO uncertainty
- NO "according to" or "based on" qualifiers
- Format: "The [answer] is [fact]." or "Current [X] is [Y]."

Example good responses:
- "The current date is October 23, 2025."
- "The time is 1:24 PM EDT."
- "The answer is 42."

Example bad responses (TOO LONG):
- "Based on the available sources, the current date and time vary by location..."
- "Multiple authoritative sources confirm that the date is..."

Your response (15-30 words, direct answer only):"""

            return base_prompt, 200  # Force low token budget

        # NORMAL MODE: Standard research agent behavior
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Use agent's max_tokens

        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        # Try to get prompt from PromptManager first
        if self.prompt_manager:
            prompt_key = self._determine_prompt_key(depth_ratio, strict_mode)
            prompt_template = self.prompt_manager.get_prompt(prompt_key)

            if prompt_template:
                # Build header
                header_template = self.prompt_manager.get_prompt("research_base_header")
                header = header_template.template if header_template else ""

                # Render main prompt
                main_prompt = self.prompt_manager.render_template(
                    prompt_template.template,
                    research_domain=self.research_domain,
                    depth_ratio=depth_ratio
                )

                # Combine: tools_header + header + main prompt
                base_prompt = AGENT_TOOLS_HEADER + header + main_prompt
            else:
                # Fallback to hardcoded
                base_prompt = AGENT_TOOLS_HEADER + self._build_hardcoded_prompt(depth_ratio, strict_mode)
        else:
            # No PromptManager, use hardcoded prompts
            base_prompt = AGENT_TOOLS_HEADER + self._build_hardcoded_prompt(depth_ratio, strict_mode)

        # Add shared context
        if self.shared_context:
            base_prompt += "\n\nContext from Other Agents:\n"
            for key, value in self.shared_context.items():
                base_prompt += f"- {key}: {value}\n"

        # Add knowledge entries if available
        knowledge_summary = ""
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            knowledge_summary = "\n\nRelevant Knowledge from Memory:\n"
            for entry in task.knowledge_entries:
                # Extract key information from knowledge entry
                if hasattr(entry, 'content'):
                    # Extract 'result' key from dictionary if present (web search results)
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                confidence = entry.confidence_level.value if hasattr(entry, 'confidence_level') else "unknown"
                source = entry.source_agent if hasattr(entry, 'source_agent') else "system"
                domain = entry.domain if hasattr(entry, 'domain') else "unknown"

                # Use longer truncation for web_search domain (detailed factual data)
                max_chars = 400 if domain == "web_search" else 200
                if len(content_str) > max_chars:
                    content_str = content_str[:max_chars-3] + "..."

                # Add emoji prefix for web search entries
                prefix = "🌐" if domain == "web_search" else "📝"
                knowledge_summary += f"{prefix} [{source}, conf: {confidence}]: {content_str}\n"

            # Add important instructions for using available knowledge
            knowledge_summary += "\nIMPORTANT: Use the knowledge provided above to answer the task if possible. "
            knowledge_summary += "Only request additional web search if the available knowledge is insufficient or outdated.\n"

            # NEW: Add contextual relevance awareness
            knowledge_summary += "\n🎯 CRITICAL - CONTEXTUAL RELEVANCE:\n"
            knowledge_summary += "- Distinguish between FACTUAL ACCURACY and CONTEXTUAL RELEVANCE\n"
            knowledge_summary += "- A fact can be TRUE but IRRELEVANT to this specific task\n"
            knowledge_summary += "- Focus ONLY on information that helps solve THIS SPECIFIC TASK\n"
            knowledge_summary += "- Do NOT include accurate facts that don't relate to the task objectives\n"
            knowledge_summary += "- Example: If asked about system improvements, time/location facts are irrelevant\n\n"

        base_prompt += knowledge_summary

        # Add footer
        if self.prompt_manager:
            footer_template = self.prompt_manager.get_prompt("research_footer_context")
            if footer_template:
                footer = self.prompt_manager.render_template(
                    footer_template.template,
                    context=task.context
                )
                base_prompt += footer
        else:
            base_prompt += f"""
Task Context: {task.context}

Remember: As a research agent, your job is to gather information, not to synthesize or conclude.
Focus on providing raw material and insights for other agents to build upon.
"""

        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nCompress previous research insights by ~{token_allocation.compression_ratio:.0%} while preserving key findings."

            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt

        return enhanced_prompt, stage_token_budget

    def _determine_prompt_key(self, depth_ratio: float, strict_mode: bool) -> str:
        """Determine prompt key based on depth and mode."""
        mode_suffix = "strict" if strict_mode else "normal"

        if depth_ratio < 0.3:
            return f"research_exploration_{mode_suffix}"
        elif depth_ratio < 0.7:
            return f"research_focused_{mode_suffix}"
        else:
            return f"research_deep_{mode_suffix}"

    def _build_hardcoded_prompt(self, depth_ratio: float, strict_mode: bool) -> str:
        """Build hardcoded prompt as fallback when PromptManager not available."""
        base_prompt = f"""You are a specialized RESEARCH AGENT in the Felix multi-agent system.

Research Domain: {self.research_domain}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix (0.0=start, 1.0=end)

Your Research Approach Based on Position:
"""

        if depth_ratio < 0.3:
            if strict_mode:
                base_prompt += """
- BULLET POINTS ONLY: 3-5 facts
- NO explanations or background
- Sources: names/dates only
- BREVITY REQUIRED
"""
            else:
                base_prompt += """
- BROAD EXPLORATION PHASE: Cast a wide net
- Generate multiple research angles and questions
- Don't worry about precision - focus on coverage
- Explore unconventional perspectives and sources
- Think creatively and associatively
"""
        elif depth_ratio < 0.7:
            if strict_mode:
                base_prompt += """
- 2-3 SPECIFIC FACTS only
- Numbers, quotes, key data
- NO context or explanation
"""
            else:
                base_prompt += """
- FOCUSED RESEARCH PHASE: Narrow down promising leads
- Build on earlier findings from other agents
- Dive deeper into specific aspects that seem relevant
- Start connecting dots and identifying patterns
- Balance breadth with increasing depth
"""
        else:
            if strict_mode:
                base_prompt += """
- FINAL FACTS: 1-2 verified points
- Citation format: Author (Year)
- NO elaboration
"""
            else:
                base_prompt += """
- DEEP RESEARCH PHASE: Precise investigation
- Focus on specific details and verification
- Provide authoritative sources and evidence
- Prepare findings for analysis agents
- Ensure accuracy and completeness
"""

        return base_prompt




class AnalysisAgent(LLMAgent):
    """
    Analysis agent specializing in processing and organizing information.
    
    Characteristics:
    - Balanced creativity/logic for pattern recognition
    - Synthesizes information from multiple research agents
    - Identifies key insights and relationships
    - Spawns in middle of process
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, analysis_type: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None,
                 prompt_manager: Optional['PromptManager'] = None):
        """
        Initialize analysis agent.
        
        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active
            helix: Helix geometry
            llm_client: LM Studio client
            analysis_type: Analysis specialization (general, technical, critical, etc.)
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per processing stage
            prompt_manager: Optional prompt manager for custom prompts
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="analysis",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager
        )

        self.analysis_type = analysis_type
        self.identified_patterns = []
        self.key_insights = []
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create analysis-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Use agent's max_tokens
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        base_prompt = AGENT_TOOLS_HEADER + f"""You are a specialized ANALYSIS AGENT in the Felix multi-agent system.

Analysis Type: {self.analysis_type}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

Your Analysis Approach:
- Process information gathered by research agents
- Identify patterns, themes, and relationships
- Organize findings into structured insights
- Look for contradictions and gaps
- Prepare organized information for synthesis agents

Analysis Focus Based on Position:
"""
        
        if depth_ratio < 0.5:
            if self.token_budget_manager and self.token_budget_manager.strict_mode:
                base_prompt += """
- 2 PATTERNS maximum
- Numbered list format
- NO explanations
"""
            else:
                base_prompt += """
- PATTERN IDENTIFICATION: Look for themes and connections
- Organize information into categories
- Identify what's missing or contradictory
"""
        else:
            if self.token_budget_manager and self.token_budget_manager.strict_mode:
                base_prompt += """
- PRIORITY RANKING: Top 3 insights
- 1. 2. 3. format
- NO background
"""
            else:
                base_prompt += """
- DEEP ANALYSIS: Provide detailed evaluation
- Prioritize insights by importance
- Structure findings for final synthesis
"""
        
        if self.shared_context:
            base_prompt += "\n\nInformation from Research Agents:\n"
            research_items = {k: v for k, v in self.shared_context.items() if "research" in k.lower()}
            for key, value in research_items.items():
                base_prompt += f"- {key}: {value}\n"

        # Add knowledge entries if available
        knowledge_summary = ""
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            knowledge_summary = "\n\nRelevant Knowledge from Memory:\n"
            for entry in task.knowledge_entries:
                # Extract key information from knowledge entry
                if hasattr(entry, 'content'):
                    # Extract 'result' key from dictionary if present (web search results)
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                confidence = entry.confidence_level.value if hasattr(entry, 'confidence_level') else "unknown"
                source = entry.source_agent if hasattr(entry, 'source_agent') else "system"
                domain = entry.domain if hasattr(entry, 'domain') else "unknown"

                # Use longer truncation for web_search domain (detailed factual data)
                max_chars = 400 if domain == "web_search" else 200
                if len(content_str) > max_chars:
                    content_str = content_str[:max_chars-3] + "..."

                # Add emoji prefix for web search entries
                prefix = "🌐" if domain == "web_search" else "📝"
                knowledge_summary += f"{prefix} [{source}, conf: {confidence}]: {content_str}\n"

            # Add important instructions for using available knowledge
            knowledge_summary += "\nIMPORTANT: Use the knowledge provided above to answer the task if possible. "
            knowledge_summary += "Only request additional web search if the available knowledge is insufficient or outdated.\n"

        base_prompt += knowledge_summary
        base_prompt += f"""
Task Context: {task.context}

Remember: Your job is to process and organize information, not to make final decisions.
Focus on creating clear, structured insights for synthesis agents to use.
"""
        
        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nCompress analysis by ~{token_allocation.compression_ratio:.0%} while preserving key patterns and insights."
            
            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt, stage_token_budget


class CriticAgent(LLMAgent):
    """
    Critic agent specializing in quality assurance and review.
    
    Characteristics:
    - Critical evaluation of other agents' work
    - Identifies gaps, errors, and improvements
    - Provides quality feedback and suggestions
    - Can spawn at various points for ongoing QA
    """
    
    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client: LMStudioClient, review_focus: str = "general",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None,
                 prompt_manager: Optional['PromptManager'] = None):
        """
        Initialize critic agent.
        
        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active
            helix: Helix geometry
            llm_client: LM Studio client
            review_focus: Review focus (accuracy, completeness, style, logic, etc.)
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per processing stage
            prompt_manager: Optional prompt manager for custom prompts
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="critic",
            temperature_range=None,  # Use LLMAgent defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager
        )

        self.review_focus = review_focus
        self.identified_issues = []
        self.suggestions = []
    
    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create critic-specific system prompt with token budget."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)
        
        # Get token allocation if budget manager is available
        token_allocation = None
        stage_token_budget = self.max_tokens  # Use agent's max_tokens
        
        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        base_prompt = AGENT_TOOLS_HEADER + f"""You are a specialized CRITIC AGENT in the Felix multi-agent system.

Review Focus: {self.review_focus}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

Your Critical Review Approach:
- Evaluate work from other agents with a critical eye
- Identify gaps, errors, inconsistencies, and weak points
- Suggest specific improvements and corrections
- Ensure quality standards are maintained
- Be constructive but thorough in your criticism

STRICT MODE OVERRIDE: If token budget < 300, list key issues in numbered format with brief explanations. Otherwise, provide comprehensive detailed critique.

Work to Review:
"""
        
        if self.shared_context:
            for key, value in self.shared_context.items():
                base_prompt += f"- {key}: {value}\n"

        # Add knowledge entries if available
        knowledge_summary = ""
        if task.knowledge_entries and len(task.knowledge_entries) > 0:
            knowledge_summary = "\n\nRelevant Knowledge from Memory:\n"
            for entry in task.knowledge_entries:
                # Extract key information from knowledge entry
                if hasattr(entry, 'content'):
                    # Extract 'result' key from dictionary if present (web search results)
                    if isinstance(entry.content, dict):
                        content_str = entry.content.get('result', str(entry.content))
                    else:
                        content_str = str(entry.content)
                else:
                    content_str = str(entry)

                confidence = entry.confidence_level.value if hasattr(entry, 'confidence_level') else "unknown"
                source = entry.source_agent if hasattr(entry, 'source_agent') else "system"
                domain = entry.domain if hasattr(entry, 'domain') else "unknown"

                # Use longer truncation for web_search domain (detailed factual data)
                max_chars = 400 if domain == "web_search" else 200
                if len(content_str) > max_chars:
                    content_str = content_str[:max_chars-3] + "..."

                # Add emoji prefix for web search entries
                prefix = "🌐" if domain == "web_search" else "📝"
                knowledge_summary += f"{prefix} [{source}, conf: {confidence}]: {content_str}\n"

            # Add important instructions for using available knowledge
            knowledge_summary += "\nIMPORTANT: Use the knowledge provided above to answer the task if possible. "
            knowledge_summary += "Only request additional web search if the available knowledge is insufficient or outdated.\n"

        base_prompt += knowledge_summary
        base_prompt += f"""
Task Context: {task.context}

Focus your review on {self.review_focus}. Provide specific, actionable feedback.
Be thorough but constructive - the goal is to improve the final output quality.
"""
        
        # Add token budget guidance if available
        if token_allocation:
            budget_guidance = f"\n\nToken Budget Guidance:\n{token_allocation.style_guidance}"
            if token_allocation.compression_ratio > 0.5:
                budget_guidance += f"\nProvide focused critique with ~{token_allocation.compression_ratio:.0%} compression while covering key quality issues."
            
            enhanced_prompt = base_prompt + budget_guidance
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt, stage_token_budget

    def evaluate_reasoning_process(self, agent_output: Dict[str, Any],
                                   agent_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the reasoning process quality of another agent's output.

        This extends CriticAgent beyond content evaluation to assess HOW agents reasoned,
        not just WHAT they produced. Implements meta-cognitive evaluation for self-improvement.

        Args:
            agent_output: Agent's output including result and metadata
            agent_metadata: Optional metadata about agent's reasoning process

        Returns:
            Dictionary with reasoning evaluation:
                - reasoning_quality_score: 0.0-1.0
                - logical_coherence: 0.0-1.0
                - evidence_quality: 0.0-1.0
                - methodology_appropriateness: 0.0-1.0
                - identified_issues: List[str]
                - improvement_recommendations: List[str]
                - re_evaluation_needed: bool
        """
        result = agent_output.get('result', '')
        confidence = agent_output.get('confidence', 0.5)
        agent_id = agent_output.get('agent_id', 'unknown')

        # Initialize evaluation
        issues = []
        recommendations = []
        scores = {'logical_coherence': 0.5, 'evidence_quality': 0.5, 'methodology': 0.5}

        # 1. Evaluate logical coherence
        if self._has_logical_fallacies(result):
            issues.append("Contains potential logical fallacies")
            scores['logical_coherence'] = 0.4
            recommendations.append("Review reasoning chain for logical consistency")
        else:
            scores['logical_coherence'] = 0.8

        # 2. Evaluate evidence quality
        if self._has_weak_evidence(result):
            issues.append("Evidence appears weak or unsupported")
            scores['evidence_quality'] = 0.4
            recommendations.append("Strengthen claims with more reliable evidence")
        else:
            scores['evidence_quality'] = 0.8

        # 3. Evaluate methodology appropriateness
        if agent_metadata:
            agent_type = agent_metadata.get('agent_type', 'unknown')
            if not self._methodology_appropriate(result, agent_type):
                issues.append(f"Methodology not well-suited for {agent_type} agent")
                scores['methodology'] = 0.4
                recommendations.append(f"Consider approaches more aligned with {agent_type} role")
            else:
                scores['methodology'] = 0.8
        else:
            # No metadata, default moderate score
            scores['methodology'] = 0.6

        # 4. Check reasoning depth
        if len(result.split()) < 50:
            issues.append("Reasoning appears shallow - insufficient depth")
            recommendations.append("Provide more detailed reasoning and analysis")
            # Penalize all scores slightly
            for key in scores:
                scores[key] *= 0.9

        # 5. Check for over/under confidence
        avg_score = sum(scores.values()) / len(scores)
        confidence_gap = abs(confidence - avg_score)
        if confidence_gap > 0.3:
            if confidence > avg_score:
                issues.append(f"Agent appears overconfident (confidence={confidence:.2f} vs quality={avg_score:.2f})")
                recommendations.append("Calibrate confidence based on reasoning quality")
            else:
                issues.append(f"Agent appears underconfident (confidence={confidence:.2f} vs quality={avg_score:.2f})")
                recommendations.append("Increase confidence when reasoning is solid")

        # Calculate overall reasoning quality score
        reasoning_quality = sum(scores.values()) / len(scores)

        # Determine if re-evaluation is needed
        re_evaluation_needed = reasoning_quality < 0.5 or len(issues) >= 3

        logger.info(f"🧠 Reasoning evaluation for {agent_id}:")
        logger.info(f"   Quality: {reasoning_quality:.2f}, Coherence: {scores['logical_coherence']:.2f}, "
                   f"Evidence: {scores['evidence_quality']:.2f}, Methodology: {scores['methodology']:.2f}")
        if issues:
            logger.info(f"   Issues: {', '.join(issues)}")
        if re_evaluation_needed:
            logger.warning(f"   ⚠️  Re-evaluation recommended for {agent_id}")

        return {
            'reasoning_quality_score': reasoning_quality,
            'logical_coherence': scores['logical_coherence'],
            'evidence_quality': scores['evidence_quality'],
            'methodology_appropriateness': scores['methodology'],
            'identified_issues': issues,
            'improvement_recommendations': recommendations,
            're_evaluation_needed': re_evaluation_needed,
            'agent_id': agent_id
        }

    def _has_logical_fallacies(self, text: str) -> bool:
        """Check for common logical fallacies in reasoning."""
        fallacy_indicators = [
            'everyone knows', 'obviously', 'clearly', 'it goes without saying',
            'all experts agree', 'no one would disagree', 'always', 'never'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in fallacy_indicators)

    def _has_weak_evidence(self, text: str) -> bool:
        """Check if evidence appears weak or unsupported."""
        weak_indicators = [
            'i think', 'i believe', 'probably', 'maybe', 'might be',
            'could be', 'seems like', 'appears to'
        ]
        text_lower = text.lower()
        # Count weak indicators
        weak_count = sum(1 for indicator in weak_indicators if indicator in text_lower)
        # High proportion of weak language suggests weak evidence
        return weak_count > 3

    def _methodology_appropriate(self, text: str, agent_type: str) -> bool:
        """Check if reasoning methodology is appropriate for agent type."""
        if agent_type == 'research':
            # Research should explore multiple perspectives
            return 'perspective' in text.lower() or 'source' in text.lower()
        elif agent_type == 'analysis':
            # Analysis should break things down
            return 'because' in text.lower() or 'therefore' in text.lower()
        elif agent_type == 'critic':
            # Critics should identify issues
            return 'issue' in text.lower() or 'problem' in text.lower() or 'improve' in text.lower()
        else:
            # Default: accept methodology
            return True


def create_specialized_team(helix: HelixGeometry, llm_client: LMStudioClient,
                          task_complexity: str = "medium",
                          token_budget_manager: Optional[TokenBudgetManager] = None,
                          random_seed: Optional[int] = None,
                          web_search_client: Optional[WebSearchClient] = None,
                          max_web_queries: int = 3) -> List[LLMAgent]:
    """
    Create a balanced team of specialized agents for a task.

    Args:
        helix: Helix geometry
        llm_client: LM Studio client
        task_complexity: Complexity level (simple, medium, complex)
        token_budget_manager: Optional token budget manager for all agents
        random_seed: Optional seed for spawn time randomization
        web_search_client: Optional web search client for Research agents
        max_web_queries: Maximum web queries per research agent (default: 3)

    Returns:
        List of specialized agents with randomized spawn times
    """
    if task_complexity == "simple":
        return _create_simple_team(helix, llm_client, token_budget_manager, random_seed,
                                  web_search_client, max_web_queries)
    elif task_complexity == "medium":
        return _create_medium_team(helix, llm_client, token_budget_manager, random_seed,
                                   web_search_client, max_web_queries)
    else:  # complex
        return _create_complex_team(helix, llm_client, token_budget_manager, random_seed,
                                    web_search_client, max_web_queries)


def _create_simple_team(helix: HelixGeometry, llm_client: LMStudioClient,
                       token_budget_manager: Optional[TokenBudgetManager] = None,
                       random_seed: Optional[int] = None,
                       web_search_client: Optional[WebSearchClient] = None,
                       max_web_queries: int = 3) -> List[LLMAgent]:
    """Create team for simple tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random spawn times within appropriate ranges for each agent type
    research_spawn = random.uniform(0.05, 0.25)  # Research agents spawn early
    analysis_spawn = random.uniform(0.3, 0.7)    # Analysis agents in middle
    # Note: Synthesis is now handled by CentralPost, not a specialized agent

    return [
        ResearchAgent("research_001", research_spawn, helix, llm_client,
                     token_budget_manager=token_budget_manager, max_tokens=800,
                     web_search_client=web_search_client, max_web_queries=max_web_queries),
        AnalysisAgent("analysis_001", analysis_spawn, helix, llm_client,
                     token_budget_manager=token_budget_manager, max_tokens=800)
    ]


def _create_medium_team(helix: HelixGeometry, llm_client: LMStudioClient,
                       token_budget_manager: Optional[TokenBudgetManager] = None,
                       random_seed: Optional[int] = None,
                       web_search_client: Optional[WebSearchClient] = None,
                       max_web_queries: int = 3) -> List[LLMAgent]:
    """Create team for medium complexity tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random spawn times within appropriate ranges
    research_spawns = [random.uniform(0.02, 0.2) for _ in range(2)]
    analysis_spawns = [random.uniform(0.25, 0.65) for _ in range(2)]
    critic_spawn = random.uniform(0.6, 0.8)
    # Note: Synthesis is now handled by CentralPost, not a specialized agent

    # Sort to maintain some ordering within types
    research_spawns.sort()
    analysis_spawns.sort()

    return [
        ResearchAgent("research_001", research_spawns[0], helix, llm_client, "general",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        ResearchAgent("research_002", research_spawns[1], helix, llm_client, "technical",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        AnalysisAgent("analysis_001", analysis_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        AnalysisAgent("analysis_002", analysis_spawns[1], helix, llm_client, "critical", token_budget_manager, 800),
        CriticAgent("critic_001", critic_spawn, helix, llm_client, "accuracy", token_budget_manager, 800)
    ]


def _create_complex_team(helix: HelixGeometry, llm_client: LMStudioClient,
                        token_budget_manager: Optional[TokenBudgetManager] = None,
                        random_seed: Optional[int] = None,
                        web_search_client: Optional[WebSearchClient] = None,
                        max_web_queries: int = 3) -> List[LLMAgent]:
    """Create team for complex tasks with randomized spawn times."""
    if random_seed is not None:
        random.seed(random_seed)

    # Generate random spawn times within appropriate ranges
    research_spawns = [random.uniform(0.01, 0.25) for _ in range(3)]
    analysis_spawns = [random.uniform(0.2, 0.7) for _ in range(3)]
    critic_spawns = [random.uniform(0.6, 0.8) for _ in range(2)]
    # Note: Synthesis is now handled by CentralPost, not a specialized agent

    # Sort to maintain some ordering within types
    research_spawns.sort()
    analysis_spawns.sort()
    critic_spawns.sort()

    return [
        ResearchAgent("research_001", research_spawns[0], helix, llm_client, "general",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        ResearchAgent("research_002", research_spawns[1], helix, llm_client, "technical",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        ResearchAgent("research_003", research_spawns[2], helix, llm_client, "creative",
                     token_budget_manager, 800, web_search_client, max_web_queries),
        AnalysisAgent("analysis_001", analysis_spawns[0], helix, llm_client, "general", token_budget_manager, 800),
        AnalysisAgent("analysis_002", analysis_spawns[1], helix, llm_client, "technical", token_budget_manager, 800),
        AnalysisAgent("analysis_003", analysis_spawns[2], helix, llm_client, "critical", token_budget_manager, 800),
        CriticAgent("critic_001", critic_spawns[0], helix, llm_client, "accuracy", token_budget_manager, 800),
        CriticAgent("critic_002", critic_spawns[1], helix, llm_client, "completeness", token_budget_manager, 800)
    ]
