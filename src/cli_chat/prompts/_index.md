# CLI Chat Prompts Module

## Purpose
Chat-specific prompt templates for conversational CLI interactions (currently empty - uses system defaults from [src/prompts/](../../prompts/)).

## Key Files

**No chat-specific prompts currently.** Felix CLI uses the system-wide prompts from `src/prompts/` which are shared across GUI, API, and CLI interfaces.

## Key Concepts

### Prompt Inheritance
CLI chat inherits prompts from the central prompt system:
- **Agent prompts**: From `src/prompts/` via `PromptManager`
- **Position-aware prompts**: Helix geometry context automatically added by `LLMAgent`
- **Role-specific prompts**: Research, Analysis, Critic agents use specialized templates

### Why Separate Directory?
This directory exists for **future CLI-specific customizations**:
- Custom greeting messages for chat interface
- CLI-specific instruction templates
- Session-aware context prompts
- Tool usage guidance prompts

### Future Chat-Specific Prompts
Potential prompts that could go here:
1. **greeting.txt**: Initial message when starting new session
2. **help_guidance.txt**: System help and available commands
3. **context_continuation.txt**: Template for conversation threading
4. **tool_usage.txt**: Instructions for using @ prefix tools
5. **command_suggestions.txt**: Suggest relevant commands based on context

### Current Prompt System
For now, all prompts managed centrally via:
- **`src/prompts/`**: System-wide prompt templates
- **`PromptManager`**: Central prompt loading and management
- **`LLMAgent.get_position_aware_prompt()`**: Dynamic prompt generation with helix context

## Related Modules
- [src/prompts/](../../prompts/) - System-wide prompt templates (current source)
- [chat.py](../chat.py) - CLI interface using prompts
- [agents/](../../agents/) - Agents using prompt templates
- [cli_workflow_orchestrator.py](../cli_workflow_orchestrator.py) - Workflow execution with prompts
