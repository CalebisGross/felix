# Execution Module

## Purpose
Safe system command execution with approval workflows, trust management, command history tracking, and error categorization for agent-initiated shell operations.

## Key Files

### [system_executor.py](system_executor.py)
Core shell command execution with safety controls.
- **`SystemExecutor`**: Executes shell commands with sandboxing and timeout controls
- **`ExecutionError`**: Custom exception for execution failures
- **`ErrorCategory`**: Enum for error classification (TIMEOUT, PERMISSION, NOT_FOUND, SYNTAX, SYSTEM, UNKNOWN)
- **`CommandResult`**: Execution result structure with stdout, stderr, return code, execution time

**Features**:
- Subprocess isolation
- Timeout enforcement (default: 30s)
- Output capture (stdout/stderr)
- Return code checking
- Error categorization
- Resource limit enforcement

### [approval_manager.py](approval_manager.py)
Multi-stage approval workflow for command authorization.
- **`ApprovalManager`**: Manages command approval requests and decisions
- **`ApprovalDecision`**: Enum for decision types (APPROVE_ONCE, APPROVE_ALWAYS, DENY_ONCE, DENY_ALWAYS, REQUEST_MORE_INFO)
- **`ApprovalStatus`**: Enum for request status (PENDING, APPROVED, DENIED, EXPIRED)
- **`WorkflowApprovalRule`**: Rule engine for automatic approval/denial based on patterns

**Decision Types**:
1. **APPROVE_ONCE**: Allow single execution without storing trust
2. **APPROVE_ALWAYS**: Grant permanent trust for command pattern
3. **DENY_ONCE**: Reject single execution without blocking future requests
4. **DENY_ALWAYS**: Permanently block command pattern
5. **REQUEST_MORE_INFO**: Ask agent for justification or clarification

### [trust_manager.py](trust_manager.py)
Trust level assessment and management.
- **`TrustManager`**: Evaluates and stores trust decisions for commands
- **`TrustLevel`**: Enum for trust classification (TRUSTED, NEUTRAL, SUSPICIOUS, BLOCKED)
- **`ApprovalRequest`**: Request structure with command, context, justification

**Trust Evaluation Factors**:
- Command pattern (read-only vs destructive)
- Agent track record
- Historical success rate
- User approval history
- Risk assessment (file system access, network operations, process spawning)

**Trust Levels**:
- **TRUSTED**: Approved always, auto-execute
- **NEUTRAL**: Request approval for each execution
- **SUSPICIOUS**: Warn user, require explicit approval
- **BLOCKED**: Deny always, prevent execution

### [command_history.py](command_history.py)
Command execution tracking and logging.
- **`CommandHistory`**: Persists command execution records
- **Tracked Data**:
  - Command text and arguments
  - Execution timestamp
  - Duration and return code
  - Output (stdout/stderr)
  - Agent/user who initiated
  - Approval decision

**Database**: `felix_command_history.db`
- `command_history` table: Full execution records
- Indexed on `timestamp` and `agent_id`

## Key Concepts

### Approval Workflow

```
Agent requests command
       ↓
Trust Manager evaluates
       ↓
    Is Trusted?
   ↙         ↘
YES           NO
 ↓             ↓
Execute    Request Approval
 ↓             ↓
           User Decides
              ↓
     ┌────────┼────────┐
  APPROVE  DENY  REQUEST_MORE_INFO
     ↓       ↓           ↓
  Execute  Block    Ask Agent
     ↓       ↓           ↓
  Update   Update    Retry
  Trust    Trust
```

### Command Safety Checks

**Pre-execution**:
- Pattern matching against blocked commands
- Risk assessment (destructive operations, privilege escalation)
- Trust level check
- Approval requirement check

**During execution**:
- Subprocess isolation (no shell=True when possible)
- Timeout enforcement
- Resource limits
- Output capture

**Post-execution**:
- Return code validation
- Error categorization
- History logging
- Trust update

### Risk Assessment

Commands categorized by risk:

**Low Risk** (auto-approve for trusted agents):
- Read operations: `cat`, `ls`, `head`, `tail`
- Status checks: `git status`, `ps`, `df`
- Queries: `which`, `whereis`, `find` (non-destructive)

**Medium Risk** (request approval):
- File modifications: `echo`, `sed`, `awk`
- Directory changes: `mkdir`, `cd`
- Downloads: `wget`, `curl` (with size limits)

**High Risk** (require explicit approval + justification):
- Destructive operations: `rm`, `dd`, `mkfs`
- Privilege escalation: `sudo`, `su`
- Network operations: `nc`, `telnet`, `ssh`
- Process control: `kill`, `killall`

**Critical Risk** (default deny, require override):
- System modifications: `chmod 777`, `chown root`
- Package management: `apt install`, `pip install` (without review)
- Kernel operations: `modprobe`, `insmod`

### Trust Evolution

Trust levels evolve based on outcomes:
- **Successful executions** → Increase trust
- **Failures with explanation** → Maintain trust
- **Suspicious patterns** → Decrease trust
- **User overrides** → Adjust trust weights

### Error Categorization

Helps agents learn from failures:
- **TIMEOUT**: Increase timeout or optimize command
- **PERMISSION**: Request elevated privileges or adjust target
- **NOT_FOUND**: Install missing tools or fix paths
- **SYNTAX**: Fix command syntax
- **SYSTEM**: Report to user, may require intervention

## Configuration

```yaml
execution:
  default_timeout: 30                 # Seconds
  max_output_size: 1048576           # 1MB
  enable_approval_workflow: true
  auto_approve_read_only: false      # Auto-approve safe read commands
  trust_decay_days: 30               # Days before trust decays
```

## Usage Example

```python
from src.execution.system_executor import SystemExecutor
from src.execution.approval_manager import ApprovalManager
from src.execution.trust_manager import TrustManager

# Initialize components
executor = SystemExecutor(timeout=30)
trust_mgr = TrustManager()
approval_mgr = ApprovalManager(trust_mgr)

# Request command execution
command = "ls -la /home"
trust_level = trust_mgr.assess_trust(command, agent_id="research_001")

if trust_level == TrustLevel.TRUSTED:
    result = executor.execute(command)
else:
    approval = approval_mgr.request_approval(command, agent_id="research_001")
    if approval.status == ApprovalStatus.APPROVED:
        result = executor.execute(command)
```

## Related Modules
- [agents/](../agents/) - SystemAgent uses execution framework
- [communication/](../communication/) - SystemCommandManager coordinates requests
- [gui/](../gui/) - Approvals and Terminal tabs for user interaction
- [memory/](../memory/) - Command history persistence
