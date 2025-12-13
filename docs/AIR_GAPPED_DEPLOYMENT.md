# Air-Gapped Deployment Guide

> Felix is the framework that **assumes** you're offline, not one that **tolerates** it.

This guide demonstrates Felix's deployment in completely air-gapped environments where network isolation is mandatory. Unlike other frameworks that *can* work offline with configuration, Felix is *designed* for offline operation from day one.

## Table of Contents

1. [Real-World Scenario: SCIF Deployment](#real-world-scenario-scif-deployment)
2. [Graceful Degradation in Action](#graceful-degradation-in-action)
3. [Trust System for Safe Autonomy](#trust-system-for-safe-autonomy)
4. [Comparison with Other Frameworks](#comparison-with-other-frameworks)
5. [Complete Deployment Configuration](#complete-deployment-configuration)
6. [Step-by-Step Deployment](#step-by-step-deployment)
7. [Audit Trail and Compliance](#audit-trail-and-compliance)

---

## Real-World Scenario: SCIF Deployment

### The Environment

A defense contractor operates a Sensitive Compartmented Information Facility (SCIF) for analyzing classified technical documents. The requirements are strict:

- **Zero internet access** - Physical air gap, no network bridge
- **No cloud services** - All processing must occur locally
- **Continuous operation** - System must work even when components fail
- **Safe autonomy** - AI agents can execute commands, but with guardrails
- **Complete audit trail** - Every action logged for compliance review

### The Challenge

Traditional multi-agent AI frameworks assume internet connectivity:

- Vector databases often require cloud services or complex local setup
- Embedding models typically call external APIs
- When components fail, workflows crash
- No built-in safety controls for autonomous command execution

### Felix's Solution

Felix addresses each requirement out of the box:

```
SCIF Environment
├── Felix Framework (Python 3.12+)
├── LM Studio (local LLM inference)
├── SQLite databases (auto-created, zero config)
└── No external services required

Fallback Chain:
LM Studio embeddings → TF-IDF → SQLite FTS5 (always works)
```

---

## Graceful Degradation in Action

Felix's three-tier embedding system ensures knowledge retrieval **never fails**, even when components go down.

### The Three Tiers

| Tier | Technology | Quality | Requirements |
|------|------------|---------|--------------|
| 1 | LM Studio Embeddings | Best (768-dim semantic vectors) | LM Studio running with embedding model |
| 2 | TF-IDF | Good (keyword-based semantic) | Document corpus fitted (automatic) |
| 3 | SQLite FTS5 | Basic (full-text search with BM25) | None (built into Python) |

### What Happens When Things Fail

```
┌─────────────────────────────────────────────────────────────────┐
│ DAY 1: Full Capability                                          │
├─────────────────────────────────────────────────────────────────┤
│ ✓ LM Studio running with Mistral-7B                             │
│ ✓ Embeddings: Tier 1 (768-dimensional semantic vectors)         │
│ ✓ Knowledge retrieval: High-quality semantic search             │
│ ✓ Query: "hydraulic system failures" finds related concepts     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ DAY 3: LM Studio Server Crashes                                 │
├─────────────────────────────────────────────────────────────────┤
│ ⚠ LM Studio embedding timeout after 5 seconds                   │
│ ✓ AUTOMATIC fallback to TF-IDF (no restart needed)              │
│ ✓ Embeddings: Tier 2 (keyword-based semantic matching)          │
│ ✓ Knowledge retrieval: Still semantic, slightly lower quality   │
│ ✓ Workflow continues uninterrupted                              │
│                                                                 │
│ Log: "LM Studio embedding failed, falling back to TF-IDF"       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ DAY 5: TF-IDF Vocabulary Corrupted                              │
├─────────────────────────────────────────────────────────────────┤
│ ⚠ TF-IDF embedding returns None                                 │
│ ✓ AUTOMATIC fallback to FTS5 (built into SQLite)                │
│ ✓ Embeddings: None needed                                       │
│ ✓ Knowledge retrieval: Full-text search with BM25 ranking       │
│ ✓ Workflow continues uninterrupted                              │
│                                                                 │
│ Log: "TF-IDF embedding failed, falling back to FTS5"            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Behavior: Workflow Never Stops

The fallback is **permanent per session** - once a tier fails, Felix switches and stays switched. This prevents repeated timeout delays:

```python
# From src/knowledge/embeddings.py
if self.active_tier == EmbeddingTier.LM_STUDIO:
    embedding = self.lm_studio_embedder.embed(text)
    if embedding is not None:
        return EmbeddingResult(...)

    # Fallback to TF-IDF if LM Studio fails
    logger.warning("LM Studio embedding failed, falling back to TF-IDF")
    self.active_tier = EmbeddingTier.TFIDF  # PERMANENT SWITCH
```

### Tier Status Monitoring

You can check the current tier at any time:

```python
from src.knowledge.embeddings import EmbeddingProvider

provider = EmbeddingProvider()
status = provider.get_tier_info()

# Returns:
{
    'active_tier': 'tfidf',  # Current tier
    'tiers_available': {
        'lm_studio': False,   # LM Studio down
        'tfidf': True,        # Fitted and ready
        'fts5': True          # Always available
    }
}
```

---

## Trust System for Safe Autonomy

Felix agents can execute system commands, but every command passes through a three-tier trust classification.

### Trust Levels

| Level | Behavior | Examples |
|-------|----------|----------|
| **SAFE** | Auto-execute immediately | `ls`, `pwd`, `git status`, `pip list` |
| **REVIEW** | Workflow pauses for user approval | `pip install`, `mkdir`, `git commit` |
| **BLOCKED** | Never execute, logged as security event | `rm -rf`, `sudo`, credential access |

### Real Workflow Example

```
┌────────────────────────────────────────────────────────────────┐
│ WORKFLOW: Analyze classified technical documents               │
└────────────────────────────────────────────────────────────────┘

STEP 1: Research Agent needs directory listing
┌─────────────────────────────────────────┐
│ Command: ls -la /data/classified/       │
│ Trust Level: SAFE                       │
│ Action: Auto-execute immediately        │
│ Result: Directory contents returned     │
└─────────────────────────────────────────┘
                    │
                    ▼
STEP 2: Analysis Agent needs pandas library
┌─────────────────────────────────────────┐
│ Command: pip install pandas             │
│ Trust Level: REVIEW                     │
│ Action: Workflow PAUSES                 │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ ⚠️  APPROVAL REQUIRED               │ │
│ │                                     │ │
│ │ Agent: analysis_agent_001           │ │
│ │ Command: pip install pandas         │ │
│ │ Risk: Low - Package installation    │ │
│ │                                     │ │
│ │ [Approve Once]                      │ │
│ │ [Always - This Command Type] ←      │ │
│ │ [Deny]                              │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ User selects: "Always - This Command"   │
│ → All future pip commands auto-approved │
│ → Rule expires when workflow ends       │
└─────────────────────────────────────────┘
                    │
                    ▼
STEP 3: Agent attempts dangerous cleanup
┌─────────────────────────────────────────┐
│ Command: rm -rf /tmp/analysis_*         │
│ Trust Level: BLOCKED                    │
│ Action: NEVER execute                   │
│                                         │
│ Result: Agent receives failure message  │
│ Audit: Security event logged            │
│ Workflow: Continues (command rejected)  │
└─────────────────────────────────────────┘
                    │
                    ▼
STEP 4: Agent needs another pip install
┌─────────────────────────────────────────┐
│ Command: pip install matplotlib         │
│ Trust Level: REVIEW                     │
│ Auto-Approval Rule: Matches "pip *"     │
│ Action: Auto-execute (no dialog)        │
│ Result: Package installed               │
└─────────────────────────────────────────┘
```

### Approval Decision Types

When a REVIEW command requires approval, users have five options:

| Decision | Scope | Persistence |
|----------|-------|-------------|
| **Approve Once** | This exact command | One-time only |
| **Always - Exact Match** | This exact command | Current workflow |
| **Always - Command Type** | All commands of this type (e.g., all `pip`) | Current workflow |
| **Always - Path Pattern** | Commands matching path pattern | Current workflow |
| **Deny** | Reject and continue workflow | N/A |

All session-scoped rules **expire when the workflow ends**, preventing permission creep.

### Security Guarantees

1. **Deny-by-default**: Unknown commands require approval (default to REVIEW)
2. **Destructive protection**: `rm -rf` patterns are hardcoded to BLOCKED, cannot be overridden
3. **No privilege escalation**: `sudo` commands always BLOCKED
4. **Credential protection**: Access to `.ssh`, `.aws`, `.docker` always BLOCKED
5. **Audit everything**: Every command logged with trust level and approver

---

## Comparison with Other Frameworks

### Honest Assessment

All major frameworks (LangChain, CrewAI, AutoGen) **can** work offline. The difference is in design philosophy:

| Capability | Felix | LangChain + FAISS | CrewAI + Ollama | AutoGen + Local |
|------------|-------|-------------------|-----------------|-----------------|
| **Works offline out of box** | Yes | Requires configuration | Requires configuration | Requires configuration |
| **Automatic embedding fallback** | 3-tier (LM Studio → TF-IDF → FTS5) | Manual configuration | Manual configuration | Manual configuration |
| **Trust system for commands** | Built-in (SAFE/REVIEW/BLOCKED) | Not included | Not included | Not included |
| **Graceful degradation** | Automatic, workflow continues | Crashes on component failure | Crashes on component failure | Crashes on component failure |
| **Zero external services** | SQLite only | Needs FAISS or Chroma setup | Needs Ollama | Needs local model server |
| **Audit trail** | Built-in with approver tracking | Not included | Not included | Not included |

### What This Means in Practice

**Scenario**: LLM embedding server goes down at 2 AM

| Framework | Result |
|-----------|--------|
| **Felix** | Warning logged, automatic TF-IDF fallback, workflow continues |
| **LangChain** | Exception thrown, workflow crashes, requires restart |
| **CrewAI** | Exception thrown, workflow crashes, requires restart |
| **AutoGen** | Exception thrown, workflow crashes, requires restart |

**Scenario**: Agent generates `rm -rf /important/data`

| Framework | Result |
|-----------|--------|
| **Felix** | Command blocked before execution, security event logged, workflow continues with failure |
| **LangChain** | Command executes if not manually prevented |
| **CrewAI** | Command executes if not manually prevented |
| **AutoGen** | Command executes if not manually prevented |

### Competitor Setup for Offline

To achieve similar offline capability with other frameworks:

**LangChain** requires:
```python
# Install FAISS
pip install faiss-cpu  # or faiss-gpu

# Configure embedding model
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up vector store
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# No automatic fallback if FAISS fails
```

**CrewAI** requires:
```python
# Install and run Ollama
# Download model manually
ollama pull llama2

# Configure CrewAI
from crewai import Agent
agent = Agent(llm="ollama/llama2")

# No automatic fallback if Ollama fails
```

**Felix** requires:
```yaml
# config/llm.yaml - that's it
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "local-model"
```

---

## Complete Deployment Configuration

### LLM Configuration

```yaml
# config/llm.yaml - Air-gapped configuration

primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "mistral-7b-instruct"  # Or any GGUF model
  timeout: 120
  max_retries: 3
  backoff_factor: 1.5

# No cloud fallbacks in air-gapped environment
fallbacks: []

router:
  retry_on_rate_limit: false
  verbose_logging: true  # For compliance auditing
```

### Trust Rules Configuration

```yaml
# config/trust_rules.yaml - High-security configuration

# Commands that auto-execute (read-only operations)
safe:
  - '^ls(\s|$)'           # Directory listing
  - '^pwd$'               # Current directory
  - '^cat\s(?!.*classified|.*secret|.*\.ssh|.*\.aws)'  # Read files (except sensitive)
  - '^head\s'             # File preview
  - '^tail\s'             # File tail
  - '^grep\s'             # Search
  - '^find\s'             # Find files
  - '^wc\s'               # Word count
  - '^git\s+status'       # Git status
  - '^git\s+log'          # Git history
  - '^git\s+diff'         # Git diff
  - '^python\s+--version' # Version check
  - '^pip\s+list'         # Installed packages
  - '^date$'              # Current date
  - '^whoami$'            # Current user
  - '^df\s'               # Disk usage
  - '^free\s'             # Memory usage

# Commands requiring user approval
review:
  - '^pip\s+install'      # Package installation
  - '^python\s+[^-]'      # Python script execution
  - '^git\s+add'          # Git staging
  - '^git\s+commit'       # Git commit
  - '^mkdir\s'            # Directory creation
  - '^touch\s'            # File creation
  - '^cp\s'               # File copy
  - '^mv\s'               # File move
  - '^chmod\s'            # Permission change

# Commands that NEVER execute
blocked:
  # Destructive operations
  - 'rm\s+-rf'            # Recursive force delete
  - 'rm\s+-fr'            # Same as above
  - 'rm\s+.*/'            # Delete directories
  - '^dd\s'               # Disk operations
  - '^mkfs\s'             # Filesystem creation
  - '^fdisk\s'            # Partition editing
  - 'truncate\s+-s\s*0'   # File truncation

  # Privilege escalation
  - '^sudo\s'             # All sudo commands
  - '^su\s'               # User switching
  - '^chmod\s+777'        # World-writable
  - '^chown\s'            # Ownership change

  # Network operations (air-gapped)
  - '^curl\s'             # HTTP requests
  - '^wget\s'             # Downloads
  - '^ssh\s'              # Remote access
  - '^scp\s'              # Remote copy
  - '^rsync\s'            # Remote sync
  - '^nc\s'               # Netcat
  - '^nmap\s'             # Port scanning

  # Credential access
  - 'cat\s+.*\.ssh/(id_rsa|id_ed25519|id_ecdsa)(?!\.pub)'
  - 'cat\s+.*\.aws/credentials'
  - 'cat\s+.*\.docker/config'
  - 'cat\s+.*/\.env'
  - 'cat\s+.*password'
  - 'cat\s+.*secret'

  # System shutdown
  - '^shutdown'
  - '^reboot'
  - '^halt'
  - '^poweroff'
  - '^systemctl\s+(halt|poweroff|reboot)'

  # Database destruction
  - 'DROP\s+DATABASE'
  - 'DROP\s+TABLE'
  - 'DELETE\s+FROM.*WHERE\s+1\s*=\s*1'
  - 'TRUNCATE\s+TABLE'
```

### Knowledge Brain Configuration

```yaml
# In config/llm.yaml (or Settings tab in GUI)

knowledge:
  # Embedding mode: auto, lm_studio, tfidf, fts5
  embedding_mode: "auto"  # Automatic tier selection with fallback

  # Processing settings
  chunk_size: 1000        # Characters per chunk
  chunk_overlap: 200      # Overlap between chunks
  max_threads: 4          # Parallel processing threads

  # Watch directories for document ingestion
  watch_directories:
    - "/data/documents"
    - "/data/reports"

  # Auto-augment workflows with relevant knowledge
  auto_augment: true
```

---

## Step-by-Step Deployment

### Phase 1: Pre-Deployment (Internet-Connected Staging)

Prepare all components before entering the air-gapped environment.

```bash
# 1. Clone Felix repository
git clone https://github.com/your-org/felix.git
cd felix

# 2. Download Python dependencies for offline install
pip download -r requirements.txt -d ./offline_packages/

# 3. Download LM Studio and models
# - Download LM Studio installer
# - Download GGUF models (e.g., mistral-7b-instruct-v0.2.Q4_K_M.gguf)

# 4. Package everything
tar -czvf felix-airgapped-bundle.tar.gz \
    felix/ \
    offline_packages/ \
    lm-studio-installer.deb \
    models/
```

### Phase 2: Transfer to Air-Gapped Environment

Transfer the bundle via approved media (encrypted USB, data diode, etc.).

```bash
# Verify integrity
sha256sum -c checksums.txt

# Extract
tar -xzvf felix-airgapped-bundle.tar.gz
```

### Phase 3: Installation

```bash
# 1. Create virtual environment
cd felix
python3 -m venv .venv
source .venv/bin/activate

# 2. Install from offline packages
pip install --no-index --find-links ./offline_packages -r requirements.txt

# 3. Install Felix
pip install -e .

# 4. Install LM Studio
sudo dpkg -i ../lm-studio-installer.deb

# 5. Load model in LM Studio
# - Open LM Studio
# - Load model from ../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
# - Start server on localhost:1234
```

### Phase 4: Configuration

```bash
# 1. Configure LLM
cat > config/llm.yaml << 'EOF'
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "mistral-7b-instruct"
  timeout: 120
fallbacks: []
EOF

# 2. Configure trust rules (use high-security template)
cp config/trust_rules_airgapped.yaml config/trust_rules.yaml

# 3. Verify configuration
python -m src.cli test-connection
```

### Phase 5: Initialize and Verify

```bash
# 1. Initialize databases (auto-created on first run)
python -m src.cli init

# Expected output:
# ✓ felix_knowledge.db initialized (21 tables)
# ✓ felix_memory.db initialized
# ✓ felix_workflow_history.db initialized
# ✓ felix_task_memory.db initialized
# ✓ felix_system_actions.db initialized

# 2. Check system status
python -m src.cli status

# Expected output:
# LLM Provider: lm_studio (connected)
# Embedding Tier: lm_studio (768-dim)
# Knowledge Brain: ready (0 documents)
# Trust System: active (15 safe, 9 review, 28 blocked patterns)

# 3. Run test workflow
python -m src.cli run "List the current directory and report system status"

# Verify:
# - Workflow completes
# - ls command auto-executed (SAFE)
# - Results returned
```

### Phase 6: Production Operation

```bash
# GUI mode (recommended for interactive use)
python -m src.gui_ctk

# CLI mode (for automation)
python -m src.cli run "Your task here" --output results.md

# API mode (for integration)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## Audit Trail and Compliance

Felix maintains a complete audit trail of all command executions for compliance review.

### What Gets Logged

Every command execution records:

| Field | Description | Example |
|-------|-------------|---------|
| `command` | The exact command | `pip install pandas` |
| `command_hash` | SHA256 for deduplication | `a1b2c3d4...` |
| `trust_level` | Classification | `REVIEW` |
| `approved_by` | Who/what approved | `user`, `auto`, `auto_rule` |
| `agent_id` | Requesting agent | `research_agent_001` |
| `workflow_id` | Parent workflow | `wf_20240115_143022` |
| `exit_code` | Result | `0` |
| `stdout` | Command output | `Successfully installed...` |
| `stderr` | Error output | (empty) |
| `duration` | Execution time | `2.34s` |
| `timestamp` | When executed | `2024-01-15T14:30:45Z` |
| `cwd` | Working directory | `/data/analysis` |

### Audit Database

All command history is stored in `felix_system_actions.db`:

```sql
-- Query recent commands
SELECT
    timestamp,
    command,
    trust_level,
    approved_by,
    agent_id,
    exit_code
FROM command_history
WHERE workflow_id = 'wf_20240115_143022'
ORDER BY timestamp;

-- Query blocked commands (security events)
SELECT *
FROM command_history
WHERE trust_level = 'BLOCKED'
ORDER BY timestamp DESC;

-- Query approval decisions
SELECT
    command,
    decided_by,
    decision_type,
    timestamp
FROM approval_history
WHERE workflow_id = 'wf_20240115_143022';
```

### Compliance Reporting

Generate compliance reports:

```bash
# Export command history for audit
python -m src.cli export-audit \
    --start "2024-01-01" \
    --end "2024-01-31" \
    --format csv \
    --output audit_january.csv

# Export blocked command attempts
python -m src.cli export-audit \
    --filter "trust_level=BLOCKED" \
    --format json \
    --output security_events.json
```

### Compliance Framework Mapping

| Requirement | Felix Feature | Evidence Location |
|-------------|--------------|-------------------|
| **HIPAA Access Logging** | Command audit trail | `felix_system_actions.db` |
| **SOX Change Control** | REVIEW approval workflow | `approval_history` table |
| **FISMA Least Privilege** | SAFE/REVIEW/BLOCKED classification | `config/trust_rules.yaml` |
| **ITAR Data Isolation** | Air-gapped operation, no cloud | Architecture design |
| **PCI-DSS Audit Trail** | Complete command logging | `command_history` table |

---

## Summary

Felix provides genuine value for air-gapped deployments through:

1. **Zero-Configuration Offline Operation**
   - Works out of the box without cloud services
   - SQLite databases auto-initialize

2. **Three-Tier Graceful Degradation**
   - LM Studio → TF-IDF → SQLite FTS5
   - Workflow never stops, automatic fallback

3. **Built-In Trust System**
   - SAFE/REVIEW/BLOCKED command classification
   - Workflow pause/resume with user approval
   - Session-scoped rules that expire

4. **Complete Audit Trail**
   - Every command logged with context
   - Compliance-ready reporting
   - Security event tracking

Other frameworks can work offline with configuration. Felix is designed for it from day one.

---

## Related Documentation

- [Configuration Reference](CONFIGURATION.md) - Complete configuration guide
- [Architecture Overview](ARCHITECTURE.md) - System design details
- [Knowledge Brain API](KNOWLEDGE_BRAIN_API.md) - Document ingestion
- [CLI Guide](CLI_GUIDE.md) - Command-line interface
- [Developer Guide](DEVELOPER_GUIDE.md) - Extension and customization
