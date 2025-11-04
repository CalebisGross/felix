# Integration Examples

## Purpose
Examples for integrating Felix with external systems: CI/CD pipelines, messaging platforms, webhooks, and custom workflows.

## Key Files

**Coming soon**: This directory will contain integration examples for:

### CI/CD Integration
- **GitHub Actions workflow**: Trigger Felix analysis on pull requests
- **GitLab CI pipeline**: Run Felix code review on commits
- **Jenkins plugin**: Integrate Felix with Jenkins builds

### Messaging Platforms
- **Slack bot**: Interactive Felix assistant for Slack
- **Discord bot**: Felix integration for Discord servers
- **Teams bot**: Microsoft Teams Felix connector

### Webhook Handlers
- **Generic webhook receiver**: Process incoming webhooks with Felix
- **Webhook forwarder**: Send Felix results to external systems

### Custom Workflows
- **Scheduled tasks**: Cron-based Felix task execution
- **Event-driven processing**: React to file changes, API events
- **Multi-system orchestration**: Coordinate Felix with other tools

## Planned Examples

### slack_bot_integration.py
```python
# Example: Slack bot using Felix API
from slack_bolt import App
from felix_api_client import FelixClient

app = App(token=os.environ["SLACK_BOT_TOKEN"])
felix = FelixClient(api_key=os.environ["FELIX_API_KEY"])

@app.message("felix")
def handle_felix_query(message, say):
    query = message["text"].replace("felix", "").strip()
    result = felix.run_workflow(query)
    say(result.synthesis.content)
```

### github_actions_workflow.yml
```yaml
# Example: GitHub Actions workflow using Felix
name: Felix Code Review
on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Felix Review
        run: |
          python -m felix_cli review --pr ${{ github.event.pull_request.number }}
```

## Integration Patterns

### Pattern 1: API-Based Integration
Use Felix REST API for external system integration:
- Create workflows via POST requests
- Monitor progress via WebSocket
- Retrieve results via GET requests

### Pattern 2: CLI-Based Integration
Use Felix CLI for scripting and automation:
- Execute workflows from shell scripts
- Integrate with cron jobs
- Pipe inputs/outputs between systems

### Pattern 3: SDK-Based Integration
Use Felix as a Python library:
- Import Felix modules directly
- Embed Felix in applications
- Custom workflow orchestration

## Coming Soon

Check back for integration examples, or contribute your own!

To contribute an integration example:
1. Fork the repository
2. Add your example to this directory
3. Update this index
4. Submit a pull request

## Related Modules
- [src/api/](../../src/api/) - REST API for integrations
- [src/cli.py](../../src/cli.py) - CLI for scripting
- [examples/api_examples/](../api_examples/) - API client examples
