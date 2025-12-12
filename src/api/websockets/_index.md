# WebSocket Module

## Purpose
Real-time WebSocket implementation for streaming workflow events, agent updates, and synthesis results to connected clients with connection lifecycle management.

## Key Files

### [workflow_stream.py](workflow_stream.py)
WebSocket endpoint for workflow event streaming.
- **`WS /api/v1/ws/workflows/{workflow_id}`**: Subscribe to real-time workflow updates
- **Event types**: `connected`, `workflow_status`, `agent_spawned`, `agent_output`, `synthesis`, `error`, `completed`
- **Authentication**: Optional API key via query parameter (`?api_key=your-key`)
- **Connection verification**: Checks workflow exists before accepting connection
- **Event structure**: JSON messages with `type`, `workflow_id`, `timestamp`, and event-specific data
- **Helper functions**: `send_workflow_event()` for broadcasting from routers

### [connection_manager.py](connection_manager.py)
WebSocket connection lifecycle and broadcasting management.
- **`ConnectionManager`**: Singleton class managing all WebSocket connections
- **Connection tracking**: Maps workflow IDs to sets of WebSocket connections
- **Broadcast methods**:
  - `broadcast_to_workflow()`: Send events to all clients watching a specific workflow
  - `broadcast_to_all()`: Send system-wide events to all connected clients
- **Metadata tracking**: Stores connection info (workflow_id, connected_at, messages_sent)
- **Thread-safe**: Uses asyncio locks for concurrent connection management
- **Cleanup**: Automatic connection cleanup on disconnect

## Key Concepts

### Event-Driven Updates
WebSocket provides real-time workflow monitoring without polling:
- **Instant feedback**: Clients receive events as they happen
- **Low overhead**: Single persistent connection vs repeated HTTP requests
- **Bidirectional**: Potential for client-to-server commands (currently unused)

### Connection Lifecycle
```
Client connects → Manager accepts → Registers workflow_id → Streams events → Client disconnects → Cleanup
```

### Event Types
1. **connected**: Initial handshake confirmation
2. **workflow_status**: State transitions (pending → running → completed/failed)
3. **agent_spawned**: New agent joins team with spawn time and type
4. **agent_output**: Agent produces response with confidence score
5. **synthesis**: Final synthesis in progress or complete
6. **error**: Workflow error with message
7. **completed**: Terminal success state with final results

### Workflow-Scoped Broadcasting
Connections are scoped to specific workflows:
- Clients only receive events for workflows they're watching
- Multiple clients can watch same workflow
- Efficient message routing via workflow_id mapping

### Authentication
WebSocket uses query parameter authentication:
```
ws://localhost:8000/api/v1/ws/workflows/wf_abc123?api_key=your-key
```
- Same API key as REST endpoints
- Optional in development mode (no `FELIX_API_KEY` env var)
- Verified before accepting connection

### Connection Metadata
Each connection tracks:
- **workflow_id**: Which workflow is being watched
- **connected_at**: Connection timestamp
- **messages_sent**: Event delivery count for monitoring

### Integration with Routers
Workflow routers broadcast events during execution:
```python
await send_workflow_event(workflow_id, "agent_spawned", {
    "agent_id": agent.agent_id,
    "agent_type": agent.role,
    "spawn_time": normalized_time
})
```

### Error Handling
- **WebSocketDisconnect**: Gracefully handled with cleanup
- **Connection errors**: Logged but don't crash other connections
- **Invalid workflow**: Rejects connection with error before accepting

## Client Example

**JavaScript WebSocket Client:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/workflows/wf_abc123?api_key=secret');

ws.onopen = () => console.log('Connected');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'agent_spawned':
      console.log(`Agent ${data.agent_id} spawned`);
      break;
    case 'synthesis':
      console.log('Synthesis:', data.content);
      break;
    case 'completed':
      console.log('Workflow complete!');
      break;
  }
};

ws.onerror = (error) => console.error('WebSocket error:', error);
ws.onclose = () => console.log('Disconnected');
```

## Related Modules
- [routers/workflows.py](../routers/workflows.py) - Broadcasts workflow events
- [dependencies.py](../dependencies.py) - Authentication and resource access
- [models.py](../models.py) - Event data structures
- [communication/](../../communication/) - CentralPost event generation
