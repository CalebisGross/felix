#!/usr/bin/env python3
"""
Example WebSocket client for Felix REST API.

Demonstrates how to:
1. Create a workflow via REST API
2. Connect to WebSocket for real-time updates
3. Process events as they occur
4. Handle completion and errors

Requirements:
    pip install websockets httpx

Usage:
    # Start Felix API server first
    python -m uvicorn src.api.main:app --port 8000

    # Run this client
    python examples/api_examples/websocket_client_example.py
"""

import asyncio
import json
import sys
from typing import Optional

try:
    import websockets
    import httpx
except ImportError:
    print("Error: Required packages not installed")
    print("Please install: pip install websockets httpx")
    sys.exit(1)


# Configuration
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
API_KEY = "your-secret-api-key-here"  # Change if using authentication


class FelixWebSocketClient:
    """
    WebSocket client for Felix workflow streaming.
    """

    def __init__(self, api_url: str, ws_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.ws_url = ws_url
        self.api_key = api_key

        # HTTP client for REST API
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def start_felix(self) -> bool:
        """Start Felix system."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/api/v1/system/start",
                    headers=self.headers,
                    timeout=30.0
                )

                if response.status_code == 200:
                    print("✅ Felix system started")
                    return True
                elif response.status_code == 409:
                    print("✅ Felix system already running")
                    return True
                else:
                    print(f"❌ Failed to start Felix: {response.status_code}")
                    print(response.json())
                    return False

            except Exception as e:
                print(f"❌ Error starting Felix: {e}")
                return False

    async def create_workflow(self, task: str, max_steps: int = 10) -> Optional[str]:
        """
        Create a new workflow.

        Args:
            task: Task description
            max_steps: Maximum workflow steps

        Returns:
            Workflow ID if successful, None otherwise
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/api/v1/workflows",
                    headers=self.headers,
                    json={"task": task, "max_steps": max_steps},
                    timeout=30.0
                )

                if response.status_code == 202:
                    workflow = response.json()
                    workflow_id = workflow["workflow_id"]
                    print(f"✅ Workflow created: {workflow_id}")
                    return workflow_id
                else:
                    print(f"❌ Failed to create workflow: {response.status_code}")
                    print(response.json())
                    return None

            except Exception as e:
                print(f"❌ Error creating workflow: {e}")
                return None

    async def stream_workflow(self, workflow_id: str):
        """
        Connect to WebSocket and stream workflow events.

        Args:
            workflow_id: Workflow ID to stream
        """
        # Build WebSocket URL with optional API key
        ws_uri = f"{self.ws_url}/api/v1/ws/workflows/{workflow_id}"
        if self.api_key:
            ws_uri += f"?api_key={self.api_key}"

        print(f"\n🔌 Connecting to WebSocket: {ws_uri}")

        try:
            async with websockets.connect(ws_uri) as websocket:
                print("✅ WebSocket connected\n")

                # Event handlers
                async def handle_event(event: dict):
                    """Process incoming events."""
                    event_type = event.get("type")

                    if event_type == "connected":
                        print("📡 Connected to workflow stream")
                        print(f"   Workflow: {event.get('workflow_id')}")

                    elif event_type == "workflow_status":
                        status = event.get("status")
                        print(f"📊 Workflow status: {status}")

                    elif event_type == "agent_spawned":
                        agent_id = event.get("agent_id")
                        agent_type = event.get("agent_type")
                        print(f"🤖 Agent spawned: {agent_id} ({agent_type})")

                    elif event_type == "agent_output":
                        agent_id = event.get("agent_id")
                        confidence = event.get("confidence", 0.0)
                        print(f"💬 Agent output: {agent_id} (confidence: {confidence:.2f})")

                    elif event_type == "synthesis_started":
                        agent_count = event.get("agent_count", 0)
                        print(f"🔄 Synthesis started with {agent_count} agents")

                    elif event_type == "workflow_complete":
                        synthesis = event.get("synthesis", {})
                        confidence = synthesis.get("confidence", 0.0)
                        print(f"\n✅ Workflow completed!")
                        print(f"   Confidence: {confidence:.2%}")
                        print(f"\n📝 Synthesis Result:")
                        print("=" * 60)
                        print(synthesis.get("content", "No content"))
                        print("=" * 60)

                    elif event_type == "workflow_error":
                        error = event.get("error", "Unknown error")
                        print(f"\n❌ Workflow failed: {error}")

                    elif event_type == "ping":
                        # Keepalive ping - respond with pong
                        await websocket.send("pong")

                    else:
                        print(f"❓ Unknown event: {event_type}")

                # Event loop
                async for message in websocket:
                    try:
                        # Parse JSON event
                        if message == "pong":
                            continue  # Response to our ping

                        event = json.loads(message)
                        await handle_event(event)

                        # Check if workflow finished
                        if event.get("type") in ["workflow_complete", "workflow_error"]:
                            break

                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON: {message}")
                    except Exception as e:
                        print(f"⚠️ Error processing event: {e}")

        except websockets.exceptions.WebSocketException as e:
            print(f"❌ WebSocket error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    async def run(self, task: str, max_steps: int = 10):
        """
        Main workflow: Create workflow and stream events.

        Args:
            task: Task description
            max_steps: Maximum workflow steps
        """
        print("=" * 60)
        print("Felix WebSocket Client")
        print("=" * 60)

        # Start Felix if not running
        if not await self.start_felix():
            return

        # Create workflow
        workflow_id = await self.create_workflow(task, max_steps)
        if not workflow_id:
            return

        # Stream events
        await self.stream_workflow(workflow_id)

        print("\n✅ Done!")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main entry point."""
    # Create client
    client = FelixWebSocketClient(
        api_url=API_URL,
        ws_url=WS_URL,
        api_key=API_KEY if API_KEY != "your-secret-api-key-here" else None
    )

    # Example tasks
    tasks = [
        "Explain quantum computing in simple terms",
        "Analyze the pros and cons of renewable energy",
        "What are the key differences between Python and JavaScript?",
    ]

    # Choose task
    print("\nAvailable tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")
    print(f"{len(tasks) + 1}. Custom task")

    try:
        choice = input(f"\nSelect task (1-{len(tasks) + 1}): ").strip()
        choice_num = int(choice)

        if 1 <= choice_num <= len(tasks):
            task = tasks[choice_num - 1]
        else:
            task = input("Enter custom task: ").strip()

        if not task:
            print("Error: Task cannot be empty")
            return

        # Run workflow with streaming
        await client.run(task, max_steps=10)

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
