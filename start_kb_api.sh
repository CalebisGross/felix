#!/bin/bash
# Start Felix API with Knowledge Brain enabled

echo "=========================================="
echo "Starting Felix API with Knowledge Brain"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Set environment variables
export FELIX_ENABLE_KNOWLEDGE_BRAIN=true
export FELIX_KNOWLEDGE_WATCH_DIRS="./knowledge_sources"

echo ""
echo "Environment variables set:"
echo "  FELIX_ENABLE_KNOWLEDGE_BRAIN=$FELIX_ENABLE_KNOWLEDGE_BRAIN"
echo "  FELIX_KNOWLEDGE_WATCH_DIRS=$FELIX_KNOWLEDGE_WATCH_DIRS"
echo ""
echo "Starting server on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="

# Start the server
python3 -m uvicorn src.api.main:app --reload --port 8000
