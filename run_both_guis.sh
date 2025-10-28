#!/bin/bash
# Startup script for running both Felix GUIs on Unix/Linux/Mac
# This script launches the tkinter control GUI and Streamlit monitoring GUI

echo "==============================================="
echo "   Felix Framework - Dual GUI Launcher"
echo "==============================================="
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}ERROR: Virtual environment not found!${NC}"
    echo "Please run: python3 -m venv .venv"
    echo "Then: source .venv/bin/activate"
    echo "And: pip install -r requirements.txt"
    echo "     pip install -r requirements_streamlit.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check for required packages
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Streamlit not installed!${NC}"
    echo "Please run: pip install -r requirements_streamlit.txt"
    exit 1
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

echo
echo "Starting Felix Framework GUIs..."
echo "-----------------------------------------------"
echo

# Start tkinter GUI in background
echo -e "${GREEN}[1/2] Launching tkinter Control GUI...${NC}"

# Check if display is available (for headless systems)
if [ -z "$DISPLAY" ]; then
    echo -e "${YELLOW}WARNING: No display detected. tkinter GUI may not work.${NC}"
    echo "Consider running with: export DISPLAY=:0"
fi

# Start tkinter in new terminal based on available terminal emulator
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal --title="Felix Control GUI" -- bash -c "source .venv/bin/activate && python -m src.gui.main; exec bash"
elif command -v konsole &> /dev/null; then
    konsole -e bash -c "source .venv/bin/activate && python -m src.gui.main; exec bash" &
elif command -v xterm &> /dev/null; then
    xterm -title "Felix Control GUI" -e bash -c "source .venv/bin/activate && python -m src.gui.main; exec bash" &
elif command -v open &> /dev/null; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$PWD' && source .venv/bin/activate && python -m src.gui.main"'
else
    # Fallback: run in background
    echo "No terminal emulator found. Running in background..."
    python -m src.gui.main &
    TKINTER_PID=$!
fi

# Wait a moment for tkinter to start
sleep 3

# Start Streamlit GUI
echo -e "${GREEN}[2/2] Launching Streamlit Monitor GUI...${NC}"

# Check if port 8501 is already in use
if check_port 8501; then
    echo -e "${YELLOW}WARNING: Port 8501 is already in use.${NC}"
    echo "Trying alternative port 8502..."
    STREAMLIT_PORT=8502
else
    STREAMLIT_PORT=8501
fi

# Start Streamlit using launcher script
python streamlit_gui/run_streamlit_gui.py &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 3

# Check if Streamlit started successfully
if check_port $STREAMLIT_PORT; then
    echo
    echo "==============================================="
    echo -e "${GREEN}   Both GUIs are running!${NC}"
    echo "==============================================="
    echo
    echo "tkinter GUI (Control):    Running in separate window"
    echo "Streamlit GUI (Monitor):  http://localhost:$STREAMLIT_PORT"
    echo
    echo "Instructions:"
    echo "1. Use tkinter GUI to start/stop Felix system"
    echo "2. Use Streamlit GUI to monitor performance"
    echo "3. Press Ctrl+C to stop both GUIs"
    echo

    # Open browser if available
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$STREAMLIT_PORT" 2>/dev/null
    elif command -v open &> /dev/null; then
        open "http://localhost:$STREAMLIT_PORT" 2>/dev/null
    fi

    echo -e "${YELLOW}Press Ctrl+C to stop both GUIs...${NC}"

    # Wait for interrupt
    trap cleanup INT

    cleanup() {
        echo
        echo "Shutting down GUIs..."
        if [ ! -z "$STREAMLIT_PID" ]; then
            kill $STREAMLIT_PID 2>/dev/null
        fi
        if [ ! -z "$TKINTER_PID" ]; then
            kill $TKINTER_PID 2>/dev/null
        fi
        echo -e "${GREEN}GUIs stopped successfully.${NC}"
        exit 0
    }

    # Keep script running
    while true; do
        sleep 1
    done

else
    echo -e "${RED}ERROR: Failed to start Streamlit GUI${NC}"
    exit 1
fi