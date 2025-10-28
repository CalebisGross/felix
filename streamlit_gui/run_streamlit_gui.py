"""
Cross-platform launcher for the Streamlit GUI.

This script provides a simple way to launch the Streamlit monitoring interface
with proper error handling and configuration.
"""

import subprocess
import sys
import time
import webbrowser
import logging
from pathlib import Path
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def check_port_available(port: int) -> bool:
    """Check if a port is available for use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0


def find_available_port(start_port: int = 8501, max_tries: int = 10) -> int:
    """Find an available port starting from start_port."""
    for i in range(max_tries):
        port = start_port + i
        if check_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port+max_tries}")


def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        return True
    except ImportError:
        logging.error("Streamlit is not installed!")
        logging.info("Please run: pip install -r requirements_streamlit.txt")
        return False


def launch_streamlit():
    """Launch the Streamlit GUI."""
    print("=" * 50)
    print("   Felix Framework - Streamlit Monitor")
    print("=" * 50)
    print()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Find available port
    try:
        port = find_available_port()
        logging.info(f"Using port: {port}")
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)

    # Launch Streamlit
    logging.info("Launching Streamlit GUI...")
    print("-" * 50)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true"
    ]

    try:
        # Start Streamlit process
        process = subprocess.Popen(cmd)

        # Wait a bit for startup
        time.sleep(3)

        # Check if process is still running
        if process.poll() is None:
            url = f"http://localhost:{port}"
            logging.info(f"✓ Streamlit GUI is running at: {url}")
            logging.info("Opening browser...")

            # Try to open browser
            try:
                webbrowser.open(url)
            except Exception as e:
                logging.warning("Could not open browser automatically.")
                logging.info(f"Please open manually: {url}")

            print("\n" + "=" * 50)
            print("Instructions:")
            print("1. Use the sidebar to navigate between pages")
            print("2. Dashboard shows real-time system metrics")
            print("3. Configuration allows viewing Felix settings")
            print("4. Testing provides workflow analysis")
            print("5. Benchmarking validates hypotheses")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 50)

            # Wait for the process to complete
            process.wait()

        else:
            logging.error("Streamlit failed to start")
            sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n\nShutting down Streamlit GUI...")
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()
        logging.info("✓ Streamlit GUI stopped")

    except Exception as e:
        logging.error(f"Failed to launch Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    launch_streamlit()