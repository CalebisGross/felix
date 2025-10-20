"""
Cross-platform launcher for the Streamlit GUI.

This script provides a simple way to launch the Streamlit monitoring interface
with proper error handling and configuration.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import socket


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
        print("ERROR: Streamlit is not installed!")
        print("Please run: pip install -r requirements_streamlit.txt")
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
        print(f"Using port: {port}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Launch Streamlit
    print("Launching Streamlit GUI...")
    print("-" * 50)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
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
            print(f"\n✓ Streamlit GUI is running at: {url}")
            print("\nOpening browser...")

            # Try to open browser
            try:
                webbrowser.open(url)
            except:
                print("Could not open browser automatically.")
                print(f"Please open manually: {url}")

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
            print("ERROR: Streamlit failed to start")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nShutting down Streamlit GUI...")
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()
        print("✓ Streamlit GUI stopped")

    except Exception as e:
        print(f"ERROR: Failed to launch Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    launch_streamlit()