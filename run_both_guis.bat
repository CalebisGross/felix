@echo off
REM Startup script for running both Felix GUIs on Windows
REM This script launches the tkinter control GUI and Streamlit monitoring GUI

echo ===============================================
echo    Felix Framework - Dual GUI Launcher
echo ===============================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\activate
    echo And: pip install -r requirements.txt
    echo      pip install -r requirements_streamlit.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check for required packages
python -c "import streamlit" 2>NUL
if errorlevel 1 (
    echo ERROR: Streamlit not installed!
    echo Please run: pip install -r requirements_streamlit.txt
    pause
    exit /b 1
)

echo.
echo Starting Felix Framework GUIs...
echo -----------------------------------------------
echo.

REM Start tkinter GUI in new window
echo [1/2] Launching tkinter Control GUI...
start "Felix Control GUI" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m src.gui.main"

REM Wait a moment for tkinter to start
timeout /t 3 /nobreak >NUL

REM Start Streamlit GUI in new window
echo [2/2] Launching Streamlit Monitor GUI...
start "Felix Monitor GUI" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python run_streamlit_gui.py"

echo.
echo ===============================================
echo    Both GUIs are starting...
echo ===============================================
echo.
echo tkinter GUI (Control):     http://localhost:8501 (window)
echo Streamlit GUI (Monitor):   http://localhost:8501 (browser)
echo.
echo Instructions:
echo 1. Use tkinter GUI to start/stop Felix system
echo 2. Use Streamlit GUI to monitor performance
echo 3. Close both windows to stop the GUIs
echo.
echo Press any key to close this launcher...
pause >NUL