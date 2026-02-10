@echo off
REM EEG FIF Analyzer - Streamlit Application Launcher
REM Windows Batch Script

echo.
echo ========================================
echo  EEG FIF Analyzer - Streamlit App
echo ========================================
echo.

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit not installed!
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Failed to install dependencies.
        echo Please run: pip install streamlit mne
        pause
        exit /b 1
    )
)

REM Check if MNE is installed
python -c "import mne" 2>nul
if errorlevel 1 (
    echo ERROR: MNE-Python not installed!
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo Starting Streamlit application...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run scripts/eeg_analyzer_app.py

pause
