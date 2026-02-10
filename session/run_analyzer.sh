#!/bin/bash
# EEG FIF Analyzer - Streamlit Application Launcher
# Linux/macOS Shell Script

echo ""
echo "========================================"
echo " EEG FIF Analyzer - Streamlit App"
echo "========================================"
echo ""

# Check if Streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "ERROR: Streamlit not installed!"
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "Failed to install dependencies."
        echo "Please run: pip install streamlit mne"
        exit 1
    fi
fi

# Check if MNE is installed
if ! python -c "import mne" 2>/dev/null; then
    echo "ERROR: MNE-Python not installed!"
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "Failed to install dependencies."
        exit 1
    fi
fi

echo "Starting Streamlit application..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run scripts/eeg_analyzer_app.py
