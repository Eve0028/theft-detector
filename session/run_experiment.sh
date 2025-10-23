#!/bin/bash
# Quick launcher for P300-CIT Experiment on Linux/Mac

echo "========================================"
echo "P300-Based Concealed Information Test"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run experiment
echo "Starting experiment..."
python src/experiment.py

read -p "Press Enter to exit..."

