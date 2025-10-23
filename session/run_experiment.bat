@echo off
REM Quick launcher for P300-CIT Experiment on Windows
REM Double-click this file to run the experiment

echo ========================================
echo P300-Based Concealed Information Test
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run experiment
echo Starting experiment...
python src\experiment.py

pause

