@echo off
cd /d "%~dp0"
echo Starting Doctor Agent Service...

REM Reuse existing venv from sibling directory if possible
if exist "..\mcp-patient-data\venv\Scripts\activate.bat" (
    call "..\mcp-patient-data\venv\Scripts\activate.bat"
) else (
    echo VENV not found in ../mcp-patient-data! Please set up mcp server first.
    pause
    exit /b
)

echo Installing/Updating Dependencies...
pip install -r requirements.txt

echo running agent...
python agent.py
