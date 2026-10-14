@echo off
cd /d "%~dp0"
title VisionCare Server

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
  echo Python not found. Install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)

echo Installing dependencies if needed...
python -m pip install -q -r requirements.txt 2>nul

echo.
echo Starting VisionCare at http://127.0.0.1:8001
echo Keep this window open while using the app.
echo Press Ctrl+C to stop.
echo.

start "" "http://127.0.0.1:8001"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload

pause
