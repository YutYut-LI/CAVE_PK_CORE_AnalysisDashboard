@echo off
rem ---------------------------------------------------------------
rem  Launch the CAVE-PK dashboard. Double-click this file.
rem  Close the window (or press Ctrl+C twice) to stop the app.
rem ---------------------------------------------------------------
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title CAVE-PK dashboard

rem "python" on PATH is often the Microsoft Store placeholder, which cannot
rem run anything, so look for an interpreter that actually has streamlit.
set "PY="
for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "%LOCALAPPDATA%\anaconda3\python.exe"
  "C:\ProgramData\anaconda3\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
) do (
  if not defined PY if exist %%P (
    %%P -c "import streamlit" >nul 2>&1 && set "PY=%%~P"
  )
)
if not defined PY (
  py -3 -c "import streamlit" >nul 2>&1 && set "PY=py"
)
if not defined PY (
  echo.
  echo   Could not find a Python with streamlit installed.
  echo   Install it with:  pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo.
echo   Using: !PY!
git rev-parse --short HEAD 2>nul && echo   ^(run "git pull" first if you want the latest^)
echo   Starting... your browser will open at http://localhost:8501
echo   Leave this window open while you work.
echo.

rem poll: the repo sits in OneDrive, where filesystem change events are
rem unreliable, so Streamlit would otherwise keep serving stale code.
"!PY!" -m streamlit run app.py --server.port 8501 --server.fileWatcherType poll

echo.
echo   The app has stopped.
pause
