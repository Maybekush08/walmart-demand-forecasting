@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Walmart Demand Forecasting - Automatic Setup and Training
cd /d "%~dp0"

echo.
echo ================================================================
echo  Walmart Demand Forecasting - Automatic All-Model Runner
echo ================================================================
echo  Project: %CD%
echo.

set "REPORT_FILE=%CD%\experiment_report.html"
set "PID_FILE=%CD%\experiment_run.pid"
set "RUN_LOG=%CD%\experiment_run.log"
set "ERROR_LOG=%CD%\experiment_error.log"
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

rem A second double-click reports an existing active run instead of duplicating it.
if exist "%PID_FILE%" (
    set /p EXISTING_PID=<"%PID_FILE%"
    powershell.exe -NoProfile -Command "if (Get-Process -Id !EXISTING_PID! -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
    if not errorlevel 1 (
        echo Training is already running in the background with PID !EXISTING_PID!.
        echo Report: %REPORT_FILE%
        echo Log:    %RUN_LOG%
        if exist "%REPORT_FILE%" start "" "%REPORT_FILE%"
        echo.
        pause
        exit /b 0
    )
    del /q "%PID_FILE%" >nul 2>&1
)

rem Do not accidentally retrain after a successfully completed full run.
if exist "%REPORT_FILE%" (
    findstr /C:"&quot;run_status&quot;: &quot;completed&quot;" "%REPORT_FILE%" >nul 2>&1
    if not errorlevel 1 (
        echo The all-model run is already complete. Opening the report.
        start "" "%REPORT_FILE%"
        echo Delete experiment_report.html only if you intentionally want a new run.
        echo.
        pause
        exit /b 0
    )
)

rem Verify the local M5 data before installing or training anything.
for %%F in (calendar.csv sell_prices.csv sales_train_evaluation.csv) do (
    if not exist "%CD%\data\m5\%%F" (
        echo ERROR: Missing data\m5\%%F
        echo Place all three M5 CSV files under %CD%\data\m5\
        echo.
        pause
        exit /b 1
    )
)
echo [1/5] M5 data files found.

rem Reuse the virtual environment when it already exists.
if exist "%VENV_PYTHON%" goto VENV_READY

set "BOOTSTRAP_PYTHON="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "BOOTSTRAP_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)

if not defined BOOTSTRAP_PYTHON (
    where python.exe >nul 2>&1
    if not errorlevel 1 (
        python.exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
        if not errorlevel 1 set "BOOTSTRAP_PYTHON=python.exe"
    )
)

if not defined BOOTSTRAP_PYTHON (
    echo [2/5] Python was not found. Installing Python 3.12 for the current user...
    where winget.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Neither Python nor Windows Package Manager was found.
        echo Install Python 3.12 from https://www.python.org/downloads/windows/
        echo and then double-click this file again.
        echo.
        pause
        exit /b 1
    )
    winget.exe install --exact --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo ERROR: Python installation failed.
        pause
        exit /b 1
    )
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "BOOTSTRAP_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    )
)

if not defined BOOTSTRAP_PYTHON (
    echo ERROR: Python 3.12 was installed but its executable could not be located.
    echo Close this window, reopen it, and double-click this file again.
    pause
    exit /b 1
)

echo [2/5] Creating the project virtual environment...
"%BOOTSTRAP_PYTHON%" -m venv "%CD%\.venv"
if errorlevel 1 (
    echo ERROR: Virtual environment creation failed.
    pause
    exit /b 1
)

:VENV_READY
echo [2/5] Python environment is ready.
echo [3/5] Installing or updating project dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto INSTALL_FAILED
"%VENV_PYTHON%" -m pip install -e ".[dev]"
if errorlevel 1 goto INSTALL_FAILED

echo [4/5] Running automated checks...
"%VENV_PYTHON%" -m pytest -q
if errorlevel 1 (
    echo ERROR: Tests failed. Training was not started.
    pause
    exit /b 1
)

echo [5/5] Starting all five models sequentially in the background...
for /f %%P in ('powershell.exe -NoProfile -Command "$process = Start-Process -FilePath '%VENV_PYTHON%' -ArgumentList @('-u','scripts\run_all.py') -WorkingDirectory '%CD%' -RedirectStandardOutput '%RUN_LOG%' -RedirectStandardError '%ERROR_LOG%' -WindowStyle Hidden -PassThru; $process.Id"') do set "RUN_PID=%%P"

if not defined RUN_PID (
    echo ERROR: The background training process could not be started.
    pause
    exit /b 1
)

>"%PID_FILE%" echo !RUN_PID!
echo.
echo Background training started successfully with PID !RUN_PID!.
echo You may close this window. The laptop must remain powered on and awake.
echo.
echo Consolidated report: %REPORT_FILE%
echo Progress log:       %RUN_LOG%
echo Error log:          %ERROR_LOG%
echo.
echo Double-click START_ALL_MODELS.cmd again to check/open the active report.
pause
exit /b 0

:INSTALL_FAILED
echo ERROR: Dependency installation failed. Review the messages above.
pause
exit /b 1
