@echo off
cd /d "%~dp0"

set PYTHON_CMD=

if exist "C:\Users\Admin\AppData\Local\Microsoft\WindowsApps\python3.13.exe" (
    set PYTHON_CMD=C:\Users\Admin\AppData\Local\Microsoft\WindowsApps\python3.13.exe
) else if exist "%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python3.13.exe" (
    set PYTHON_CMD=%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python3.13.exe
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set PYTHON_CMD=py -3.11
    )
)

if "%PYTHON_CMD%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        set PYTHON_CMD=python
    ) else (
        where python3 >nul 2>nul
        if not errorlevel 1 (
            set PYTHON_CMD=python3
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo No Python interpreter found. Install Python 3.11+ and retry.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_CMD%

%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

if not exist "data\raw\eo\demo_eo.png" (
    echo Generating demo data...
    %PYTHON_CMD% tools/generate_demo_data.py
)

if not exist "outputs\visualizations" mkdir "outputs\visualizations"
if not exist "outputs\reports" mkdir "outputs\reports"
if not exist "outputs\state" mkdir "outputs\state"

echo Starting UAV dashboard...
%PYTHON_CMD% -m streamlit run app/dashboard.py
