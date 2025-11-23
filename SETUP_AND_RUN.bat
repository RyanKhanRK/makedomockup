@echo off
color 0A
cls

echo ================================================================
echo   MLflow Extended Dashboard - Complete Setup
echo   Senior Project: Ryan Khan (64070503446)
echo ================================================================
echo.

REM Step 1: Install dependencies
echo [STEP 1] Installing Python dependencies...
pip install mlflow scikit-learn shap pandas numpy matplotlib seaborn flask flask-cors

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Step 2: Create directories
echo [STEP 2] Creating project structure...
if not exist "mlruns" mkdir mlruns
if not exist "artifacts" mkdir artifacts
if not exist "data" mkdir data
echo [OK] Directories created
echo.

REM Step 3: Check datasets
echo [STEP 3] Checking for datasets...
if not exist "titanic.csv" (
    echo [INFO] Titanic dataset not found. Scripts will use seaborn's built-in dataset.
) else (
    echo [OK] Titanic dataset found
)
echo.

REM Step 4: Start MLflow server
echo [STEP 4] Starting MLflow tracking server...
echo Starting on http://localhost:5000
echo.

REM Kill existing MLflow processes
taskkill /F /IM python.exe /FI "WINDOWTITLE eq mlflow*" 2>nul

REM Start MLflow in new window
start "MLflow Server" cmd /k "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns"

echo Waiting for MLflow to start...
timeout /t 5 /nobreak >nul
echo [OK] MLflow server started
echo.

REM Step 5: Run experiments
echo [STEP 5] Running ML experiments...
echo.

set /p TITANIC="Run Titanic experiments (UC1 & UC2)? [Y/n]: "
if /i "%TITANIC%"=="Y" goto run_titanic
if /i "%TITANIC%"=="" goto run_titanic
goto skip_titanic

:run_titanic
echo Running Titanic experiments...
python titanic_shap_example.py
if %ERRORLEVEL% EQU 0 (
    echo [OK] Titanic experiments complete
) else (
    echo [ERROR] Titanic experiments failed
)
echo.

:skip_titanic
set /p IRIS="Run Iris experiments (UC3)? [Y/n]: "
if /i "%IRIS%"=="Y" goto run_iris
if /i "%IRIS%"=="" goto run_iris
goto skip_iris

:run_iris
echo Running Iris experiments...
python iris_shap_example.py
if %ERRORLEVEL% EQU 0 (
    echo [OK] Iris experiments complete
) else (
    echo [ERROR] Iris experiments failed
)
echo.

:skip_iris

REM Step 6: Open dashboard
echo [STEP 6] Opening dashboard...
echo.
start mlflow-dashboard.html

echo.
echo ================================================================
echo                   SETUP COMPLETE!
echo ================================================================
echo.
echo Access your dashboard:
echo   - Extended Dashboard: mlflow-dashboard.html
echo   - MLflow UI: http://localhost:5000
echo.
echo MLflow server is running in a separate window
echo Close that window to stop the server
echo.
echo Next steps:
echo   1. Explore experiments in the dashboard
echo   2. Check SHAP plots in Run Details
echo   3. Compare different runs
echo   4. View fairness analysis
echo.
pause
