@echo off
REM OpenHands Enhanced Setup Launcher for Windows 11
REM This script provides an interactive setup experience

echo.
echo ========================================
echo   OpenHands Enhanced Setup Launcher
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires Administrator privileges.
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo Select installation type:
echo 1. Local installation (recommended for development)
echo 2. Docker installation (recommended for production)
echo 3. Lightning Labs cloud installation (recommended for heavy workloads)
echo.
set /p install_type="Enter your choice (1-3): "

echo.
echo Select stability level:
echo 1. Stable (production-ready features)
echo 2. Testing (not yet verified stable features)
echo 3. Experimental (cutting-edge features)
echo.
set /p stability="Enter your choice (1-3): "

echo.
set /p performance="Enter performance level (10-100%%): "

echo.
echo Additional features:
set /p idle_improvement="Enable self-improvement on idle? (y/n): "
set /p vm_mode="Enable Lightning Labs VM mode? (y/n): "
set /p scifi_ui="Enable Sci-Fi themed UI (eDEX-UI)? (y/n): "
set /p multi_api="Enable multi-API account management? (y/n): "

REM Convert choices to PowerShell parameters
if "%install_type%"=="1" set ps_install=local
if "%install_type%"=="2" set ps_install=docker
if "%install_type%"=="3" set ps_install=lightning

if "%stability%"=="1" set ps_stability=stable
if "%stability%"=="2" set ps_stability=testing
if "%stability%"=="3" set ps_stability=experimental

if /i "%idle_improvement%"=="y" (set ps_idle=true) else (set ps_idle=false)
if /i "%vm_mode%"=="y" (set ps_vm=true) else (set ps_vm=false)
if /i "%scifi_ui%"=="y" (set ps_scifi=true) else (set ps_scifi=false)
if /i "%multi_api%"=="y" (set ps_multi=true) else (set ps_multi=false)

echo.
echo Starting OpenHands Enhanced Setup...
echo Installation Type: %ps_install%
echo Stability Level: %ps_stability%
echo Performance Level: %performance%%%
echo Idle Improvement: %ps_idle%
echo VM Mode: %ps_vm%
echo Sci-Fi UI: %ps_scifi%
echo Multi-API: %ps_multi%
echo.

REM Run PowerShell setup script
powershell -ExecutionPolicy Bypass -File "%~dp0setup_openhands_windows.ps1" -InstallType %ps_install% -StabilityLevel %ps_stability% -PerformanceLevel %performance% -EnableIdleImprovement %ps_idle% -EnableVMMode %ps_vm% -EnableSciFiUI %ps_scifi% -EnableMultiAPI %ps_multi%

if %errorLevel% equ 0 (
    echo.
    echo ========================================
    echo   Setup completed successfully!
    echo ========================================
    echo.
    echo OpenHands Enhanced is now ready to use.
    echo.
    echo Access points:
    echo - Main WebUI: http://localhost:8000
    if /i "%ps_scifi%"=="true" echo - Sci-Fi UI: http://localhost:3001
    echo - API Documentation: http://localhost:8000/docs
    echo.
    echo Press any key to open the WebUI...
    pause >nul
    start http://localhost:8000
) else (
    echo.
    echo ========================================
    echo   Setup failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo.
    pause
)