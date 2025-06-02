@echo off
setlocal enabledelayedexpansion

:: OpenHands Enhanced Auto-Setup Script for Windows 11
:: Supports local installation, Docker deployment, and Lightning Labs integration
:: With model switching, sci-fi UI, and auto-optimization features

title OpenHands Enhanced Setup - Windows 11

:: Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo ========================================
    echo   ADMINISTRATOR PRIVILEGES REQUIRED
    echo ========================================
    echo.
    echo This script requires administrator privileges to install dependencies.
    echo Please right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

:: Display banner
echo.
echo ████████████████████████████████████████████████████████████████
echo █                                                              █
echo █    ██████╗ ██████╗ ███████╗███╗   ██╗██╗  ██╗ █████╗ ███╗   ██╗██████╗ ███████╗    █
echo █   ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║  ██║██╔══██╗████╗  ██║██╔══██╗██╔════╝    █
echo █   ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║███████║██╔██╗ ██║██║  ██║███████╗    █
echo █   ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██╔══██║██║╚██╗██║██║  ██║╚════██║    █
echo █   ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║  ██║██║ ╚████║██████╔╝███████║    █
echo █    ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝    █
echo █                                                              █
echo █                    ENHANCED AUTO-SETUP                      █
echo █              Model Switching • Sci-Fi UI • Auto-Optimization █
echo █                                                              █
echo ████████████████████████████████████████████████████████████████
echo.

:: Configuration variables
set INSTALL_TYPE=
set STABILITY_LEVEL=
set PERFORMANCE_LEVEL=
set ENABLE_IDLE_IMPROVEMENT=
set ENABLE_VM_MODE=
set ENABLE_SCIFI_UI=
set ENABLE_MULTI_API=
set LIGHTNING_LABS_ENABLED=
set EDEX_UI_ENABLED=

:: Step 1: Installation Type Selection
echo ========================================
echo   STEP 1: INSTALLATION TYPE
echo ========================================
echo.
echo Choose your installation type:
echo.
echo 1. Local Installation (Recommended for development)
echo    - Installs directly on your system
echo    - Full control and customization
echo    - Requires Python 3.11+
echo.
echo 2. Docker Installation (Recommended for production)
echo    - Containerized deployment
echo    - Easy scaling and management
echo    - Requires Docker Desktop
echo.
echo 3. Lightning Labs Cloud (Recommended for heavy workloads)
echo    - Cloud-based deployment
echo    - Auto-scaling capabilities
echo    - Requires Lightning Labs account
echo.
set /p INSTALL_TYPE="Enter your choice (1-3): "

if "%INSTALL_TYPE%"=="1" (
    set INSTALL_TYPE=local
    echo Selected: Local Installation
) else if "%INSTALL_TYPE%"=="2" (
    set INSTALL_TYPE=docker
    echo Selected: Docker Installation
) else if "%INSTALL_TYPE%"=="3" (
    set INSTALL_TYPE=lightning
    echo Selected: Lightning Labs Cloud
) else (
    echo Invalid choice. Defaulting to Local Installation.
    set INSTALL_TYPE=local
)

echo.
pause

:: Step 2: Stability Level Selection
echo ========================================
echo   STEP 2: STABILITY LEVEL
echo ========================================
echo.
echo Choose your stability level:
echo.
echo 1. Stable (Production-ready features only)
echo    - Thoroughly tested features
echo    - Maximum reliability
echo    - Recommended for production use
echo.
echo 2. Testing (Beta features with safety nets)
echo    - New features with extensive testing
echo    - Good balance of features and stability
echo    - Recommended for development
echo.
echo 3. Experimental (Cutting-edge capabilities)
echo    - Latest features and improvements
echo    - May have occasional issues
echo    - Recommended for research and testing
echo.
set /p STABILITY_CHOICE="Enter your choice (1-3): "

if "%STABILITY_CHOICE%"=="1" (
    set STABILITY_LEVEL=stable
    echo Selected: Stable
) else if "%STABILITY_CHOICE%"=="2" (
    set STABILITY_LEVEL=testing
    echo Selected: Testing
) else if "%STABILITY_CHOICE%"=="3" (
    set STABILITY_LEVEL=experimental
    echo Selected: Experimental
) else (
    echo Invalid choice. Defaulting to Stable.
    set STABILITY_LEVEL=stable
)

echo.
pause

:: Step 3: Performance Level Selection
echo ========================================
echo   STEP 3: PERFORMANCE LEVEL
echo ========================================
echo.
echo Choose your performance level (10-100%%):
echo.
echo 10-30%%: Low resource usage, basic features
echo 40-70%%: Balanced performance and features (Recommended)
echo 80-100%%: Maximum performance, all features enabled
echo.
set /p PERFORMANCE_LEVEL="Enter performance level (10-100): "

:: Validate performance level
if %PERFORMANCE_LEVEL% LSS 10 set PERFORMANCE_LEVEL=10
if %PERFORMANCE_LEVEL% GTR 100 set PERFORMANCE_LEVEL=100

echo Selected: %PERFORMANCE_LEVEL%%% performance level
echo.
pause

:: Step 4: Feature Selection
echo ========================================
echo   STEP 4: FEATURE SELECTION
echo ========================================
echo.

:: Idle Improvement
echo Enable Idle Improvement System?
echo - Automatically optimizes OpenHands when system is idle
echo - Improves performance, accuracy, and efficiency
echo - Safe rollback on errors
set /p IDLE_CHOICE="Enable Idle Improvement? (y/n): "
if /i "%IDLE_CHOICE%"=="y" (
    set ENABLE_IDLE_IMPROVEMENT=true
    echo ✓ Idle Improvement enabled
) else (
    set ENABLE_IDLE_IMPROVEMENT=false
    echo ✗ Idle Improvement disabled
)

echo.

:: VM Mode
echo Enable VM Mode (Lightning Labs Integration)?
echo - Offloads heavy processing to cloud
echo - Prevents local system slowdown
echo - Requires Lightning Labs account
set /p VM_CHOICE="Enable VM Mode? (y/n): "
if /i "%VM_CHOICE%"=="y" (
    set ENABLE_VM_MODE=true
    set LIGHTNING_LABS_ENABLED=true
    echo ✓ VM Mode enabled
) else (
    set ENABLE_VM_MODE=false
    set LIGHTNING_LABS_ENABLED=false
    echo ✗ VM Mode disabled
)

echo.

:: Sci-Fi UI
echo Enable Sci-Fi UI (eDEX-UI Integration)?
echo - Futuristic terminal-style interface
echo - Real-time system monitoring
echo - World map visualization
echo - Live metrics display
set /p SCIFI_CHOICE="Enable Sci-Fi UI? (y/n): "
if /i "%SCIFI_CHOICE%"=="y" (
    set ENABLE_SCIFI_UI=true
    set EDEX_UI_ENABLED=true
    echo ✓ Sci-Fi UI enabled
) else (
    set ENABLE_SCIFI_UI=false
    set EDEX_UI_ENABLED=false
    echo ✗ Sci-Fi UI disabled
)

echo.

:: Multi-API
echo Enable Multi-API Account Management?
echo - Use multiple API accounts for different providers
echo - Automatic account rotation
echo - Increased rate limits and reliability
set /p MULTI_API_CHOICE="Enable Multi-API Management? (y/n): "
if /i "%MULTI_API_CHOICE%"=="y" (
    set ENABLE_MULTI_API=true
    echo ✓ Multi-API Management enabled
) else (
    set ENABLE_MULTI_API=false
    echo ✗ Multi-API Management disabled
)

echo.
pause

:: Step 5: Dependency Installation
echo ========================================
echo   STEP 5: INSTALLING DEPENDENCIES
echo ========================================
echo.

:: Check if Chocolatey is installed
choco --version >nul 2>&1
if %errorLevel% neq 0 (
    echo Installing Chocolatey package manager...
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    if %errorLevel% neq 0 (
        echo Failed to install Chocolatey. Please install manually.
        pause
        exit /b 1
    )
    echo ✓ Chocolatey installed successfully
) else (
    echo ✓ Chocolatey already installed
)

:: Install required software based on installation type
if "%INSTALL_TYPE%"=="local" (
    echo Installing Python 3.11, Node.js, and Git...
    choco install python311 nodejs git -y
    if %errorLevel% neq 0 (
        echo Warning: Some packages may have failed to install
    )
    echo ✓ Local dependencies installed
)

if "%INSTALL_TYPE%"=="docker" (
    echo Installing Docker Desktop, Python 3.11, and Git...
    choco install docker-desktop python311 git -y
    if %errorLevel% neq 0 (
        echo Warning: Some packages may have failed to install
    )
    echo ✓ Docker dependencies installed
)

if "%INSTALL_TYPE%"=="lightning" (
    echo Installing Python 3.11, Node.js, and Git for Lightning Labs...
    choco install python311 nodejs git -y
    if %errorLevel% neq 0 (
        echo Warning: Some packages may have failed to install
    )
    echo ✓ Lightning Labs dependencies installed
)

echo.
pause

:: Step 6: Repository Setup
echo ========================================
echo   STEP 6: REPOSITORY SETUP
echo ========================================
echo.

:: Check if repository already exists
if exist "openhands" (
    echo Repository already exists. Updating...
    cd openhands
    git pull origin arxiv-research-improvements
    cd ..
) else (
    echo Cloning OpenHands Enhanced repository...
    git clone https://github.com/Subikshaa1910/openhands.git
    if %errorLevel% neq 0 (
        echo Failed to clone repository. Please check your internet connection.
        pause
        exit /b 1
    )
)

cd openhands
git checkout arxiv-research-improvements

echo ✓ Repository setup complete
echo.
pause

:: Step 7: Configuration Generation
echo ========================================
echo   STEP 7: CONFIGURATION GENERATION
echo ========================================
echo.

echo Generating configuration file...

:: Create .env file with user selections
(
echo # OpenHands Enhanced Configuration
echo # Generated by auto-setup script
echo.
echo # Installation Settings
echo INSTALL_TYPE=%INSTALL_TYPE%
echo STABILITY_LEVEL=%STABILITY_LEVEL%
echo PERFORMANCE_LEVEL=%PERFORMANCE_LEVEL%
echo.
echo # Feature Flags
echo ENABLE_IDLE_IMPROVEMENT=%ENABLE_IDLE_IMPROVEMENT%
echo ENABLE_VM_MODE=%ENABLE_VM_MODE%
echo ENABLE_SCIFI_UI=%ENABLE_SCIFI_UI%
echo ENABLE_MULTI_API=%ENABLE_MULTI_API%
echo.
echo # Security Configuration
echo ADMIN_TOKEN=%RANDOM%%RANDOM%%RANDOM%
echo ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:8000
echo.
echo # Performance Settings
echo MAX_CONCURRENT_REQUESTS=10
echo REQUEST_TIMEOUT=30
echo MEMORY_LIMIT_GB=4
echo CPU_LIMIT=%PERFORMANCE_LEVEL%
echo.
echo # Idle Improvement Settings
echo IDLE_THRESHOLD_MINUTES=10
echo IDLE_IMPROVEMENT_INTERVAL=3600
echo AC_POWER_REQUIRED=true
echo SYSTEM_IDLE=%ENABLE_IDLE_IMPROVEMENT%
echo VM_IDLE=%ENABLE_VM_MODE%
echo.
echo # Lightning Labs Settings
echo LIGHTNING_LABS_ENABLED=%LIGHTNING_LABS_ENABLED%
echo LIGHTNING_LABS_INSTANCE_TYPE=cpu-small
echo.
echo # Sci-Fi UI Settings
echo EDEX_UI_ENABLED=%EDEX_UI_ENABLED%
echo EDEX_UI_PORT=3001
echo WORLD_MAP_API_ENABLED=true
echo.
echo # Multi-API Settings
echo MULTI_API_ENABLED=%ENABLE_MULTI_API%
echo ACCOUNT_ROTATION_INTERVAL=30
echo MAX_ACCOUNTS_PER_PROVIDER=5
echo.
echo # Monitoring Settings
echo ENABLE_METRICS=true
echo METRICS_PORT=9090
echo ENABLE_LOGGING=true
echo LOG_LEVEL=INFO
) > .env

echo ✓ Configuration file generated
echo.
pause

:: Step 8: Installation Process
echo ========================================
echo   STEP 8: INSTALLATION PROCESS
echo ========================================
echo.

if "%INSTALL_TYPE%"=="local" (
    call :install_local
) else if "%INSTALL_TYPE%"=="docker" (
    call :install_docker
) else if "%INSTALL_TYPE%"=="lightning" (
    call :install_lightning
)

:: Step 9: Post-Installation Setup
echo ========================================
echo   STEP 9: POST-INSTALLATION SETUP
echo ========================================
echo.

:: Setup eDEX-UI if enabled
if "%ENABLE_SCIFI_UI%"=="true" (
    echo Setting up eDEX-UI for Sci-Fi interface...
    if not exist "edex-ui" (
        git clone https://github.com/GitSquared/edex-ui.git edex-ui
        cd edx-ui
        npm install
        cd ..
    )
    echo ✓ eDEX-UI setup complete
)

:: Setup Lightning Labs if enabled
if "%LIGHTNING_LABS_ENABLED%"=="true" (
    echo Setting up Lightning Labs integration...
    pip install lightning lightning-sdk
    echo ✓ Lightning Labs setup complete
    echo.
    echo To complete Lightning Labs setup:
    echo 1. Run: lightning login
    echo 2. Follow the authentication process
    echo 3. Your cloud integration will be ready
)

echo.
echo ========================================
echo   INSTALLATION COMPLETE!
echo ========================================
echo.
echo OpenHands Enhanced has been successfully installed with the following configuration:
echo.
echo Installation Type: %INSTALL_TYPE%
echo Stability Level: %STABILITY_LEVEL%
echo Performance Level: %PERFORMANCE_LEVEL%%%
echo Idle Improvement: %ENABLE_IDLE_IMPROVEMENT%
echo VM Mode: %ENABLE_VM_MODE%
echo Sci-Fi UI: %ENABLE_SCIFI_UI%
echo Multi-API: %ENABLE_MULTI_API%
echo.
echo Access Points:
echo - Main WebUI: http://localhost:8000
if "%ENABLE_SCIFI_UI%"=="true" (
    echo - Sci-Fi UI: http://localhost:3001
)
echo - API Documentation: http://localhost:8000/docs
echo.
echo Next Steps:
echo 1. Add your API keys in the Multi-API Management panel
echo 2. Configure your preferred providers and models
echo 3. Test the model switching functionality
echo 4. Explore the enhanced features and optimization options
echo.
echo For detailed documentation, see:
echo - ENHANCED_SETUP_GUIDE.md
echo - README_ENHANCED.md
echo - AUTO_OPTIMIZATION_PLAN.md
echo.

if "%INSTALL_TYPE%"=="local" (
    echo To start OpenHands Enhanced:
    echo python -m src.api.server
) else if "%INSTALL_TYPE%"=="docker" (
    echo To start OpenHands Enhanced:
    echo docker-compose -f docker-compose.enhanced.yml up -d
)

echo.
echo Thank you for using OpenHands Enhanced!
echo.
pause
exit /b 0

:: Installation functions
:install_local
echo Installing OpenHands Enhanced locally...
echo.

:: Create virtual environment
echo Creating Python virtual environment...
python -m venv venv
if %errorLevel% neq 0 (
    echo Failed to create virtual environment
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-enhanced.txt

if %errorLevel% neq 0 (
    echo Warning: Some dependencies may have failed to install
)

echo ✓ Local installation complete
goto :eof

:install_docker
echo Installing OpenHands Enhanced with Docker...
echo.

:: Start Docker Desktop if not running
echo Checking Docker Desktop status...
docker version >nul 2>&1
if %errorLevel% neq 0 (
    echo Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to start...
    timeout /t 30 /nobreak
)

:: Build and start containers
echo Building Docker containers...
if "%ENABLE_SCIFI_UI%"=="true" (
    docker-compose -f docker-compose.enhanced.yml --profile scifi-ui up -d --build
) else (
    docker-compose -f docker-compose.enhanced.yml up -d --build
)

if %errorLevel% neq 0 (
    echo Failed to start Docker containers
    exit /b 1
)

echo ✓ Docker installation complete
goto :eof

:install_lightning
echo Installing OpenHands Enhanced for Lightning Labs...
echo.

:: Create virtual environment
echo Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Install dependencies including Lightning SDK
echo Installing dependencies with Lightning SDK...
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-enhanced.txt
pip install lightning lightning-sdk

if %errorLevel% neq 0 (
    echo Warning: Some dependencies may have failed to install
)

echo ✓ Lightning Labs installation complete
goto :eof