# OpenHands Enhanced Setup Guide for Windows 11

This guide provides comprehensive instructions for setting up OpenHands Enhanced with model switching, sci-fi UI, and advanced features on Windows 11.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

1. **Download the setup script**:
   ```bash
   git clone https://github.com/Subikshaa1910/openhands.git
   cd openhands
   ```

2. **Run the automated setup**:
   - Right-click on `setup_openhands.bat`
   - Select "Run as administrator"
   - Follow the interactive prompts

3. **Access the enhanced WebUI**:
   - Main Interface: http://localhost:8000
   - Sci-Fi UI (if enabled): http://localhost:3001
   - API Documentation: http://localhost:8000/docs

### Option 2: Manual Setup

Follow the detailed instructions below for manual installation.

## 📋 Prerequisites

### System Requirements
- Windows 11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Administrator privileges
- Internet connection

### Required Software
- Python 3.11+ 
- Node.js 18+
- Git
- Docker Desktop (for Docker installation)

## 🛠️ Installation Options

### 1. Local Installation

#### Step 1: Install Dependencies
```powershell
# Install Chocolatey (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install required software
choco install python311 nodejs git -y
```

#### Step 2: Clone and Setup
```bash
git clone https://github.com/Subikshaa1910/openhands.git
cd openhands
git checkout arxiv-research-improvements

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-enhanced.txt
```

#### Step 3: Configuration
```bash
# Copy example configuration
copy .env.example .env

# Edit configuration file
notepad .env
```

#### Step 4: Start OpenHands
```bash
# Start the enhanced server
python -m src.api.server
```

### 2. Docker Installation

#### Step 1: Install Docker Desktop
```powershell
choco install docker-desktop -y
```

#### Step 2: Start with Docker Compose
```bash
# Clone repository
git clone https://github.com/Subikshaa1910/openhands.git
cd openhands

# Start with basic features
docker-compose -f docker-compose.enhanced.yml up -d

# Start with Sci-Fi UI
docker-compose -f docker-compose.enhanced.yml --profile scifi-ui up -d

# Start with monitoring
docker-compose -f docker-compose.enhanced.yml --profile monitoring up -d
```

### 3. Lightning Labs Cloud Installation

#### Step 1: Install Lightning SDK
```bash
pip install lightning lightning-sdk
```

#### Step 2: Configure Lightning Labs
```bash
# Login to Lightning Labs
lightning login

# Deploy to cloud
lightning run app lightning_app.py --cloud
```

## ⚙️ Configuration Options

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Installation Settings
INSTALL_TYPE=local
STABILITY_LEVEL=stable
PERFORMANCE_LEVEL=70

# Feature Flags
ENABLE_IDLE_IMPROVEMENT=false
ENABLE_VM_MODE=false
ENABLE_SCIFI_UI=false
ENABLE_MULTI_API=true

# Security Configuration
ADMIN_TOKEN=your-secure-admin-token
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
MEMORY_LIMIT_GB=4

# Idle Improvement Settings
IDLE_THRESHOLD_MINUTES=10
IDLE_IMPROVEMENT_INTERVAL=3600
AC_POWER_REQUIRED=true

# Lightning Labs Settings
LIGHTNING_LABS_ENABLED=false
LIGHTNING_LABS_INSTANCE_TYPE=cpu-small

# Sci-Fi UI Settings
EDEX_UI_ENABLED=false
EDEX_UI_PORT=3001
WORLD_MAP_API_ENABLED=true
```

### Stability Levels

1. **Stable**: Production-ready features only
2. **Testing**: Beta features with extensive testing
3. **Experimental**: Cutting-edge features (may be unstable)

### Performance Levels

- **10-30%**: Low resource usage, basic features
- **40-70%**: Balanced performance and features
- **80-100%**: Maximum performance, all features enabled

## 🎛️ Enhanced WebUI Features

### 1. OpenHands Improvement Panel
- **Auto-Optimize**: Automatic system optimization
- **Manual Control**: User-directed improvements
- **Scheduled**: Time-based improvement cycles

### 2. Model Switching Panel
- **Primary Provider**: Main LLM provider selection
- **Fallback Strategy**: Backup provider configuration
- **Real-time Switching**: Dynamic model selection

### 3. Stability Level Control
- **Production Mode**: Stable, tested features
- **Beta Mode**: New features with safety nets
- **Experimental Mode**: Cutting-edge capabilities

### 4. Performance Control
- **CPU Usage**: Adjustable CPU utilization (10-100%)
- **Memory Limit**: RAM allocation control (1-16GB)
- **Concurrent Requests**: Parallel processing limit

### 5. Idle Improvement Settings
- **System Idle**: Improve when computer is idle
- **VM Idle**: Improve when virtual machine is idle
- **AC Power Required**: Only improve when plugged in
- **Lightning Labs VM**: Use cloud computing for improvements

### 6. Multi-API Management
- **Account Rotation**: Automatic switching between accounts
- **Usage Tracking**: Monitor API usage across accounts
- **Rate Limit Management**: Intelligent request distribution

## 🔬 Sci-Fi UI (eDEX-UI Integration)

### Features
- **Futuristic Terminal Interface**: Sci-fi themed command interface
- **Real-time System Monitoring**: Live CPU, GPU, memory metrics
- **World Map Visualization**: Global API request tracking
- **Network Activity Display**: Real-time network monitoring
- **Temperature Monitoring**: System thermal management

### Setup
1. Enable in configuration: `ENABLE_SCIFI_UI=true`
2. Install eDEX-UI dependencies:
   ```bash
   git clone https://github.com/GitSquared/edex-ui.git edex-ui
   cd edex-ui
   npm install
   ```
3. Start with OpenHands:
   ```bash
   npm start
   ```

### Access
- Sci-Fi Interface: http://localhost:3001
- Integrated with OpenHands API for real-time data

## ⚡ Lightning Labs Integration

### Benefits
- **Cloud Computing**: Offload heavy processing to cloud
- **Scalability**: Auto-scaling based on demand
- **Cost Efficiency**: Pay-per-use cloud resources
- **Performance**: High-performance GPU instances

### Setup
1. Install Lightning SDK:
   ```bash
   pip install lightning lightning-sdk
   ```

2. Configure Lightning Labs:
   ```bash
   lightning login
   ```

3. Enable in configuration:
   ```env
   ENABLE_VM_MODE=true
   LIGHTNING_LABS_ENABLED=true
   LIGHTNING_LABS_INSTANCE_TYPE=gpu.a10g.1x
   ```

### Instance Types
- `cpu-small`: Basic CPU instance
- `cpu-medium`: Enhanced CPU instance
- `gpu.t4.1x`: NVIDIA T4 GPU
- `gpu.a10g.1x`: NVIDIA A10G GPU (recommended)

## 🔄 Idle Improvement System

### How It Works
1. **Idle Detection**: Monitors system activity
2. **Condition Checking**: Verifies improvement conditions
3. **Improvement Execution**: Runs optimization tasks
4. **Result Validation**: Tests and validates improvements

### Conditions
- System CPU usage < 20%
- No user activity for specified time
- AC power connected (optional)
- VM resources available (if enabled)

### Improvement Tasks
- Model performance optimization
- Provider routing enhancement
- Cache optimization
- Database query optimization
- Security configuration updates

## 🔐 Security Features

### Authentication
- Admin token-based access control
- API key rotation
- Encrypted credential storage

### Network Security
- CORS configuration
- Rate limiting
- Request validation

### Data Protection
- Encrypted environment variables
- Secure API key management
- Audit logging

## 📊 Monitoring and Analytics

### System Metrics
- CPU, memory, disk usage
- Network activity
- Temperature monitoring
- Process tracking

### OpenHands Metrics
- API request rates
- Model performance
- Provider availability
- Error rates

### Visualization
- Real-time dashboards
- Historical trends
- Performance analytics
- Usage statistics

## 🚨 Troubleshooting

### Common Issues

#### 1. Setup Script Fails
```bash
# Check administrator privileges
net session

# Verify PowerShell execution policy
Get-ExecutionPolicy
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Docker Issues
```bash
# Restart Docker Desktop
# Check Docker daemon status
docker version

# Reset Docker if needed
docker system prune -a
```

#### 3. Port Conflicts
```bash
# Check port usage
netstat -an | findstr :8000

# Kill process using port
taskkill /PID <process_id> /F
```

#### 4. Python Dependencies
```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Log Files
- Application logs: `logs/openhands_enhanced.log`
- Error logs: `logs/error.log`
- System logs: Windows Event Viewer

### Support
- GitHub Issues: https://github.com/Subikshaa1910/openhands/issues
- Documentation: https://github.com/Subikshaa1910/openhands/wiki
- Community: Discord/Slack (links in repository)

## 🔄 Updates and Maintenance

### Automatic Updates
- Enable auto-updater in configuration
- Monitors GitHub for new releases
- Downloads and applies updates safely

### Manual Updates
```bash
git pull origin arxiv-research-improvements
pip install -r requirements.txt --upgrade
```

### Backup and Restore
```bash
# Backup configuration
copy .env config_backup.env

# Backup data
xcopy data data_backup /E /I

# Restore from backup
copy config_backup.env .env
```

## 🎯 Performance Optimization

### System Optimization
- Disable unnecessary Windows services
- Configure power settings for performance
- Optimize virtual memory settings

### OpenHands Optimization
- Adjust performance levels based on usage
- Configure provider priorities
- Optimize caching strategies

### Resource Management
- Monitor resource usage
- Set appropriate limits
- Use cloud resources for heavy tasks

## 📈 Advanced Features

### Custom Providers
- Add new LLM providers
- Configure custom endpoints
- Implement provider-specific logic

### Plugin System
- Develop custom plugins
- Extend functionality
- Community plugin marketplace

### API Extensions
- Custom API endpoints
- Webhook integrations
- Third-party service connections

## 🎉 Getting Started Checklist

- [ ] Install prerequisites (Python, Node.js, Git)
- [ ] Clone OpenHands repository
- [ ] Run setup script or manual installation
- [ ] Configure environment variables
- [ ] Start OpenHands Enhanced
- [ ] Access WebUI and verify functionality
- [ ] Configure model providers and API keys
- [ ] Test model switching functionality
- [ ] Enable optional features (Sci-Fi UI, Lightning Labs)
- [ ] Set up monitoring and analytics
- [ ] Configure idle improvement system
- [ ] Create backup of configuration

## 🚀 Next Steps

1. **Explore the WebUI**: Familiarize yourself with all panels and features
2. **Add API Keys**: Configure your preferred LLM providers
3. **Test Model Switching**: Verify automatic fallback and routing
4. **Enable Advanced Features**: Try Sci-Fi UI and Lightning Labs integration
5. **Monitor Performance**: Use built-in analytics and monitoring
6. **Join the Community**: Contribute to the project and get support

---

**Happy coding with OpenHands Enhanced!** 🤖✨