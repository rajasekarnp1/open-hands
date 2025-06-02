# 🤖 OpenHands Enhanced - Advanced AI Model Switching Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![Lightning Labs](https://img.shields.io/badge/lightning-labs-purple.svg)](https://lightning.ai/)

> **The most advanced LLM API aggregator with intelligent model switching, sci-fi UI, and autonomous improvement capabilities.**

## 🌟 What Makes OpenHands Enhanced Special?

OpenHands Enhanced transforms the way you interact with multiple LLM providers by offering:

- **🔄 Intelligent Model Switching**: Seamlessly switch between 25+ free and trial LLM providers
- **🚀 Self-Improving System**: Automatically optimizes performance during idle periods
- **🎮 Sci-Fi Themed UI**: Futuristic eDEX-UI integration with real-time system monitoring
- **☁️ Cloud Integration**: Lightning Labs support for scalable cloud execution
- **🔐 Multi-Account Management**: Secure rotation across multiple API accounts
- **📊 Advanced Analytics**: Real-time performance monitoring and optimization
- **⚡ Production-Ready**: Enterprise-grade reliability and security

## 🎯 Key Features

### 🔄 Advanced Model Switching
- **25+ Provider Support**: OpenRouter, Groq, Cerebras, HuggingFace, Mistral, and more
- **Intelligent Routing**: Performance-based provider selection
- **Automatic Fallback**: Seamless switching when providers fail or hit limits
- **Cost Optimization**: Prioritizes free tiers and trial credits
- **Real-time Monitoring**: Live provider status and performance metrics

### 🚀 Autonomous Improvement System
- **Idle Detection**: Automatically improves when system is idle
- **Cloud Processing**: Optional Lightning Labs integration for heavy workloads
- **Performance Optimization**: Continuous enhancement of routing algorithms
- **Self-Learning**: Adapts to usage patterns and optimizes accordingly
- **Safe Rollback**: Automatic rollback on errors or performance degradation

### 🎮 Sci-Fi Themed Interface
- **eDEX-UI Integration**: Futuristic terminal-style interface
- **Real-time Metrics**: Live CPU, GPU, memory, and temperature monitoring
- **World Map Visualization**: Global API request tracking
- **System Monitoring**: Comprehensive system health dashboard
- **Customizable Themes**: Multiple sci-fi themed interfaces

### ☁️ Cloud & Scaling
- **Lightning Labs Support**: Cloud-based execution and scaling
- **Docker Integration**: Containerized deployment with docker-compose
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Multi-instance**: Support for distributed deployments
- **Load Balancing**: Intelligent request distribution

### 🔐 Security & Management
- **Encrypted Storage**: Secure API key and credential management
- **Account Rotation**: Automatic switching between multiple accounts
- **Rate Limiting**: Intelligent request throttling and management
- **Audit Logging**: Comprehensive security and usage logging
- **Access Control**: Token-based authentication and authorization

## 🚀 Quick Start

### Option 1: Automated Setup (Windows 11)

1. **Download and run the setup script**:
   ```bash
   git clone https://github.com/Subikshaa1910/openhands.git
   cd openhands
   ```

2. **Run as Administrator**:
   - Right-click `setup_openhands.bat`
   - Select "Run as administrator"
   - Follow interactive prompts

3. **Access the interfaces**:
   - **Main WebUI**: http://localhost:8000
   - **Sci-Fi UI**: http://localhost:3001 (if enabled)
   - **API Docs**: http://localhost:8000/docs

### Option 2: Docker Deployment

```bash
# Basic deployment
docker-compose -f docker-compose.enhanced.yml up -d

# With Sci-Fi UI
docker-compose -f docker-compose.enhanced.yml --profile scifi-ui up -d

# With monitoring
docker-compose -f docker-compose.enhanced.yml --profile monitoring up -d
```

### Option 3: Manual Installation

```bash
# Clone repository
git clone https://github.com/Subikshaa1910/openhands.git
cd openhands
git checkout arxiv-research-improvements

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-enhanced.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start the server
python -m src.api.server
```

## 🎛️ Enhanced WebUI Features

### 1. 🚀 OpenHands Improvement Panel
- **Auto-Optimize**: Continuous system optimization
- **Manual Control**: User-directed improvements
- **Scheduled**: Time-based improvement cycles
- **Target Areas**: Performance, accuracy, efficiency, features

### 2. 🔄 Model Switching Panel
- **Primary Provider**: Main LLM provider selection
- **Fallback Strategy**: Performance/cost/availability-based routing
- **Real-time Status**: Live provider health monitoring
- **Usage Analytics**: Request distribution and performance metrics

### 3. ⚖️ Stability Level Control
- **Stable**: Production-ready features only
- **Testing**: Beta features with safety nets
- **Experimental**: Cutting-edge capabilities
- **Auto-rollback**: Automatic error recovery

### 4. ⚡ Performance Control
- **CPU Usage**: Adjustable utilization (10-100%)
- **Memory Limit**: RAM allocation (1-16GB)
- **Concurrent Requests**: Parallel processing (1-50)
- **Real-time Metrics**: Live performance monitoring

### 5. 🔋 Idle Improvement Settings
- **System Idle**: Improve when computer is idle
- **VM Idle**: Cloud-based improvement processing
- **AC Power**: Only improve when plugged in
- **Lightning Labs**: Use cloud computing for heavy tasks

### 6. 🔑 Multi-API Management
- **Account Rotation**: Automatic switching (1-60 min intervals)
- **Usage Tracking**: Monitor API consumption
- **Rate Limit Management**: Intelligent distribution
- **Credential Security**: Encrypted storage and rotation

## 🎮 Sci-Fi UI (eDEX-UI Integration)

Experience the future of AI interaction with our sci-fi themed interface:

### Features
- **🌌 Futuristic Design**: Terminal-style interface inspired by sci-fi movies
- **📊 Real-time Monitoring**: Live system metrics with visual effects
- **🌍 World Map**: Global API request visualization
- **🔥 Temperature Display**: System thermal monitoring
- **⚡ Network Activity**: Real-time network traffic visualization
- **🎯 Process Tracking**: Live process monitoring and management

### Screenshots
```
┌─────────────────────────────────────────────────────────────┐
│  ███████╗██████╗ ███████╗██╗  ██╗    ██╗   ██╗██╗          │
│  ██╔════╝██╔══██╗██╔════╝╚██╗██╔╝    ██║   ██║██║          │
│  █████╗  ██║  ██║█████╗   ╚███╔╝     ██║   ██║██║          │
│  ██╔══╝  ██║  ██║██╔══╝   ██╔██╗     ██║   ██║██║          │
│  ███████╗██████╔╝███████╗██╔╝ ██╗    ╚██████╔╝██║          │
│  ╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝     ╚═════╝ ╚═╝          │
│                                                             │
│  OpenHands Enhanced - Sci-Fi Interface                     │
│  ═══════════════════════════════════════════════════════   │
│                                                             │
│  CPU: ████████████████████░░░░ 75%    Temp: 42°C          │
│  RAM: ██████████████░░░░░░░░░░ 65%    GPU:  ████████ 80%  │
│  NET: ↑ 1.2MB/s ↓ 3.4MB/s            Disk: ████████ 45%  │
│                                                             │
│  [WORLD MAP] Active Connections: 42                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │    ●US-East     ●Europe      ●Asia-Pacific         │   │
│  │      65%         78%           52%                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  > openhands status                                         │
│  ✓ Model switching: ACTIVE                                 │
│  ✓ Providers online: 23/25                                 │
│  ✓ Idle improvement: STANDBY                               │
│  ✓ Lightning Labs: CONNECTED                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ☁️ Lightning Labs Integration

Leverage cloud computing for enhanced performance:

### Benefits
- **🚀 Scalability**: Auto-scaling based on demand
- **💰 Cost Efficiency**: Pay-per-use cloud resources
- **⚡ Performance**: High-performance GPU instances
- **🔄 Flexibility**: Switch between local and cloud execution

### Instance Types
- **cpu-small**: Basic CPU instance for light workloads
- **cpu-medium**: Enhanced CPU for moderate processing
- **gpu.t4.1x**: NVIDIA T4 GPU for ML workloads
- **gpu.a10g.1x**: NVIDIA A10G GPU for heavy processing

### Setup
```bash
# Install Lightning SDK
pip install lightning lightning-sdk

# Login to Lightning Labs
lightning login

# Deploy to cloud
lightning run app lightning_app.py --cloud
```

## 🔄 Idle Improvement System

Automatically enhance your system during idle periods:

### How It Works
1. **🔍 Idle Detection**: Monitors system activity and resource usage
2. **✅ Condition Checking**: Verifies improvement conditions are met
3. **🚀 Improvement Execution**: Runs optimization tasks safely
4. **🧪 Result Validation**: Tests and validates all improvements
5. **📊 Performance Tracking**: Monitors improvement effectiveness

### Improvement Tasks
- **🎯 Model Performance**: Optimize routing algorithms and response times
- **🔧 Provider Configuration**: Update and optimize provider settings
- **💾 Cache Optimization**: Enhance caching strategies and hit rates
- **🗄️ Database Tuning**: Optimize queries and database performance
- **🔐 Security Updates**: Apply security patches and configuration updates

### Conditions
- System CPU usage < 20%
- No user activity for specified time (default: 10 minutes)
- AC power connected (configurable)
- Available system resources
- VM resources available (if cloud mode enabled)

## 📊 Monitoring & Analytics

### Real-time Dashboards
- **System Metrics**: CPU, memory, disk, network, temperature
- **OpenHands Metrics**: Request rates, response times, error rates
- **Provider Status**: Availability, performance, usage statistics
- **Improvement History**: Optimization results and performance gains

### Performance Analytics
- **Response Time Trends**: Historical performance analysis
- **Provider Comparison**: Comparative performance metrics
- **Usage Patterns**: Request distribution and peak usage analysis
- **Cost Optimization**: Usage cost analysis and optimization suggestions

## 🔐 Security Features

### Authentication & Authorization
- **🔑 Token-based Access**: Secure API access with admin tokens
- **🔄 Credential Rotation**: Automatic API key rotation
- **🔒 Encrypted Storage**: Secure credential and configuration storage
- **📝 Audit Logging**: Comprehensive security and access logging

### Network Security
- **🌐 CORS Configuration**: Secure cross-origin resource sharing
- **⚡ Rate Limiting**: Intelligent request throttling
- **🛡️ Request Validation**: Input validation and sanitization
- **🔍 Security Scanning**: Automated vulnerability detection

## 🛠️ Configuration

### Environment Variables
```env
# Installation Settings
INSTALL_TYPE=local|docker|lightning
STABILITY_LEVEL=stable|testing|experimental
PERFORMANCE_LEVEL=10-100

# Feature Flags
ENABLE_IDLE_IMPROVEMENT=true|false
ENABLE_VM_MODE=true|false
ENABLE_SCIFI_UI=true|false
ENABLE_MULTI_API=true|false

# Security
ADMIN_TOKEN=your-secure-token
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
MEMORY_LIMIT_GB=4

# Lightning Labs
LIGHTNING_LABS_ENABLED=true|false
LIGHTNING_LABS_INSTANCE_TYPE=cpu-small|gpu.a10g.1x

# Sci-Fi UI
EDEX_UI_ENABLED=true|false
EDEX_UI_PORT=3001
WORLD_MAP_API_ENABLED=true|false
```

## 📚 API Documentation

### Core Endpoints
```bash
# Chat completion with automatic model selection
POST /v1/chat/completions
{
  "model": "auto",
  "messages": [{"role": "user", "content": "Hello!"}]
}

# Provider-specific request
POST /v1/chat/completions
{
  "model": "openrouter/deepseek-coder-33b-instruct",
  "messages": [{"role": "user", "content": "Write Python code"}]
}

# System status
GET /v1/status

# Available models
GET /v1/models

# Provider status
GET /v1/providers/status
```

### Enhanced Endpoints
```bash
# Start improvement process
POST /api/v1/improvement/start
{
  "mode": "auto",
  "areas": ["performance", "accuracy"]
}

# Update model configuration
PUT /api/v1/models/config
{
  "primaryProvider": "openrouter",
  "fallbackStrategy": "performance"
}

# System metrics
GET /api/v1/metrics

# Export configuration
GET /api/v1/config/export
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow our coding standards
4. **Add tests**: Ensure your changes are tested
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**: Describe your changes

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/openhands.git
cd openhands

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
black src/
isort src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenRouter**: For providing excellent LLM API aggregation
- **Lightning Labs**: For cloud computing infrastructure
- **eDEX-UI**: For the amazing sci-fi terminal interface
- **FastAPI**: For the robust API framework
- **Community Contributors**: For making this project better

## 📞 Support

- **📖 Documentation**: [Enhanced Setup Guide](ENHANCED_SETUP_GUIDE.md)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/Subikshaa1910/openhands/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Subikshaa1910/openhands/discussions)
- **📧 Email**: support@openhands.dev

## 🗺️ Roadmap

### 🎯 Current Focus (v1.0)
- [x] Multi-provider LLM aggregation
- [x] Intelligent model switching
- [x] Enhanced WebUI with real-time controls
- [x] Idle improvement system
- [x] Lightning Labs integration
- [x] Sci-Fi UI (eDEX-UI) integration
- [x] Multi-account management
- [x] Production-ready deployment

### 🚀 Next Release (v1.1)
- [ ] Advanced ML-based routing
- [ ] Custom provider plugins
- [ ] Mobile app interface
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Enterprise SSO integration

### 🌟 Future Vision (v2.0)
- [ ] AI-powered conversation optimization
- [ ] Multi-modal support (vision, audio)
- [ ] Distributed deployment
- [ ] Marketplace for custom providers
- [ ] Advanced workflow automation
- [ ] Integration with popular dev tools

---

<div align="center">

**🤖 Built with ❤️ for the AI community**

[⭐ Star us on GitHub](https://github.com/Subikshaa1910/openhands) • [🐛 Report Bug](https://github.com/Subikshaa1910/openhands/issues) • [💡 Request Feature](https://github.com/Subikshaa1910/openhands/issues)

</div>