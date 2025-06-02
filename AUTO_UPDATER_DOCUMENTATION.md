# Auto-Updater System Documentation

## Overview

The Auto-Updater system provides comprehensive automatic discovery and updating capabilities for the LLM API Aggregator. It continuously monitors multiple sources to discover new free LLM providers, models, rate limits, and pricing information, automatically integrating these updates into the main aggregator system.

## Key Features

### 🔍 Multi-Source Discovery
- **GitHub Integration**: Monitors community-maintained lists like `cheahjs/free-llm-api-resources`
- **API Discovery**: Direct API calls to discover new models and capabilities
- **Web Scraping**: Automated scraping of provider websites for model information
- **Browser Automation**: Advanced monitoring using Playwright for dynamic content

### 🔄 Intelligent Integration
- **Automatic Updates**: Seamlessly integrates discovered changes into the aggregator
- **Meta-Controller Adaptation**: Updates model capability profiles and routing decisions
- **Ensemble System Updates**: Refreshes model rankings and fusion strategies
- **Configuration Management**: Maintains provider configurations and rate limits

### 📊 Real-Time Monitoring
- **Live Status Dashboard**: Real-time monitoring of update sources and system status
- **Update History**: Comprehensive tracking of all discovered changes
- **Performance Metrics**: Monitoring of update frequency and success rates
- **Alert System**: Notifications for significant changes or errors

## Architecture

### Core Components

#### 1. AutoUpdater Class (`src/core/auto_updater.py`)
The main orchestrator that manages all update sources and coordinates discovery activities.

```python
class AutoUpdater:
    def __init__(self, account_manager: AccountManager = None):
        self.sources = self._load_update_sources()
        self.cache = {}
        self.account_manager = account_manager
        self.last_full_update = None
```

**Key Methods:**
- `start_auto_update()`: Begins continuous monitoring
- `force_update_all()`: Triggers immediate update from all sources
- `get_update_status()`: Returns current system status
- `close()`: Cleanup and shutdown

#### 2. UpdateSource Class
Represents individual update sources with their configuration and state.

```python
@dataclass
class UpdateSource:
    name: str
    type: str  # "github", "api", "web_scrape", "browser"
    url: str
    enabled: bool = True
    update_interval: int = 24  # hours
    config: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None
```

#### 3. BrowserMonitor Class (`src/core/browser_monitor.py`)
Handles advanced browser automation for monitoring provider dashboards.

```python
class BrowserMonitor:
    def __init__(self):
        self.browser = None
        self.context = None
        self.provider_configs = self._load_provider_configs()
```

### Integration with Main Aggregator

The auto-updater is fully integrated into the main `LLMAggregator` class:

```python
class LLMAggregator:
    def __init__(self, ..., enable_auto_updater: bool = True):
        if self.enable_auto_updater:
            self.auto_updater = AutoUpdater(account_manager=self.account_manager)
            asyncio.create_task(self._start_auto_updater())
```

## Configuration

### Auto-Update Configuration (`config/auto_update.yaml`)

```yaml
update_sources:
  # GitHub community projects
  - name: "cheahjs/free-llm-api-resources"
    type: "github"
    url: "https://api.github.com/repos/cheahjs/free-llm-api-resources"
    enabled: true
    update_interval: 6  # hours
    config:
      content_path: "src/data.py"
      parser: "python_dict"
      
  # Direct API discovery
  - name: "openrouter_api"
    type: "api"
    url: "https://openrouter.ai/api/v1/models"
    enabled: true
    update_interval: 2  # hours
    config:
      requires_key: false
      
  # Web scraping
  - name: "openrouter_website"
    type: "web_scrape"
    url: "https://openrouter.ai/models"
    enabled: true
    update_interval: 24  # hours
    config:
      selectors:
        model_cards: ".model-card"
        model_name: ".model-name"
        pricing: ".pricing-info"
        
  # Browser automation
  - name: "openrouter_browser"
    type: "browser"
    url: "https://openrouter.ai/models"
    enabled: false  # Requires Playwright setup
    update_interval: 12  # hours
```

### Browser Monitor Configuration (`config/browser_config.yaml`)

```yaml
providers:
  openrouter:
    models_page: "https://openrouter.ai/models"
    dashboard_url: "https://openrouter.ai/account"
    login_url: "https://openrouter.ai/auth"
    selectors:
      model_cards: ".model-card, [data-testid='model-card']"
      model_name: ".model-name, h3, .font-semibold"
      free_badge: ".free-badge, .badge-free"
      pricing: ".pricing, .cost"
```

## Usage

### Basic Usage

```python
from src.core.aggregator import LLMAggregator

# Create aggregator with auto-updater enabled
aggregator = LLMAggregator(
    providers=providers,
    account_manager=account_manager,
    router=router,
    rate_limiter=rate_limiter,
    enable_auto_updater=True,
    auto_update_interval=60  # minutes
)

# Force immediate update
update_result = await aggregator.force_update_providers()
print(f"Found {update_result['updates_found']} updates")

# Get auto-updater status
status = await aggregator.get_auto_update_status()
print(f"Auto-updater enabled: {status['enabled']}")

# Configure auto-updater
config_result = await aggregator.configure_auto_updater({
    "update_interval": 30,
    "sources": [
        {"name": "openrouter_api", "enabled": True}
    ]
})
```

### Standalone Usage

```python
from src.core.auto_updater import AutoUpdater

# Create standalone auto-updater
auto_updater = AutoUpdater()

# Start continuous monitoring
await auto_updater.start_auto_update(interval_minutes=60)

# Force update from all sources
updates = await auto_updater.force_update_all()

# Get status
status = await auto_updater.get_update_status()
```

### Browser Monitoring

```python
from src.core.browser_monitor import BrowserMonitor

# Create browser monitor
browser_monitor = BrowserMonitor()
await browser_monitor.start()

# Monitor specific provider
result = await browser_monitor.monitor_provider("openrouter")
print(f"Found {len(result.get('models', []))} models")

await browser_monitor.stop()
```

## API Endpoints

When integrated with the FastAPI server, the auto-updater provides these endpoints:

### GET `/auto-update/status`
Returns current auto-updater status and configuration.

```json
{
  "enabled": true,
  "sources_count": 10,
  "enabled_sources": 8,
  "last_update": "2024-01-15T10:30:00Z",
  "cache_entries": 25
}
```

### POST `/auto-update/force`
Triggers immediate update from all sources.

```json
{
  "status": "success",
  "updates_found": 3,
  "updates": [
    {
      "provider": "openrouter",
      "models_added": 2,
      "models_removed": 0,
      "models_updated": 1,
      "rate_limits_updated": true
    }
  ]
}
```

### PUT `/auto-update/configure`
Updates auto-updater configuration.

```json
{
  "update_interval": 30,
  "sources": [
    {
      "name": "openrouter_api",
      "enabled": true,
      "update_interval": 2
    }
  ]
}
```

### GET `/auto-update/history/{provider?}`
Returns update history for all providers or a specific provider.

```json
{
  "providers": {
    "openrouter": {
      "cached_models": 15,
      "last_update": "2024-01-15T10:30:00Z"
    }
  }
}
```

## Update Sources

### 1. GitHub Integration

Monitors community-maintained repositories for free LLM API resources:

- **cheahjs/free-llm-api-resources**: Comprehensive list of free LLM APIs
- **zukixa/cool-ai-stuff**: Community-curated AI resources
- **wdhdev/free-for-life**: Free-tier services including AI APIs

**Features:**
- Automatic parsing of Python dictionaries and Markdown files
- GitHub API integration for efficient monitoring
- Commit-based change detection
- Rate limit compliance

### 2. API Discovery

Direct API calls to provider endpoints to discover models:

- **OpenRouter**: `/api/v1/models` endpoint
- **Groq**: `/openai/v1/models` endpoint  
- **Cerebras**: `/v1/models` endpoint
- **Hugging Face**: Inference API model discovery

**Features:**
- Real-time model discovery
- Automatic capability detection
- Rate limit and pricing extraction
- Authentication handling

### 3. Web Scraping

Automated scraping of provider websites:

- **OpenRouter Models Page**: Model listings and pricing
- **Groq Models Page**: Available models and capabilities
- **Provider Documentation**: Rate limits and usage guidelines

**Features:**
- CSS selector-based extraction
- Robust error handling
- Content change detection
- Respect for robots.txt

### 4. Browser Automation

Advanced monitoring using Playwright:

- **Dynamic Content**: JavaScript-rendered model lists
- **Dashboard Monitoring**: Account-specific information
- **Interactive Elements**: Forms and dynamic updates

**Features:**
- Full browser automation
- Screenshot capture for debugging
- Cookie and session management
- Headless operation

## Data Flow

### 1. Discovery Phase
```
GitHub API → Parse Content → Extract Models
API Calls → JSON Response → Model Mapping
Web Scraping → HTML Parsing → Model Extraction
Browser Automation → DOM Interaction → Data Collection
```

### 2. Processing Phase
```
Raw Data → Validation → Deduplication → ModelInfo Creation
Rate Limits → Parsing → RateLimit Objects
Provider Info → Mapping → ProviderConfig Updates
```

### 3. Integration Phase
```
New Models → Provider.add_model()
Updated Limits → Provider.update_rate_limit()
Config Changes → Provider.update_config()
Meta-Controller → Profile Updates
Ensemble System → Ranking Updates
```

### 4. Notification Phase
```
Changes Detected → Event Generation → Notification System
Status Updates → Dashboard Refresh → User Alerts
Error Conditions → Logging → Admin Notifications
```

## Monitoring and Alerting

### Status Monitoring

The system provides comprehensive monitoring:

```python
# Real-time status
status = await auto_updater.get_update_status()
{
    "enabled": True,
    "sources_active": 8,
    "last_update": "2024-01-15T10:30:00Z",
    "cache_size": 1024,
    "errors_last_24h": 2
}
```

### Performance Metrics

- **Update Frequency**: How often each source is checked
- **Success Rate**: Percentage of successful updates
- **Response Time**: Time taken for each update cycle
- **Cache Hit Rate**: Efficiency of caching system

### Error Handling

- **Graceful Degradation**: System continues operating with partial failures
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Logging**: Comprehensive error tracking and reporting
- **Fallback Sources**: Alternative sources when primary fails

## Security Considerations

### API Key Management
- Secure storage of API keys using encryption
- Rotation of keys based on provider requirements
- Scope limitation for discovery-only access

### Rate Limiting
- Respect for provider rate limits
- Intelligent backoff strategies
- Distributed rate limiting across sources

### Data Validation
- Input sanitization for scraped content
- Schema validation for API responses
- Malicious content detection

### Privacy Protection
- No storage of user-specific data
- Anonymized usage statistics
- GDPR compliance for EU users

## Troubleshooting

### Common Issues

#### 1. GitHub API Rate Limiting
```
Error: GitHub API rate limit exceeded
Solution: Configure GitHub token or reduce update frequency
```

#### 2. Web Scraping Failures
```
Error: CSS selectors not found
Solution: Update selectors in configuration or disable source
```

#### 3. Browser Automation Issues
```
Error: Playwright browser not found
Solution: Install Playwright: pip install playwright && playwright install
```

#### 4. Provider API Changes
```
Error: API response format changed
Solution: Update parser configuration or disable source temporarily
```

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger("src.core.auto_updater").setLevel(logging.DEBUG)
```

Check update source status:
```python
status = await auto_updater.get_update_status()
for source in status["sources"]:
    if not source["enabled"]:
        print(f"Disabled source: {source['name']}")
```

### Performance Optimization

#### 1. Caching Strategy
- Implement intelligent caching with TTL
- Use Redis for distributed caching
- Cache invalidation on significant changes

#### 2. Parallel Processing
- Concurrent updates from multiple sources
- Async/await for non-blocking operations
- Worker pools for CPU-intensive tasks

#### 3. Resource Management
- Connection pooling for HTTP requests
- Memory management for large datasets
- Cleanup of temporary resources

## Future Enhancements

### Planned Features

#### 1. Machine Learning Integration
- **Predictive Updates**: ML models to predict when providers will add new models
- **Anomaly Detection**: Automatic detection of unusual changes or errors
- **Quality Scoring**: ML-based quality assessment of discovered models

#### 2. Advanced Monitoring
- **Real-time Dashboards**: Live monitoring with charts and graphs
- **Alert System**: Configurable alerts for various conditions
- **Performance Analytics**: Detailed analytics on update patterns

#### 3. Community Integration
- **Crowdsourced Updates**: Allow community contributions
- **Validation System**: Community validation of discovered changes
- **Reputation System**: Track reliability of different sources

#### 4. Enhanced Browser Automation
- **Anti-Detection**: Advanced techniques to avoid detection
- **CAPTCHA Solving**: Integration with CAPTCHA solving services
- **Session Management**: Persistent sessions across updates

### Roadmap

- **Q1 2024**: ML integration and predictive updates
- **Q2 2024**: Advanced monitoring and alerting
- **Q3 2024**: Community integration features
- **Q4 2024**: Enhanced browser automation

## Contributing

### Adding New Sources

1. **Define Source Configuration**:
```yaml
- name: "new_provider_api"
  type: "api"
  url: "https://api.newprovider.com/models"
  enabled: true
  update_interval: 4
  config:
    requires_key: false
    parser: "openai_format"
```

2. **Implement Parser** (if needed):
```python
def parse_new_provider_response(response_data: Dict) -> List[ModelInfo]:
    models = []
    for model_data in response_data.get("models", []):
        model = ModelInfo(
            name=model_data["id"],
            display_name=model_data["name"],
            provider="new_provider",
            capabilities=["text_generation"],
            is_free=model_data.get("free", False)
        )
        models.append(model)
    return models
```

3. **Test Integration**:
```python
# Test the new source
auto_updater = AutoUpdater()
updates = await auto_updater._update_from_api(new_source)
assert len(updates) > 0
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include unit tests for new features

### Testing

Run the test suite:
```bash
python -m pytest tests/test_auto_updater.py -v
```

Run integration tests:
```bash
python test_auto_updater_integration.py
```

## License

This auto-updater system is part of the LLM API Aggregator project and is licensed under the MIT License. See the main project LICENSE file for details.

## Support

For issues, questions, or contributions:

1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check this documentation and code comments
3. **Community**: Join discussions in project forums
4. **Email**: Contact maintainers for urgent issues

---

*Last updated: January 2024*
*Version: 1.0.0*