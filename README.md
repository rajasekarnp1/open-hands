# OpenHands LLM Aggregator

A production-grade, open-source solution for standardizing and managing requests to 100+ LLM providers.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## Problem Statement

The user needs a system that can:
1. **Switch between different free LLM API providers** from various sources
2. **Manage different accounts** for these providers
3. **Handle model switching dynamically** with intelligent routing
4. **Optimize costs** by using free tiers and trial credits effectively
5. **Provide fallback mechanisms** when providers fail or hit rate limits

## Key Features

- 🔄 **Multi-Provider Support**: Integrate with 25+ free LLM providers
- 🔐 **Secure Account Management**: Encrypted credential storage and rotation
- 🎯 **Intelligent Routing**: Automatic provider selection based on model, cost, and availability
- 📊 **Rate Limit Management**: Track and respect provider-specific limits
- 🔄 **Automatic Fallbacks**: Seamless switching when providers fail
- 💰 **Cost Optimization**: Prioritize free tiers and trial credits
- 🎯 **Flexible Routing Control**: Specify a `provider` directly in your API requests, or influence model selection with `model_quality` (`fastest`, `best_quality`, `balanced`).
- 🔑 **Bring Your Own Key (BYOK)**: Easily configure your own API keys for premium providers.
- 📈 **Usage Analytics**: Track usage across providers and accounts.
- 🛡️ **Error Handling**: Robust error handling and retry mechanisms.
- 🔄 **Auto-Updater**: Automatic discovery of new providers and models.
- 🧠 **Meta-Controller**: Research-based intelligent model selection.
- 🎯 **Ensemble System**: Multi-model response fusion and quality assessment.

## Supported Providers

### Free Providers
- OpenRouter (50+ free models)
- Google AI Studio (Gemini models)
- NVIDIA NIM (Various open models)
- Mistral (La Plateforme & Codestral)
- HuggingFace Inference
- Cerebras (Fast inference)
- Groq (Ultra-fast inference)
- Together (Free tier)
- Cohere (Command models)
- GitHub Models (Premium models)
- Chutes (Decentralized)
- Cloudflare Workers AI
- Google Cloud Vertex AI

### Trial Credit Providers
- Together ($1 credit)
- Fireworks ($1 credit)
- Unify ($5 credit)
- Baseten ($30 credit)
- Nebius ($1 credit)
- Novita ($0.5-$10 credit)
- AI21 ($10 credit)
- Upstage ($10 credit)
- And many more...

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   API Gateway   │    │  Provider Pool  │
│                 │────│                 │────│                 │
│ - Web UI        │    │ - Rate Limiting │    │ - OpenRouter    │
│ - CLI Tool      │    │ - Load Balancer │    │ - Google AI     │
│ - API Client    │    │ - Auth Handler  │    │ - NVIDIA NIM    │
└─────────────────┘    └─────────────────┘    │ - Groq          │
                                              │ - Cerebras      │
                                              │ - ...           │
                                              └─────────────────┘
```

## Auto-Updater System

The system includes a comprehensive auto-updater that continuously discovers new free LLM providers and models:

### 🔍 Multi-Source Discovery
- **GitHub Integration**: Monitors community projects like `cheahjs/free-llm-api-resources`
- **API Discovery**: Real-time discovery of new models via provider APIs
- **Web Scraping**: Automated monitoring of provider websites
- **Browser Automation**: Advanced monitoring using Playwright

### 🔄 Intelligent Integration
- **Automatic Updates**: Seamlessly integrates discovered changes
- **Meta-Controller Adaptation**: Updates model capability profiles
- **Configuration Management**: Maintains provider configs and rate limits
- **Real-time Monitoring**: Live status dashboard and update history

### 📊 Community Integration
Integrates with existing GitHub projects:
- [cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources)
- [zukixa/cool-ai-stuff](https://github.com/zukixa/cool-ai-stuff)
- [wdhdev/free-for-life](https://github.com/wdhdev/free-for-life)

```bash
# Run auto-updater demo
python auto_updater_demo.py

# Test integration
python test_auto_updater_integration.py
```

## Bring Your Own Key (BYOK)

The LLM API Aggregator supports using your personal or commercial API keys for various providers, allowing you to leverage your existing subscriptions and access premium models.

### Configuration
API keys are managed securely using the `AccountManager` and are stored in an encrypted `credentials.json` file in the root directory.

To configure your keys:
1. Run the interactive setup script:
   ```bash
   python setup.py configure
   ```
2. The script will guide you through adding accounts for supported providers.
3. For example, when prompted for **Anthropic**, you can enter your Anthropic API key. This key will be securely stored and used for any requests routed to Anthropic.

If you skip adding a key for a provider, that provider will only be usable if it offers a free tier that doesn't require authentication or if other public credentials are configured.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure providers
python setup.py configure

# Start the service
python main.py --port 8000

# Test the API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration

The system uses a hierarchical configuration approach:

1. **Provider Configuration**: Define available providers and their capabilities
2. **Account Management**: Securely store API keys and credentials
3. **Routing Rules**: Define how to select providers for different requests
4. **Rate Limits**: Configure provider-specific limits and quotas

## API Endpoints

The primary way to interact with the LLM API Aggregator is through its FastAPI endpoints.

### Standard Chat Completions

Endpoint: `POST /v1/chat/completions`

This endpoint is compatible with the OpenAI chat completions API structure.

**Request Body:**
```json
{
  "model": "auto", // Or a specific model name like "openrouter/deepseek/deepseek-coder-33b-instruct"
  "messages": [{"role": "user", "content": "Hello, world!"}],
  "provider": null, // Optional: "openrouter", "groq", "anthropic", etc.
  "model_quality": null, // Optional: "fastest", "best_quality", "balanced"
  "stream": false, // Optional: true for streaming
  // ... other standard parameters like max_tokens, temperature
}
```

**Example Curl:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}],
    "model_quality": "fastest"
  }'
```

## VS Code Extension: OpenHands AI Assistant

Enhance your coding workflow with the OpenHands AI Assistant, bringing the power of the LLM Aggregator directly into your VS Code editor.

### Purpose
The extension allows you to interact with the LLM Aggregator for tasks like explaining code, generating code snippets, refactoring, and more, all within the context of your currently open files.

### Installation & Setup (Development)
Currently, the extension is under development. To use it:
1.  **Clone the Repository**: Ensure you have the main LLM API Aggregator project cloned.
2.  **Navigate to Extension Directory**: `cd openhands-vscode-extension`
3.  **Install Dependencies**: Run `npm install`.
4.  **Compile Extension**: Run `npm run compile` (or `npm run watch` for automatic recompilation on changes).
5.  **Run from VS Code**:
    *   Open the `openhands-vscode-extension` directory in VS Code.
    *   Go to the "Run and Debug" panel (Ctrl+Shift+D).
    *   Select "Run Extension" from the dropdown and click the play button (F5). This will open a new VS Code window (Extension Development Host) with the extension active.
6.  **Configure API Endpoint**:
    *   In the Extension Development Host window, open VS Code settings (Ctrl+,).
    *   Search for "OpenHands AI Assistant".
    *   Set the `Openhands: Api: Base Url` to the URL where your LLM API Aggregator backend is running (default is `http://localhost:8000`).

### Features & Commands

#### Ask OpenHands (`openhands.askOpenHands`)
   - **Purpose**: Get explanations, suggestions, or answers related to your code or general queries.
   - **How to use**:
     1. Optionally, select a piece of code in your editor that you want to include as context.
     2. Open the Command Palette (Ctrl+Shift+P).
     3. Type "Ask OpenHands" and select the command.
     4. Enter your question or prompt in the input box that appears.
   - **Functionality**: The extension sends your prompt, any selected code, and the content of your active file to the LLM Aggregator's `/v1/chat/completions` endpoint. The response is displayed in a new Markdown document.

## Usage Examples

### Basic Chat Completion
```python
from llm_aggregator import LLMAggregator

aggregator = LLMAggregator()
response = aggregator.chat_completion(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    model="auto"  # Automatically select best available model
)
```

### Provider-Specific Request
```python
response = aggregator.chat_completion(
    messages=[{"role": "user", "content": "Write Python code"}],
    provider="openrouter",
    model="deepseek/deepseek-coder-33b-instruct"
)
```

### Streaming Response
```python
for chunk in aggregator.chat_completion_stream(
    messages=[{"role": "user", "content": "Tell me a story"}],
    model="auto"
):
    print(chunk.choices[0].delta.content, end="")
```

## Advanced Features

### Intelligent Provider Selection
The system automatically selects the best provider based on:
- Model availability and capabilities
- Current rate limits and quotas
- Provider reliability and response time
- Cost optimization (prioritizing free tiers)

### Account Rotation
Automatically rotate between multiple accounts for the same provider to maximize free tier usage.

### Fallback Chains
Configure fallback chains for high availability:
```yaml
fallback_chains:
  coding:
    - provider: openrouter
      model: deepseek/deepseek-coder-33b-instruct
    - provider: groq
      model: llama-3.1-70b-versatile
    - provider: cerebras
      model: llama3.1-8b
```

## Monitoring and Analytics

- Real-time usage dashboard
- Provider performance metrics
- Cost tracking and optimization suggestions
- Rate limit monitoring and alerts

## Security

- Encrypted credential storage using industry-standard encryption
- API key rotation and management
- Audit logging for all requests
- Rate limiting and abuse prevention

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## License

MIT License - see `LICENSE` for details.