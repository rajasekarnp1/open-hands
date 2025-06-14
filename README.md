# 🤖 Multi-Provider LLM API Aggregator

A production-grade solution for switching between different free LLM API providers with intelligent routing, account management, and fallback mechanisms.

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
- 🎯 **Flexible Routing Control**:
    - Specify a `provider` directly in your API requests.
    - Influence model selection with `model_quality` parameter (`fastest`, `best_quality`, `balanced`).
- 🔑 **Bring Your Own Key (BYOK)**: Easily configure your own API keys for premium providers.
- 📈 **Usage Analytics**: Track usage across providers and accounts
- 🛡️ **Error Handling**: Robust error handling and retry mechanisms
- 🔄 **Auto-Updater**: Automatic discovery of new providers and models
- 🧠 **Meta-Controller**: Research-based intelligent model selection
- 🎯 **Ensemble System**: Multi-model response fusion and quality assessment

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

### Coding Agent

Endpoint: `POST /v1/agents/code/invoke`

This endpoint provides access to a specialized agent for code generation tasks.

**Request Body (`CodeAgentRequest`):**
```json
{
  "instruction": "Create a Python function that returns the factorial of a number.",
  "context": "def existing_function():\n  pass", // Optional: existing code or context
  "language": "python", // Optional: "python", "javascript", etc.
  "project_directory": "/path/to/user/project", // Optional: Enables filesystem tools if provided
  "project_directory": "/path/to/user/project", // Optional: Enables filesystem tools
  "thread_id": "thread_abc123", // Optional: For stateful, multi-turn conversations
  "model_quality": "best_quality", // Optional
  "provider": null // Optional
}
```

**Example Response (`CodeAgentResponse`):**
```json
{
  "generated_code": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
  "explanation": "This function calculates factorial recursively. The base case is n=0, where factorial is 1.",
  "agent_status": "completed", // or "requires_human_input", "error"
  "human_input_request": null, // Or details if agent_status is "requires_human_input"
  "request_params": { /* original request parameters */ },
  "model_used": "anthropic/claude-3-opus-20240229", // Example
  "error_details": null // Or error message if agent_status is "error"
}
```

If `agent_status` is `"requires_human_input"`, the `human_input_request` field will contain:
```json
{
  "tool_call_id": "tool_call_xyz789", // ID of the ask_human_for_input tool call
  "question_for_human": "Should I proceed with deleting the old file after refactoring?"
}
```

**Example Curl:**
```bash
curl -X POST http://localhost:8000/v1/agents/code/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Write a javascript function to greet a user.",
    "language": "javascript",
    "model_quality": "best_quality"
  }'
```

### Resume Agent Execution

Endpoint: `POST /v1/agents/resume`

Used to resume an agent's operation after it has paused for human input (i.e., when a previous call to `/v1/agents/code/invoke` returned `agent_status="requires_human_input"`).

**Request Body (`ResumeAgentRequest`):**
```json
{
  "thread_id": "thread_abc123", // The ID of the conversation thread to resume
  "tool_call_id": "tool_call_xyz789", // The ID from the HumanInputRequest
  "human_response": "Yes, please proceed with deleting the old file." // User's answer
}
```

**Response:** The response will be a standard `CodeAgentResponse`, reflecting the agent's state after processing the human input and continuing its operation (it might complete, error out, or even ask for human input again).


## Agent Capabilities

The OpenHands agent framework includes several advanced features:

### 1. Standardized Tool Definition & Usage
Agents can be equipped with tools to interact with external environments (like a file system).
- **Tool Definition**: Tools are defined as Python functions and decorated with `@openhands_tool` (from `src.agents.tools.base`). This decorator inspects the function's signature and docstring to create a structured `ToolDefinition` (including name, description, and parameter schemas).
- **Tool Registry**: A `global_tool_registry` (instance of `ToolRegistry` from `src.agents.tools.registry`) stores all defined tools. Agents use this registry to:
    - Discover available tools and their schemas to present to the LLM.
    - Retrieve the actual tool function for execution based on the LLM's request.
- **Parameter Validation**: When an LLM requests a tool call, the provided parameters are validated against the tool's schema before execution. If validation fails, an error is returned to the LLM, allowing it to correct the parameters.
- **Filesystem Tools**: Pre-built tools like `read_file`, `write_file`, and `list_files` are available. These tools operate securely within a specified `project_directory`.
    - `read_file(filepath: str)`: Reads content of a relative path.
    - `write_file(filepath: str, content: str)`: Writes content to a relative path.
    - `list_files(directory_path: str = "")`: Lists files/subdirectories at a relative path.

### 2. Stateful Agents & Checkpointing
Agents can maintain conversational state across multiple API calls using a `thread_id`.
- **`thread_id`**: Include this in your `CodeAgentRequest` to maintain context. If a `thread_id` is provided, the agent will load its previous state (conversation history, etc.) associated with that ID. If no `thread_id` is given or no state is found, a new conversation thread is started.
- **Checkpointing**: The agent's state (instance of `SessionState` from `src.agents.state`) is saved after each significant interaction (e.g., LLM response, tool execution).
- **`InMemoryCheckpointManager`**: Currently, an in-memory checkpoint manager is used. This means state persists for the lifetime of the server application but is lost on restart. Future implementations may include persistent storage options (e.g., Redis, database).

### 3. Human-in-the-Loop (HITL)
Agents can request human input or clarification during their execution.
- **`ask_human_for_input` Tool**: An agent can decide to use this special tool if it needs human guidance. The tool takes a `question` parameter.
- **API Flow for HITL**:
    1. Client sends a request to `/v1/agents/code/invoke` (optionally with a `thread_id`).
    2. If the agent calls the `ask_human_for_input` tool, the API response will have:
       - `agent_status: "requires_human_input"`
       - `human_input_request`: An object containing `tool_call_id` (identifying this specific pause) and `question_for_human`.
       The `conversation_history` up to this point is saved by the agent.
    3. The client application should display `question_for_human` to the user.
    4. Once the user provides input, the client sends this input to the `POST /v1/agents/resume` endpoint using a `ResumeAgentRequest`. This request must include the original `thread_id`, the `tool_call_id` from the `human_input_request`, and the `human_response`.
    5. The backend loads the saved state for the `thread_id`, appends the human's response as a "tool" message (linked via `tool_call_id`), and resumes the agent's execution.
    6. The response from `/v1/agents/resume` will be another `CodeAgentResponse`, indicating the agent's subsequent actions (e.g., completion, another tool call, or even another HITL request).

**Example HITL API Interaction:**

*Initial Request to `/v1/agents/code/invoke` (same as Coding Agent example, include `thread_id`)*

*Example Response indicating HITL pause:*
```json
{
  "generated_code": null,
  "explanation": "I need to know if I should overwrite the existing file 'config.py'.",
  "agent_status": "requires_human_input",
  "human_input_request": {
    "tool_call_id": "hitl_call_12345",
    "question_for_human": "The file 'config.py' already exists. Should I overwrite it?"
  },
  "request_params": { /* ... */ },
  "model_used": "gpt-4-turbo"
}
```

*Client sends user's answer to `/v1/agents/resume`:*
```json
{
  "thread_id": "thread_abc123",
  "tool_call_id": "hitl_call_12345",
  "human_response": "Yes, please overwrite the file."
}
```
The agent then continues processing.


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

#### 1. Ask OpenHands (`openhands.askOpenHands`)
   - **Purpose**: Get explanations, suggestions, or answers related to your code or general queries.
   - **How to use**:
     1. Optionally, select a piece of code in your editor that you want to include as context.
     2. Open the Command Palette (Ctrl+Shift+P).
     3. Type "Ask OpenHands" and select the command.
     4. Enter your question or prompt in the input box that appears.
   - **Functionality**: The extension sends your prompt, any selected code, and the content of your active file to the LLM Aggregator's `/v1/contextual_chat/completions` endpoint. The response is displayed in a new Markdown document.

#### 2. Ask OpenHands Code Agent (`openhands.invokeCodeAgent`)
   - **Purpose**: Generate code, refactor existing code, or perform other coding-specific tasks.
   - **How to use**:
     1. Optionally, select a piece of code in your editor to be used as context for modification or as a reference.
     2. Open the Command Palette (Ctrl+Shift+P).
     3. Type "Ask OpenHands Code Agent" and select the command.
     4. Enter your instruction for the code agent (e.g., "Create a Python class for a User", "Refactor this selected Java code to use streams").
   - **Functionality**:
     *   The extension sends your instruction, selected code (as context), the detected language of the active file, and the **current VS Code workspace root path (if a workspace is open) as `project_directory`** to the `/v1/agents/code/invoke` endpoint.
     *   If a `project_directory` is sent, the Code Agent can use filesystem tools (read, write, list files) to fulfill your request. For example, "Read 'src/utils.py' and then create a new file 'src/importer.py' that uses functions from utils."
     *   The `generated_code` from the response is inserted directly into your editor:
         - If you had text selected, it replaces the selection.
         - If no text was selected, it inserts the code at your cursor position.
     *   If an `explanation` is provided by the agent, it's shown in a new Markdown document opened beside your current editor.

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