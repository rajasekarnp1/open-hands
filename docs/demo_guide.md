# Guide to `demo.py`

## Purpose

The `demo.py` script is designed to showcase the core functionalities of the LLM API Aggregator's underlying components in a direct, programmatic way. Unlike `main.py` (which runs the API server) or `cli.py` (which interacts with the server via HTTP), `demo.py` instantiates and uses the `LLMAggregator` and its related classes (`AccountManager`, `ProviderRouter`, etc.) directly within the script itself.

This makes it useful for:
*   Understanding how the aggregator can be used as a library.
*   Demonstrating features without needing to run the full API server.
*   Basic testing or illustration of component interactions.

## How It Works

The script is primarily composed of two asynchronous functions: `demo_basic_functionality` and `demo_api_compatibility`, orchestrated by a `main` synchronous function.

### `demo_basic_functionality()`
This function demonstrates several key aspects:
1.  **Initialization**:
    *   It creates instances of `AccountManager`, `RateLimiter`.
    *   It instantiates providers (OpenRouter, Groq, Cerebras) using their respective `create_<provider>_provider([])` factory functions, passing empty lists of credentials (meaning they are initialized without any API keys for this demo).
    *   It sets up a `ProviderRouter` and then the main `LLMAggregator`.
2.  **List Available Models**: Calls `aggregator.list_available_models()` and uses `rich.table.Table` to display the first few models for each configured provider, showing their name, display name, context length, and free status.
3.  **Provider Status**: Calls `aggregator.get_provider_status()` and displays key status information for each provider in a table.
4.  **Intelligent Routing Demo (Conceptual)**:
    *   Defines a list of sample user prompts with "expected capabilities" (e.g., Code Generation, Reasoning).
    *   For each prompt, it creates a `ChatCompletionRequest` (with `model="auto"`) and then calls `router.get_provider_chain(request)`.
    *   It prints the resulting provider chain (e.g., `openrouter -> groq`). **Note**: This step only demonstrates the router's decision based on the request; it does *not* actually execute the chat completion against an LLM.
5.  **Rate Limiting Demo**: Calls `rate_limiter.get_rate_limit_status()` to display the initial (default) state of the global rate limits.
6.  **Account Management Demo**:
    *   Adds a couple of dummy (non-functional) credentials to the `AccountManager` for OpenRouter and Groq using `account_manager.add_credentials()`.
    *   Lists all stored credentials (showing provider, account ID, status, usage count) via `account_manager.list_credentials()`.
    *   Displays total usage statistics per provider via `account_manager.get_usage_stats()`.
7.  **Cleanup**: Calls `aggregator.close()` to properly shut down components like the internal HTTP client in providers.

### `demo_api_compatibility()`
*   This function does not execute any live calls.
*   It prints several example JSON request structures that are compatible with the OpenAI API. This serves to illustrate how users should format their requests if they are familiar with the OpenAI API and want to use the aggregator as a drop-in replacement (pointing to its `/v1/chat/completions` endpoint).

### `main()`
*   Synchronously calls `asyncio.run()` for `demo_basic_functionality()` and then for `demo_api_compatibility()`.
*   Prints introductory and concluding panels using `rich.panel.Panel`.

## How to Run

Execute `demo.py` from the project's root directory:
```bash
python demo.py
```

## Output Signification
The script will print a series of formatted tables and messages to the console, walking through each demonstration step. You will see:
*   Initialization messages.
*   Tables of available models.
*   Provider status tables.
*   The conceptual provider chains for different types of requests.
*   Initial rate limit statuses.
*   Tables of dummy credentials and their usage.
*   Example JSON request payloads.

## Key Interactions
*   Directly instantiates and calls methods on `LLMAggregator`, `AccountManager`, `ProviderRouter`, `RateLimiter`, and specific provider instances.
*   Does **not** start or communicate with the FastAPI server (`main.py`).
*   Uses `src.models` for `ChatMessage` and `ChatCompletionRequest`.

## Dependencies
*   Directly imports from `src.core.*` and `src.providers.*`.
*   `rich`: For all formatted console output.
*   `asyncio`: For running the asynchronous demo functions.
