# Guide to `cli.py`

## Purpose

The `cli.py` script provides a command-line interface (CLI) for interacting with the LLM API Aggregator server. It allows users to perform various actions such as initiating chat sessions, listing available models, and checking system status directly from the terminal.

## How It Works

The script is structured around the `LLMClient` class and several command functions.

### `LLMClient` Class
*   Acts as an asynchronous HTTP client for the aggregator's API.
*   Uses the `httpx.AsyncClient` for making requests.
*   The base URL of the API server defaults to `http://localhost:8000` but can be configured via the `--url` argument.
*   **Methods**:
    *   `chat_completion()`: Sends requests to the `/v1/chat/completions` endpoint. It can handle both single responses and streaming responses.
    *   `list_models()`: Fetches data from `/v1/models`.
    *   `get_provider_status()`: Fetches data from `/admin/providers`.
    *   `get_usage_stats()`: Fetches data from `/admin/usage-stats`.
    *   `health_check()`: Fetches data from `/health`.
    *   `close()`: Closes the `httpx` client.

### Command Functions
*   **`chat_interactive()`**: Initiates an interactive chat session in the terminal. Uses `rich.prompt.Prompt` for user input and `rich.console.Console` for displaying messages. Supports `exit` and `clear` commands.
*   **`chat_single()`**: Sends a single message (provided via `--message`) and prints the assistant's response. Supports streaming.
*   **`list_models_command()`**: Retrieves and displays all available models, grouped by provider, using `rich.table.Table` for formatted output.
*   **`status_command()`**: Retrieves and displays the status of each provider and the overall system health check in tables.
*   **`stats_command()`**: Retrieves and displays usage statistics, including account usage and provider performance scores.

### Main Execution (`main` function)
*   Uses `argparse` to define commands and their respective arguments.
*   Global `--url` argument allows specifying the API server's base URL.
*   Supported commands: `chat`, `models`, `status`, `stats`.
*   The `chat` command has sub-arguments like `--model`, `--provider`, `--message`, and `--stream`.
*   Initializes `LLMClient` and calls the appropriate command function.
*   Includes basic error handling and ensures the client is closed on exit.

## How to Run

Execute `cli.py` from the project's root directory with a command:

```bash
python cli.py [command] [options]
```

**Examples:**
*   Start an interactive chat session:
    ```bash
    python cli.py chat
    ```
*   Send a single message:
    ```bash
    python cli.py chat --message "Hello, world!"
    ```
*   List all available models:
    ```bash
    python cli.py models
    ```
*   Check provider status:
    ```bash
    python cli.py status --url http://myapi.com:1234
    ```

## Key Interactions
*   Makes HTTP requests to the LLM API Aggregator server (endpoints defined in `src.api.server.py`).
*   Uses the `ChatMessage` model from `src.models`.

## Dependencies
*   `httpx`: For asynchronous HTTP requests.
*   `rich`: For formatted and interactive console output.

## Important Note on Admin Commands
The `status` and `stats` commands interact with admin-protected API endpoints (e.g., `/admin/providers`). If the server (`main.py` / `src.api.server.py`) has an `ADMIN_TOKEN` configured for security, these CLI commands will likely fail with an authentication error (401/403) because `cli.py` currently does not have a mechanism to send an admin token with its requests.
