# Guide to `main.py`

## Purpose

The `main.py` script serves as the primary entry point for starting the LLM API Aggregator's FastAPI server. It is responsible for initializing the server with the correct configurations and launching the Uvicorn ASGI server process.

## How It Works

1.  **Path Setup**: It adds the `src` directory to the Python system path to ensure that internal project modules (like those in `src.api` and `src.core`) can be correctly imported.
2.  **Logging Configuration (`setup_logging`)**:
    *   Defines a function to configure Python's standard `logging` module.
    *   Log messages are output to both the console (standard output) and a file named `llm_aggregator.log` in the root directory.
    *   The log level (e.g., INFO, DEBUG) can be set via a command-line argument.
3.  **Command-Line Argument Parsing**:
    *   Uses the `argparse` module to handle command-line arguments. Supported arguments include:
        *   `--host`: The network host to bind the server to (default: `0.0.0.0`, meaning all available network interfaces).
        *   `--port`: The port number for the server (default: `8000`).
        *   `--reload`: A flag to enable Uvicorn's auto-reload feature, useful during development (default: False).
        *   `--log-level`: Sets the application's logging level (default: `INFO`).
4.  **Server Execution**:
    *   After parsing arguments and setting up logging, it calls the `run_server` function imported from `src.api.server`.
    *   The `run_server` function (defined in `src/api/server.py`) is responsible for actually starting the Uvicorn server process with the FastAPI application instance.

## How to Run

To start the LLM API Aggregator server, execute `main.py` from the project's root directory:

```bash
python main.py [options]
```

**Examples:**
*   Run with default settings:
    ```bash
    python main.py
    ```
*   Run on a specific port with reload enabled and debug logging:
    ```bash
    python main.py --port 8080 --reload --log-level DEBUG
    ```

## Key Interactions
*   Imports and calls `run_server` from `src.api.server`.
*   Relies on the FastAPI application defined in `src.api.server`.
*   Uses settings from `src.config.settings` indirectly via the components initialized in `src.api.server.lifespan`.

## Dependencies
*   Relies on modules within the `src` directory.
*   Uses `argparse` (standard library).
*   The underlying server started by `src.api.server.run_server` uses `uvicorn` and `fastapi`.
