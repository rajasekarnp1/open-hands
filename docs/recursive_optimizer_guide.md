# Guide to `recursive_optimizer.py`

## Purpose

The `recursive_optimizer.py` script is an experimental component designed to simulate a recursive self-improvement process for a part of the OpenHands system. Specifically, it targets the `OpenHandsIntegrator` class within `experimental_optimizer.py`. Its goal is to demonstrate a conceptual loop where the system:
1.  Analyzes its own codebase (the `OpenHandsIntegrator`).
2.  Generates a new, theoretically "improved" version (clone) of that codebase with added features.
3.  Simulates testing this new version.
4.  Simulates "deploying" the improved version.
5.  Repeats this cycle.

It's important to note that the "improvements" and "analysis" are largely simulated and based on predefined templates and hardcoded logic rather than actual AI-driven code generation or deep semantic analysis.

## How It Works

The script defines a `RecursiveSelfOptimizer` class that orchestrates the self-improvement cycle.

### 1. Self-Analysis Phase (`analyze_openhands_implementation`)
*   **Target**: The optimizer is hardcoded to read and analyze the `OpenHandsIntegrator` class within the `experimental_optimizer.py` file (located in the project root).
*   **Method**: It uses Python's Abstract Syntax Tree (`ast`) module to parse the source code of `experimental_optimizer.py`. It walks the AST to find the `OpenHandsIntegrator` class definition.
*   **Analysis**:
    *   It extracts a list of method names within the class.
    *   It calculates a basic complexity score based on the number of methods and lines in the class body.
    *   Crucially, the "improvement opportunities," "suggested enhancements," and "performance bottlenecks" are **hardcoded lists of strings** within this method. They are not dynamically derived from the code's content beyond the structural parsing.

### 2. Clone Creation Phase (`create_improved_clone`)
*   **Naming**: Generates a new class name for the clone (e.g., `OpenHandsIntegratorV1`, `OpenHandsIntegratorV2`, etc.) based on the current improvement cycle.
*   **Code Generation**:
    *   It calls `_generate_improved_implementation` which uses a large f-string template to create the Python code for the new clone class.
    *   This template embeds some of the "suggested enhancements" (from the hardcoded list) into the docstring of the generated class.
    *   The generated class includes new attributes like `task_cache`, `performance_predictor` (an instance of a nested `SimpleMLPredictor` class), `executor` (a `ThreadPoolExecutor`), `optimization_patterns` (a `deque`), and `success_metrics`.
    *   Methods like `create_openhands_optimization_session` and `simulate_openhands_execution` in the clone are "enhanced" versions from the template, often including calls to the new `performance_predictor` or branching logic for simulated parallel execution.
*   **Saving**: The generated Python code for the clone is saved as a new file (e.g., `openhandsintegratorv1.py`) in the project root directory.
*   **`SimpleMLPredictor`**: The generated clone code also includes the definition for `SimpleMLPredictor`. This class has a `predict_task_success` method with hardcoded logic to return a score based on task attributes.

### 3. Testing & Validation Phase (`test_clone_performance`)
*   This phase is entirely simulated.
*   It includes an `asyncio.sleep(2)` to mimic testing time.
*   It calculates an "actual gain" based on the `performance_gain` attributed to the clone during its creation (which itself is based on the number of hardcoded enhancements).
*   It returns a dictionary of mock test results, including a hardcoded stability score.

### 4. Deployment Phase (`deploy_improved_clone`)
*   This is also simulated.
*   It checks if the clone was "tested."
*   It includes an `asyncio.sleep(1)` to mimic deployment time.
*   It updates the `status` of the clone to "deployed" and sets it as the `active_clone` in the optimizer.
*   The `improvement_cycles` counter is incremented here.

### Continuous Evolution Cycle (`start_continuous_self_improvement`)
*   This method runs the above phases in a loop for a specified number of cycles (`max_cycles`).
*   It uses the `rich` library to print formatted panels and tables to the console, showing the progress and results of each cycle.

## How to Run

The script can be executed directly:
```bash
python recursive_optimizer.py
```
By default, its `main()` function will run 3 improvement cycles with a 5-second interval between them.

## Output Signification

The script will output:
*   Progress of each phase (analysis, clone creation, testing, deployment) for each cycle.
*   A summary table after each cycle showing the status of each component.
*   A final summary table showing the evolution history of generated clones.

The key things to observe are:
*   Whether it successfully finds and "analyzes" `experimental_optimizer.py`.
*   The names of the generated clone files (e.g., `openhandsintegratorv1.py`). These files will appear in the root directory.
*   The (simulated) performance gains reported.

## Current Status & Limitations

*   **Simulated Nature**: The core "intelligence" (analysis, enhancement generation, ML prediction, performance testing) is simulated with hardcoded logic and templates. It does not perform actual AI-driven code analysis or improvement.
*   **File Path Dependency**: It relies on `experimental_optimizer.py` being present in the same directory where `recursive_optimizer.py` is run. This path was previously hardcoded incorrectly and has been fixed. Generated files are also saved to this directory.
*   **AST Parsing**: The analysis relies on basic AST parsing to find a class and its methods. Complex code structures or errors in the target file could break this parsing.
*   **Error Handling**: Basic, primarily focused on `FileNotFoundError` for the target script.
*   **Dependencies**: Requires the `rich` library for console output.

This script serves as a conceptual demonstration of a self-improving system rather than a production-ready optimization engine.
