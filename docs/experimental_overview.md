# Overview of Experimental and Self-Improving Scripts

This project contains several Python scripts focused on experimental ideas around system self-improvement, AI-driven optimization, and advanced automation concepts. Most of these scripts operate at a **simulated level**, demonstrating conceptual workflows rather than implementing production-ready, AI-driven functionalities.

Here's a brief overview of the key scripts and their conceptual relationships:

-   **`experimental_optimizer.py`**:
    -   Acts as a central sandbox for various experimental features.
    -   It simulates integrations with concepts from DSPy (prompt optimization), AutoGen (multi-agent systems), and LangChain (processing chains).
    -   Contains an `OpenHandsIntegrator` class which is a target for self-improvement by `recursive_optimizer.py`.
    -   The simulations within this script are largely hardcoded. For detailed guide, see `docs/experimental_optimizer_guide.md`.

-   **`recursive_optimizer.py`**:
    -   This script attempts to simulate a recursive self-improvement loop.
    -   It is designed to analyze the `OpenHandsIntegrator` class within `experimental_optimizer.py`.
    -   It then generates "improved" clone versions of this class (e.g., `OpenHandsIntegratorV1.py`, `V2.py`, etc.) by injecting predefined enhancements using code templates.
    -   The analysis and improvements are simulated. For detailed guide, see `docs/recursive_optimizer_guide.md`.

-   **`openhands_improver.py`**:
    -   This script takes a broader, project-level approach. It's designed to simulate cloning the entire OpenHands GitHub repository.
    -   It then performs a mixed (partially real AST parsing, partially hardcoded) analysis to identify "improvement areas."
    -   It "improves" the cloned codebase by injecting a set of predefined, new Python modules (e.g., for caching, AI optimization, monitoring) into the clone.
    -   It simulates creating a pull request with these changes.
    -   This script operates independently of `experimental_optimizer.py` and `recursive_optimizer.py`'s direct class modification. For detailed guide, see `docs/openhands_improver_guide.md`.

-   **`ai_scientist_openhands.py`**:
    -   This is another standalone, highly conceptual script.
    -   It simulates an "AI-Scientist" approach to project improvement, including generating research hypotheses, running (mocked) experiments, and integrating with (mocked) advanced infrastructure like Lightning AI and custom analysis VMs.
    -   Its core logic, analyses, and outcomes are heavily simulated and based on hardcoded data. For detailed guide, see `docs/ai_scientist_openhands_guide.md`.

**General Notes on These Scripts:**
*   **Simulation**: Their primary value is in demonstrating complex concepts. They do not represent production-ready AI systems.
*   **Dependencies**: Some may have implicit dependencies (like `gitpython` or the `git` CLI for `openhands_improver.py`) not fully listed in `requirements.txt`.
*   **Execution in Sandbox**: Due to environmental limitations (e.g., missing `git`, disk space for full dependencies like `torch`), running all of them fully or installing all their potential dependencies might be challenging in some environments.
