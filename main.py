#!/usr/bin/env python3
"""
Main entry point for the LLM API Aggregator.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.server import run_server


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("llm_aggregator.log")
        ]
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LLM API Aggregator")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run server
    print(f"Starting LLM API Aggregator on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()