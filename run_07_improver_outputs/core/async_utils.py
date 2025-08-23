#!/usr/bin/env python3
"""
Async Optimization Utilities

Utilities for converting and optimizing async/await patterns.
"""

import asyncio
import functools
from typing import Any, Callable, Coroutine, List
import logging

logger = logging.getLogger(__name__)

def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for async retry logic."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")

            raise last_exception

        return wrapper
    return decorator

async def gather_with_concurrency(coros: List[Coroutine], max_concurrency: int = 10) -> List[Any]:
    """Execute coroutines with limited concurrency."""

    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[sem_coro(coro) for coro in coros])

async def timeout_wrapper(coro: Coroutine, timeout: float) -> Any:
    """Wrap coroutine with timeout."""

    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout} seconds")
        raise

class AsyncBatchProcessor:
    """Process items in async batches."""

    def __init__(self, batch_size: int = 100, max_concurrency: int = 10):
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    async def process(self, items: List[Any], processor: Callable) -> List[Any]:
        """Process items in batches."""

        results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_coros = [processor(item) for item in batch]

            batch_results = await gather_with_concurrency(
                batch_coros,
                self.max_concurrency
            )

            results.extend(batch_results)

            # Small delay between batches to prevent overwhelming
            if i + self.batch_size < len(items):
                await asyncio.sleep(0.1)

        return results
