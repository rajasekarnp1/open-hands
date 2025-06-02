"""
Cerebras provider implementation.
"""

from typing import List
from .base import OpenAICompatibleProvider
from ..models import (
    ProviderConfig, 
    ModelInfo, 
    RateLimit, 
    ProviderType,
    ModelCapability
)


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras provider implementation."""
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Cerebras."""
        
        # Define available models on Cerebras
        models = [
            ModelInfo(
                name="llama3.1-8b",
                display_name="Llama 3.1 8B",
                provider="cerebras",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,  # Free tier restricted to 8K context
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_minute=30,
                    tokens_per_minute=60000,
                    requests_per_hour=900,
                    tokens_per_hour=1000000,
                    requests_per_day=14400,
                    tokens_per_day=1000000
                )
            ),
            ModelInfo(
                name="llama3.3-70b",
                display_name="Llama 3.3 70B",
                provider="cerebras",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,  # Free tier restricted to 8K context
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_minute=30,
                    tokens_per_minute=60000,
                    requests_per_hour=900,
                    tokens_per_hour=1000000,
                    requests_per_day=14400,
                    tokens_per_day=1000000
                )
            ),
            ModelInfo(
                name="qwen3-32b",
                display_name="Qwen 3 32B",
                provider="cerebras",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,  # Free tier restricted to 8K context
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_minute=30,
                    tokens_per_minute=60000,
                    requests_per_hour=900,
                    tokens_per_hour=1000000,
                    requests_per_day=14400,
                    tokens_per_day=1000000
                )
            ),
            ModelInfo(
                name="llama-4-scout",
                display_name="Llama 4 Scout",
                provider="cerebras",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,  # Free tier restricted to 8K context
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_minute=30,
                    tokens_per_minute=60000,
                    requests_per_hour=900,
                    tokens_per_hour=1000000,
                    requests_per_day=14400,
                    tokens_per_day=1000000
                )
            ),
        ]
        
        return ProviderConfig(
            name="cerebras",
            display_name="Cerebras",
            provider_type=ProviderType.FREE,
            base_url="https://api.cerebras.ai/v1",
            api_key_required=True,
            auth_header="Authorization",
            auth_prefix="Bearer",
            models=models,
            rate_limit=RateLimit(
                requests_per_minute=30,
                tokens_per_minute=60000,
                requests_per_hour=900,
                tokens_per_hour=1000000,
                requests_per_day=14400,
                tokens_per_day=1000000,
                concurrent_requests=5
            ),
            priority=3,  # Good priority due to fast inference
            supports_streaming=True,
            supports_function_calling=False,
            max_retries=3,
            timeout=30
        )


def create_cerebras_provider(credentials: List) -> CerebrasProvider:
    """Create Cerebras provider instance."""
    config = CerebrasProvider.get_default_config()
    return CerebrasProvider(config, credentials)