"""
Groq provider implementation.
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


class GroqProvider(OpenAICompatibleProvider):
    """Groq provider implementation."""
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Groq."""
        
        # Define available models on Groq
        models = [
            ModelInfo(
                name="llama-3.3-70b-versatile",
                display_name="Llama 3.3 70B Versatile",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=32768,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=1000,
                    tokens_per_minute=12000
                )
            ),
            ModelInfo(
                name="llama-3.1-8b-instant",
                display_name="Llama 3.1 8B Instant",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=131072,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=14400,
                    tokens_per_minute=6000
                )
            ),
            ModelInfo(
                name="llama-3-70b-8192",
                display_name="Llama 3 70B",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=14400,
                    tokens_per_minute=6000
                )
            ),
            ModelInfo(
                name="llama-3-8b-8192",
                display_name="Llama 3 8B",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=14400,
                    tokens_per_minute=6000
                )
            ),
            ModelInfo(
                name="gemma2-9b-it",
                display_name="Gemma 2 9B Instruct",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=14400,
                    tokens_per_minute=15000
                )
            ),
            ModelInfo(
                name="deepseek-r1-distill-llama-70b",
                display_name="DeepSeek R1 Distill Llama 70B",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                context_length=32768,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=1000,
                    tokens_per_minute=6000
                )
            ),
            ModelInfo(
                name="qwq-32b-preview",
                display_name="QwQ 32B Preview",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                context_length=32768,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=1000,
                    tokens_per_minute=6000
                )
            ),
            ModelInfo(
                name="llama-4-scout-instruct",
                display_name="Llama 4 Scout Instruct",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=32768,
                is_free=True,
                rate_limit=RateLimit(
                    requests_per_day=1000,
                    tokens_per_minute=30000
                )
            ),
        ]
        
        return ProviderConfig(
            name="groq",
            display_name="Groq",
            provider_type=ProviderType.FREE,
            base_url="https://api.groq.com/openai/v1",
            api_key_required=True,
            auth_header="Authorization",
            auth_prefix="Bearer",
            models=models,
            rate_limit=RateLimit(
                requests_per_minute=30,
                requests_per_day=14400,
                tokens_per_minute=30000,
                concurrent_requests=10
            ),
            priority=2,  # High priority due to fast inference
            supports_streaming=True,
            supports_function_calling=True,
            max_retries=3,
            timeout=30
        )


def create_groq_provider(credentials: List) -> GroqProvider:
    """Create Groq provider instance."""
    config = GroqProvider.get_default_config()
    return GroqProvider(config, credentials)