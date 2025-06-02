"""
OpenRouter provider implementation.
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


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider implementation."""
    
    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for OpenRouter."""
        
        # Define available free models on OpenRouter
        models = [
            ModelInfo(
                name="deepseek/deepseek-r1:free",
                display_name="DeepSeek R1 (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                context_length=32768,
                is_free=True
            ),
            ModelInfo(
                name="deepseek/deepseek-chat:free",
                display_name="DeepSeek V3 (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=32768,
                is_free=True
            ),
            ModelInfo(
                name="meta-llama/llama-3.1-8b-instruct:free",
                display_name="Llama 3.1 8B Instruct (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=131072,
                is_free=True
            ),
            ModelInfo(
                name="meta-llama/llama-3.3-70b-instruct:free",
                display_name="Llama 3.3 70B Instruct (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=131072,
                is_free=True
            ),
            ModelInfo(
                name="qwen/qwen-2.5-72b-instruct:free",
                display_name="Qwen 2.5 72B Instruct (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=32768,
                is_free=True
            ),
            ModelInfo(
                name="qwen/qwen-2.5-coder-32b-instruct:free",
                display_name="Qwen 2.5 Coder 32B (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.CODE_GENERATION],
                context_length=32768,
                is_free=True
            ),
            ModelInfo(
                name="google/gemma-2-9b-it:free",
                display_name="Gemma 2 9B Instruct (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,
                is_free=True
            ),
            ModelInfo(
                name="mistralai/mistral-7b-instruct:free",
                display_name="Mistral 7B Instruct (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=32768,
                is_free=True
            ),
            ModelInfo(
                name="microsoft/phi-4-reasoning:free",
                display_name="Phi-4 Reasoning (Free)",
                provider="openrouter",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                context_length=16384,
                is_free=True
            ),
        ]
        
        return ProviderConfig(
            name="openrouter",
            display_name="OpenRouter",
            provider_type=ProviderType.FREE,
            base_url="https://openrouter.ai/api/v1",
            api_key_required=True,
            auth_header="Authorization",
            auth_prefix="Bearer",
            models=models,
            rate_limit=RateLimit(
                requests_per_minute=20,
                requests_per_day=50,  # Free tier: 50 requests/day, 1000 with $10 topup
                concurrent_requests=5
            ),
            priority=1,  # High priority due to many free models
            supports_streaming=True,
            supports_function_calling=False,
            max_retries=3,
            timeout=30
        )


def create_openrouter_provider(credentials: List) -> OpenRouterProvider:
    """Create OpenRouter provider instance."""
    config = OpenRouterProvider.get_default_config()
    return OpenRouterProvider(config, credentials)