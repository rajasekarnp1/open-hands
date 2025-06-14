"""
Anthropic provider implementation.
"""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import anthropic
from .base import BaseProvider, ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionUsage,
    ModelInfo,
    ProviderConfig,
    AccountCredentials,
    RateLimit,
    ProviderType,
    ModelCapability,
)


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    @classmethod
    def get_default_config(cls) -> ProviderConfig:
        """Get default configuration for Anthropic."""
        models = [
            ModelInfo(
                name="claude-3-haiku-20240307",
                display_name="Claude 3 Haiku",
                provider="anthropic",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.SUMMARIZATION],
                context_length=200000, # Placeholder, verify actual
                is_free=False, # Typically not free
                rate_limit=RateLimit(requests_per_minute=100, tokens_per_minute=100000) # Placeholder
            ),
            ModelInfo(
                name="claude-3-sonnet-20240229",
                display_name="Claude 3 Sonnet",
                provider="anthropic",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.SUMMARIZATION, ModelCapability.ADVANCED_REASONING],
                context_length=200000, # Placeholder, verify actual
                is_free=False, # Typically not free
                rate_limit=RateLimit(requests_per_minute=50, tokens_per_minute=50000) # Placeholder
            ),
            ModelInfo(
                name="claude-3-opus-20240229",
                display_name="Claude 3 Opus",
                provider="anthropic",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, ModelCapability.SUMMARIZATION, ModelCapability.ADVANCED_REASONING],
                context_length=200000, # Placeholder, verify actual
                is_free=False, # Typically not free
                rate_limit=RateLimit(requests_per_minute=25, tokens_per_minute=25000) # Placeholder
            ),
        ]

        return ProviderConfig(
            name="anthropic",
            display_name="Anthropic",
            provider_type=ProviderType.COMMERCIAL, # Assuming it's a paid service
            base_url="https://api.anthropic.com", # Placeholder, SDK might handle this
            api_key_required=True,
            auth_header="x-api-key", # Common for Anthropic
            auth_prefix=None, # Anthropic usually doesn't use a prefix like "Bearer"
            models=models,
            default_model="claude-3-haiku-20240307",
            rate_limit=RateLimit(
                requests_per_minute=100, # General placeholder
                concurrent_requests=10
            ),
            priority=3, # Adjust as needed
            supports_streaming=True,
            supports_function_calling=False, # Verify this
            max_retries=3,
            timeout=60 # Increased timeout for potentially larger models
        )

    def __init__(self, config: ProviderConfig, credentials: List[AccountCredentials]):
        super().__init__(config, credentials)
        # The Anthropic SDK client will be initialized when a credential is available.
        self._sdk_client: Optional[anthropic.AsyncAnthropic] = None

    def _get_sdk_client(self, credentials: AccountCredentials) -> anthropic.AsyncAnthropic:
        """Initializes and returns the Anthropic SDK client."""
        if not credentials.api_key:
            raise AuthenticationError("API key is missing for Anthropic provider.")
        return anthropic.AsyncAnthropic(api_key=credentials.api_key)

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        credentials: AccountCredentials
    ) -> ChatCompletionResponse:
        """Perform chat completion."""
        client = self._get_sdk_client(credentials)

        # Convert request messages to Anthropic format
        # Anthropic expects messages in the format:
        # [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}, ...]
        # System prompt is handled separately.

        system_prompt = None
        anthropic_messages = []
        for msg in request.messages:
            if msg.role == "system":
                if system_prompt is None: # Anthropic only takes one system prompt
                    system_prompt = msg.content
                else: # Append to existing system prompt if multiple are given
                    system_prompt += "\n" + msg.content
            elif msg.role in ["user", "assistant"]:
                anthropic_messages.append({"role": msg.role, "content": msg.content})
            # "tool" role might need special handling if supported/added later

        if not anthropic_messages:
            raise ProviderError("No user or assistant messages provided.")

        try:
            start_time = time.time()
            api_response = await client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens or 1024, # Anthropic requires max_tokens
                messages=anthropic_messages,
                system=system_prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                # stop_sequences=request.stop, # Ensure compatibility
                stream=False,
            )
            response_time = time.time() - start_time
            self._update_metrics(response_time, 200) # Assuming 200 OK

        except anthropic.APIStatusError as e:
            self._update_metrics(time.time() - start_time, e.status_code)
            if e.status_code == 429:
                raise RateLimitError(f"Anthropic API rate limit exceeded: {e.message}", retry_after=e.response.headers.get("retry-after"))
            elif e.status_code == 401 or e.status_code == 403:
                raise AuthenticationError(f"Anthropic API authentication error: {e.message}")
            elif e.status_code == 404:
                raise ModelNotFoundError(f"Anthropic model not found: {request.model}. Error: {e.message}")
            else:
                raise ProviderError(f"Anthropic API error (status {e.status_code}): {e.message}")
        except anthropic.APIConnectionError as e:
            self._update_metrics(time.time() - start_time, 500) # Generic server error
            raise ProviderError(f"Anthropic API connection error: {e}")
        except Exception as e:
            self._update_metrics(time.time() - start_time, 500) # Generic server error
            raise ProviderError(f"An unexpected error occurred with Anthropic provider: {str(e)}")

        # Parse Anthropic response to ChatCompletionResponse
        choices = []
        if api_response.content:
            for i, content_block in enumerate(api_response.content):
                if content_block.type == "text":
                    choice = ChatCompletionChoice(
                        index=i,
                        message=ChatMessage(role="assistant", content=content_block.text),
                        finish_reason=api_response.stop_reason,
                    )
                    choices.append(choice)

        usage = ChatCompletionUsage(
            prompt_tokens=api_response.usage.input_tokens,
            completion_tokens=api_response.usage.output_tokens,
            total_tokens=(api_response.usage.input_tokens + api_response.usage.output_tokens)
        )

        return ChatCompletionResponse(
            id=api_response.id,
            created=int(time.time()), # Anthropic response doesn't provide 'created' timestamp
            model=request.model, # Or api_response.model
            provider=self.name,
            choices=choices,
            usage=usage,
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        credentials: AccountCredentials
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Perform streaming chat completion."""
        client = self._get_sdk_client(credentials)

        system_prompt = None
        anthropic_messages = []
        for msg in request.messages:
            if msg.role == "system":
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    system_prompt += "\n" + msg.content
            elif msg.role in ["user", "assistant"]:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        if not anthropic_messages:
            raise ProviderError("No user or assistant messages provided for streaming.")

        try:
            start_time = time.time() # For overall stream attempt

            async with await client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens or 1024,
                messages=anthropic_messages,
                system=system_prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                # stop_sequences=request.stop,
                stream=True,
            ) as stream:
                # Successfully initiated stream
                self._update_metrics(time.time() - start_time, 200) # Assuming 200 OK for stream start

                async for event in stream:
                    if event.type == "message_start":
                        # message_start event contains the initial message object with usage info
                        # We can yield an initial response with this info if needed, or store it
                        # For now, we'll use it at the end if it's the final event.
                        # Anthropic's SDK v1.0+ provides usage in message_start
                        prompt_tokens = event.message.usage.input_tokens
                        # Create an initial ChatCompletionResponse, id and model are important
                        # Other fields will be populated by delta events
                        yield ChatCompletionResponse(
                            id=event.message.id,
                            created=int(time.time()),
                            model=request.model, # or event.message.model
                            provider=self.name,
                            choices=[], # Will be filled by subsequent deltas
                            usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens)
                        )

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            choice = ChatCompletionChoice(
                                index=event.index, # Anthropic provides content_block_index
                                delta=ChatMessage(role="assistant", content=event.delta.text),
                                finish_reason=None,
                            )
                            yield ChatCompletionResponse(
                                id="", # Will be same as message_start id, can be omitted in delta
                                created=int(time.time()),
                                model=request.model,
                                provider=self.name,
                                choices=[choice],
                                usage=None # Usage often comes at the end or start for streams
                            )
                    elif event.type == "message_delta":
                        # This event often contains the final usage and stop_reason
                        # We might need to send one last message with this
                        usage_data = event.usage
                        completion_tokens = usage_data.output_tokens

                        # Yield a final message with finish reason and updated usage
                        yield ChatCompletionResponse(
                            id="", # Can be omitted
                            created=int(time.time()),
                            model=request.model,
                            provider=self.name,
                            choices=[ChatCompletionChoice(
                                index=0, # Assuming one choice for now
                                delta=None, # No new content
                                finish_reason=event.delta.stop_reason # Get stop reason from message_delta
                            )],
                            usage=ChatCompletionUsage(
                                prompt_tokens=0, # Prompt tokens were in message_start
                                completion_tokens=completion_tokens,
                                total_tokens=completion_tokens # This is partial, needs prompt_tokens too
                            )
                        )
                    elif event.type == "message_stop":
                        # The stream has ended.
                        # Anthropic SDK handles this, we might not need to do anything specific here
                        # unless we need to aggregate final metrics or response.
                        # The SDK's stream object should automatically stop iteration.
                        pass


        except anthropic.APIStatusError as e:
            # Note: Cannot call _update_metrics here if headers not sent yet for stream
            if e.status_code == 429:
                raise RateLimitError(f"Anthropic API rate limit exceeded: {e.message}", retry_after=e.response.headers.get("retry-after"))
            elif e.status_code == 401 or e.status_code == 403:
                raise AuthenticationError(f"Anthropic API authentication error: {e.message}")
            elif e.status_code == 404:
                raise ModelNotFoundError(f"Anthropic model not found: {request.model}. Error: {e.message}")
            else:
                raise ProviderError(f"Anthropic API error (status {e.status_code}): {e.message}")
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"Anthropic API connection error: {e}")
        except Exception as e:
            raise ProviderError(f"An unexpected error occurred with Anthropic streaming: {str(e)}")

def create_anthropic_provider(credentials: List[AccountCredentials]) -> AnthropicProvider:
    """Create Anthropic provider instance."""
    config = AnthropicProvider.get_default_config()
    # Potentially load dynamic config or merge with user provided config here
    return AnthropicProvider(config, credentials)
