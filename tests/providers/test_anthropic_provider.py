import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.providers.anthropic import AnthropicProvider, create_anthropic_provider
from src.models import (
    ChatCompletionRequest,
    ChatMessage,
    AccountCredentials,
    ProviderConfig,
    ModelInfo,
    RateLimit,
    ProviderType,
    ModelCapability
)
from src.providers.base import ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def anthropic_config() -> ProviderConfig:
    """Returns a default AnthropicProvider config."""
    return AnthropicProvider.get_default_config()

@pytest.fixture
def mock_anthropic_credentials() -> AccountCredentials:
    """Returns mock Anthropic credentials."""
    return AccountCredentials(
        provider="anthropic",
        account_id="test_anthropic_key",
        api_key="sk-anthropic-test-key",
        is_active=True
    )

@pytest.fixture
def anthropic_provider(anthropic_config: ProviderConfig, mock_anthropic_credentials: AccountCredentials) -> AnthropicProvider:
    """Returns an AnthropicProvider instance with mocked credentials."""
    # The provider's __init__ expects a list of credentials, but for testing a single one is fine.
    # The actual credentials list is managed by AccountManager in the live app.
    provider = AnthropicProvider(config=anthropic_config, credentials=[mock_anthropic_credentials])
    return provider

# --- Tests for chat_completion ---

async def test_anthropic_chat_completion_success(anthropic_provider: AnthropicProvider, mock_anthropic_credentials: AccountCredentials):
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        model="claude-3-haiku-20240307",
        max_tokens=10
    )

    mock_sdk_client = AsyncMock()
    mock_anthropic_response = MagicMock()
    mock_anthropic_response.id = "msg_123"
    mock_anthropic_response.model = "claude-3-haiku-20240307"
    mock_anthropic_response.stop_reason = "end_turn"
    mock_anthropic_response.usage.input_tokens = 5
    mock_anthropic_response.usage.output_tokens = 10
    mock_anthropic_response.content = [MagicMock(type="text", text="Hi there!")]

    mock_sdk_client.messages.create = AsyncMock(return_value=mock_anthropic_response)

    # Patch the _get_sdk_client method to return our mock
    with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client):
        response = await anthropic_provider.chat_completion(request, mock_anthropic_credentials)

    assert response is not None
    assert response.id == "msg_123"
    assert response.model == "claude-3-haiku-20240307" # Should reflect requested or actual model from response
    assert len(response.choices) == 1
    assert response.choices[0].message.content == "Hi there!"
    assert response.choices[0].finish_reason == "end_turn"
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 10

async def test_anthropic_chat_completion_api_error_mapping(anthropic_provider: AnthropicProvider, mock_anthropic_credentials: AccountCredentials):
    import anthropic # Import for exception types

    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello")], model="claude-3-haiku-20240307")

    # Test different Anthropic errors and ensure they map to ProviderError subclasses
    error_mappings = [
        (anthropic.RateLimitError(message="Rate limit exceeded", response=MagicMock(headers={}), body=None), RateLimitError),
        (anthropic.AuthenticationError(message="Auth error", response=MagicMock(headers={}), body=None), AuthenticationError),
        (anthropic.NotFoundError(message="Not found", response=MagicMock(headers={}), body=None), ModelNotFoundError),
        (anthropic.APIStatusError(message="Generic API error", status_code=500, response=MagicMock(headers={}), body=None), ProviderError),
        (anthropic.APIConnectionError(message="Connection error"), ProviderError), # No status_code attribute
    ]

    for anthropic_exception, expected_provider_exception in error_mappings:
        mock_sdk_client = AsyncMock()
        mock_sdk_client.messages.create = AsyncMock(side_effect=anthropic_exception)

        with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client):
            with pytest.raises(expected_provider_exception):
                await anthropic_provider.chat_completion(request, mock_anthropic_credentials)

# --- Tests for chat_completion_stream ---

async def test_anthropic_chat_completion_stream_success(anthropic_provider: AnthropicProvider, mock_anthropic_credentials: AccountCredentials):
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Stream test")],
        model="claude-3-sonnet-20240229",
        stream=True
    )

    # Mock the stream events
    mock_stream_events = [
        MagicMock(type="message_start", message=MagicMock(id="msg_stream_123", usage=MagicMock(input_tokens=10, output_tokens=0))),
        MagicMock(type="content_block_delta", index=0, delta=MagicMock(type="text_delta", text="Hello ")),
        MagicMock(type="content_block_delta", index=0, delta=MagicMock(type="text_delta", text="World!")),
        MagicMock(type="message_delta", delta=MagicMock(stop_reason="end_turn"), usage=MagicMock(output_tokens=2)), # Example: 2 tokens for "Hello World!"
        MagicMock(type="message_stop")
    ]

    async def mock_event_stream():
        for event in mock_stream_events:
            yield event

    mock_sdk_client_stream = AsyncMock()
    # The stream object itself needs to be an async iterable context manager
    mock_stream_context_manager = AsyncMock()
    mock_stream_context_manager.__aenter__.return_value = mock_event_stream() # The async iterator
    mock_stream_context_manager.__aexit__.return_value = None

    mock_sdk_client_stream.messages.create = AsyncMock(return_value=mock_stream_context_manager)

    with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client_stream):
        chunks = []
        async for chunk in anthropic_provider.chat_completion_stream(request, mock_anthropic_credentials):
            chunks.append(chunk)

    assert len(chunks) > 0 # Should receive multiple chunks

    # Validate message_start event (first chunk usually)
    assert chunks[0].id == "msg_stream_123"
    assert chunks[0].usage.prompt_tokens == 10

    # Concatenate content from delta chunks
    streamed_content = ""
    finish_reason = None
    completion_tokens_from_delta_usage = 0

    for chunk in chunks:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            streamed_content += chunk.choices[0].delta.content
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if chunk.usage and chunk.usage.completion_tokens > 0 and chunk.choices and not chunk.choices[0].delta : # from message_delta
             completion_tokens_from_delta_usage = chunk.usage.completion_tokens


    assert streamed_content == "Hello World!"
    assert finish_reason == "end_turn"
    assert completion_tokens_from_delta_usage == 2 # from the message_delta event

async def test_anthropic_chat_completion_stream_api_error(anthropic_provider: AnthropicProvider, mock_anthropic_credentials: AccountCredentials):
    import anthropic
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Stream error test")], model="claude-3-opus-20240229", stream=True)

    mock_sdk_client_stream = AsyncMock()
    # Simulate an error during stream creation or iteration
    # For an error directly on .create() before streaming starts:
    mock_sdk_client_stream.messages.create = AsyncMock(side_effect=anthropic.APIConnectionError(message="Stream connection failed"))

    with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client_stream):
        with pytest.raises(ProviderError): # Should map to ProviderError or a subclass
            async for _ in anthropic_provider.chat_completion_stream(request, mock_anthropic_credentials):
                pass # pragma: no cover

# TODO: Add more tests for specific stream event sequences or error conditions during streaming if needed.

def test_create_anthropic_provider(mock_anthropic_credentials: AccountCredentials):
    """Test the create_anthropic_provider factory function."""
    provider = create_anthropic_provider([mock_anthropic_credentials])
    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic"
    assert provider.credentials == [mock_anthropic_credentials]

async def test_anthropic_system_prompt_handling(anthropic_provider: AnthropicProvider, mock_anthropic_credentials: AccountCredentials):
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is the capital of France?")
        ],
        model="claude-3-haiku-20240307"
    )

    mock_sdk_client = AsyncMock()
    mock_anthropic_response = MagicMock() # Configure as needed
    mock_anthropic_response.id = "msg_sys_123"
    mock_anthropic_response.model = "claude-3-haiku-20240307"
    mock_anthropic_response.stop_reason = "end_turn"
    mock_anthropic_response.usage.input_tokens = 15
    mock_anthropic_response.usage.output_tokens = 5
    mock_anthropic_response.content = [MagicMock(type="text", text="Paris.")]

    mock_sdk_client.messages.create = AsyncMock(return_value=mock_anthropic_response)

    with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client):
        await anthropic_provider.chat_completion(request, mock_anthropic_credentials)

    # Assert that messages.create was called with the system prompt correctly
    args, kwargs = mock_sdk_client.messages.create.call_args
    assert kwargs.get("system") == "You are a helpful assistant."
    assert kwargs.get("messages") == [{"role": "user", "content": "What is the capital of France?"}]

async def test_anthropic_multiple_system_prompts_concatenated(anthropic_provider: AnthropicProvider, mock_anthropic_credentials: AccountCredentials):
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="First system part."),
            ChatMessage(role="user", content="User message."),
            ChatMessage(role="system", content="Second system part.") # This should be appended
        ],
        model="claude-3-haiku-20240307"
    )
    mock_sdk_client = AsyncMock()
    mock_anthropic_response = MagicMock()
    mock_anthropic_response.id = "msg_multi_sys_123"
    # ... other mock attributes ...
    mock_sdk_client.messages.create = AsyncMock(return_value=mock_anthropic_response)

    with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client):
        await anthropic_provider.chat_completion(request, mock_anthropic_credentials)

    args, kwargs = mock_sdk_client.messages.create.call_args
    assert kwargs.get("system") == "First system part.\nSecond system part."
    assert kwargs.get("messages") == [{"role": "user", "content": "User message."}]

    # Test streaming as well
    mock_sdk_client_stream = AsyncMock()
    mock_stream_context_manager = AsyncMock()
    mock_stream_context_manager.__aenter__.return_value = AsyncMock() # Empty async iterator for this check
    mock_stream_context_manager.__aexit__.return_value = None
    mock_sdk_client_stream.messages.create = AsyncMock(return_value=mock_stream_context_manager)

    with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client_stream):
        async for _ in anthropic_provider.chat_completion_stream(request, mock_anthropic_credentials):
            pass # We only care about the call_args

    args_stream, kwargs_stream = mock_sdk_client_stream.messages.create.call_args
    assert kwargs_stream.get("system") == "First system part.\nSecond system part."
    assert kwargs_stream.get("messages") == [{"role": "user", "content": "User message."}]

async def test_anthropic_no_user_assistant_messages_error(anthropic_provider: AnthropicProvider, mock_anthropic_credentials: AccountCredentials):
    request_no_user_msg = ChatCompletionRequest(
        messages=[ChatMessage(role="system", content="System prompt only")],
        model="claude-3-haiku-20240307"
    )
    with pytest.raises(ProviderError, match="No user or assistant messages provided"):
        await anthropic_provider.chat_completion(request_no_user_msg, mock_anthropic_credentials)

    async def empty_stream_gen(): # Helper for testing stream error
        if False: yield # Make it an async generator

    mock_sdk_client_stream_err = AsyncMock()
    mock_stream_context_manager_err = AsyncMock()
    mock_stream_context_manager_err.__aenter__.return_value = empty_stream_gen()
    mock_stream_context_manager_err.__aexit__.return_value = None
    mock_sdk_client_stream_err.messages.create = AsyncMock(return_value=mock_stream_context_manager_err)

    with patch.object(anthropic_provider, '_get_sdk_client', return_value=mock_sdk_client_stream_err):
        with pytest.raises(ProviderError, match="No user or assistant messages provided for streaming"):
            async for _ in anthropic_provider.chat_completion_stream(request_no_user_msg, mock_anthropic_credentials):
                pass # pragma: no cover
```
