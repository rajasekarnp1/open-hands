import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.code_agent import CodeAgent, CODE_EXPLANATION_SEPARATOR
from src.models import (
    CodeAgentRequest,
    CodeAgentResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionChoice,
    ChatCompletionUsage
)
from src.core.aggregator import LLMAggregator

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_llm_aggregator() -> AsyncMock:
    """Mocks the LLMAggregator."""
    mock = AsyncMock(spec=LLMAggregator)
    return mock

@pytest.fixture
def code_agent(mock_llm_aggregator: AsyncMock) -> CodeAgent:
    """Fixture for CodeAgent with a mocked LLMAggregator."""
    return CodeAgent(llm_aggregator=mock_llm_aggregator)

# --- Test Cases ---

async def test_code_agent_generate_code_simple_instruction(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="Create a hello world function in Python.")

    # Mock LLMAggregator's response
    mock_llm_response_content = "def hello():\n    print('Hello, World!')"
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-123",
        created=1234567890,
        model="mock_model_used",
        provider="mock_provider",
        choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=mock_llm_response_content), finish_reason="stop")
        ],
        usage=ChatCompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)

    response = await code_agent.generate_code(request)

    assert response.generated_code == "def hello():\n    print('Hello, World!')"
    assert response.explanation is None
    assert response.model_used == "mock_model_used"

    # Verify prompt construction and call to aggregator
    mock_llm_aggregator.chat_completion.assert_called_once()
    called_chat_request: ChatCompletionRequest = mock_llm_aggregator.chat_completion.call_args[0][0]

    assert "Instruction:\nCreate a hello world function in Python." in called_chat_request.messages[0].content
    assert "Infer the programming language" in called_chat_request.messages[0].content # Default language hint
    assert called_chat_request.model_quality == "best_quality" # Default for coding
    assert called_chat_request.temperature == 0.3

async def test_code_agent_generate_code_with_language_and_context(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(
        instruction="Add a docstring to this function.",
        context="def my_func(a, b):\n    return a + b",
        language="python",
        model_quality="balanced" # User override
    )

    mock_llm_response_content = 'def my_func(a, b):\n    """Adds two numbers."""\n    return a + b'
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-ctx", model="model-ctx", provider="prov-ctx", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=mock_llm_response_content), finish_reason="stop")
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)

    response = await code_agent.generate_code(request)

    assert '"""Adds two numbers."""' in response.generated_code
    assert response.explanation is None

    called_chat_request: ChatCompletionRequest = mock_llm_aggregator.chat_completion.call_args[0][0]
    assert "Generate the code in python." in called_chat_request.messages[0].content
    assert "Here is some existing code or context to consider:\n```\ndef my_func(a, b):\n    return a + b\n```" in called_chat_request.messages[0].content
    assert called_chat_request.model_quality == "balanced" # User specified


async def test_code_agent_parse_code_and_explanation(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="Explain this code.")

    mock_llm_response_content = (
        "```python\nprint('Hello')\n```\n"
        f"{CODE_EXPLANATION_SEPARATOR}\n"
        "This is a simple Python script that prints Hello."
    )
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-exp", model="model-exp", provider="prov-exp", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=mock_llm_response_content), finish_reason="stop")
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)

    response = await code_agent.generate_code(request)

    # Check stripping of markdown fences
    assert response.generated_code == "print('Hello')"
    assert response.explanation == "This is a simple Python script that prints Hello."

async def test_code_agent_parse_code_only_no_explanation(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="Generate code.")
    mock_llm_response_content = "```javascript\nconsole.log('test');\n```" # Markdown fences
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-co", model="model-co", provider="prov-co", choices=[
             ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=mock_llm_response_content), finish_reason="stop")
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)
    response = await code_agent.generate_code(request)
    assert response.generated_code == "console.log('test');" # Stripped
    assert response.explanation is None

async def test_code_agent_parse_explanation_only_no_code(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="Explain concept.")
    mock_llm_response_content = f"{CODE_EXPLANATION_SEPARATOR}\nThis is just an explanation."
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-eo", model="model-eo", provider="prov-eo", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=mock_llm_response_content), finish_reason="stop")
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)
    response = await code_agent.generate_code(request)
    assert response.generated_code == "" # Code part is empty
    assert response.explanation == "This is just an explanation."

async def test_code_agent_llm_call_failure(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="Test failure.")

    mock_llm_aggregator.chat_completion = AsyncMock(side_effect=Exception("LLM simulated error"))

    response = await code_agent.generate_code(request)

    assert "Error generating code: LLM simulated error" in response.generated_code
    assert response.explanation == "An error occurred while processing your request."

async def test_code_agent_empty_llm_response(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="Test empty response.")
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-empty", model="model-empty", provider="prov-empty", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=""), finish_reason="stop") # Empty content
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)
    response = await code_agent.generate_code(request)
    assert response.generated_code == ""
    assert response.explanation is None

async def test_code_agent_markdown_fence_stripping_various(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="Generate code with fences.")

    test_cases = {
        "```python\ncode here\n```": "code here",
        "```\ncode here\n```": "code here", # No language
        "```typescript\nlet x = 10;\n```": "let x = 10;",
        "code without fences": "code without fences", # No change
        "```\n```": "", # Only fences
        "```python\n```": "", # Language but no code
    }

    for llm_content, expected_code in test_cases.items():
        mock_chat_response = ChatCompletionResponse(
            id="test-resp-fence", model="model-fence", provider="prov-fence", choices=[
                ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=llm_content), finish_reason="stop")
            ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
        )
        mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)
        response = await code_agent.generate_code(request)
        assert response.generated_code == expected_code
        assert response.explanation is None
```
