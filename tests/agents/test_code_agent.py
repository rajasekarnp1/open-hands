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


# --- Mocks for filesystem_tools ---
@pytest.fixture
def mock_fs_tools():
    with patch('src.agents.code_agent.filesystem_tools') as mock_fs:
        mock_fs.read_file = AsyncMock(return_value="File content from mock_read_file")
        mock_fs.write_file = AsyncMock(return_value="File 'mock_file.txt' written successfully.")
        mock_fs.list_files = AsyncMock(return_value=["mock_file.txt", "mock_subdir/"])
        yield mock_fs

# --- Test Cases (Original and New) ---

async def test_code_agent_generate_code_simple_instruction(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
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
    call_args = mock_llm_aggregator.chat_completion.call_args
    called_chat_request: ChatCompletionRequest = call_args[0][0]

    assert "Instruction:\nCreate a hello world function in Python." in called_chat_request.messages[0].content # User prompt check
    # System prompt check - should NOT include tool instructions if project_directory is None
    system_prompt_content = ""
    for msg in called_chat_request.messages: # Assuming system prompt is one of the messages
        if msg.role == "system":
            system_prompt_content = msg.content
            break
    assert "Available Tools:" not in system_prompt_content
    assert "Infer the programming language" in called_chat_request.messages[1].content # Initial user prompt content
    assert called_chat_request.model_quality == "best_quality" # Default for coding
    assert called_chat_request.temperature == 0.3
    mock_fs_tools.read_file.assert_not_called() # No tools should be called

async def test_code_agent_generate_code_with_language_and_context(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
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
    assert "Here is some existing code or context to consider:\n```\ndef my_func(a, b):\n    return a + b\n```" in called_chat_request.messages[1].content # User prompt content
    assert called_chat_request.model_quality == "balanced" # User specified
    mock_fs_tools.read_file.assert_not_called()


async def test_code_agent_parse_code_and_explanation(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
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
    mock_fs_tools.read_file.assert_not_called()

async def test_code_agent_parse_code_only_no_explanation(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
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
    mock_fs_tools.read_file.assert_not_called()

async def test_code_agent_parse_explanation_only_no_code(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
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
    mock_fs_tools.read_file.assert_not_called()

async def test_code_agent_llm_call_failure(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    request = CodeAgentRequest(instruction="Test failure.")

    mock_llm_aggregator.chat_completion = AsyncMock(side_effect=Exception("LLM simulated error"))

    response = await code_agent.generate_code(request)

    assert "Error: Could not get response from LLM. LLM simulated error" in response.explanation # Error is now in explanation
    assert response.generated_code == "" # Code should be empty on critical LLM error
    mock_fs_tools.read_file.assert_not_called()


async def test_code_agent_empty_llm_response(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    request = CodeAgentRequest(instruction="Test empty response.")
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-empty", model="model-empty", provider="prov-empty", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=""), finish_reason="stop") # Empty content
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)
    response = await code_agent.generate_code(request)
    assert response.generated_code == ""
    assert response.explanation == "Error: LLM response was empty." # Error is now in explanation
    mock_fs_tools.read_file.assert_not_called()

async def test_code_agent_markdown_fence_stripping_various(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
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
    mock_fs_tools.read_file.assert_not_called()

# --- New Tests for Tool Usage ---

async def test_code_agent_system_prompt_with_project_dir(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock):
    request = CodeAgentRequest(instruction="List files.", project_directory="/test/project")

    # Mock LLM response to return simple code, no tool call needed for this specific test
    mock_llm_response_content = "Final answer, no tools needed for this."
    mock_chat_response = ChatCompletionResponse(
        id="test-resp-sys", model="model-sys", provider="prov-sys", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=mock_llm_response_content), finish_reason="stop")
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_chat_response)

    await code_agent.generate_code(request) # We don't care about the response here, just the call

    called_chat_request: ChatCompletionRequest = mock_llm_aggregator.chat_completion.call_args[0][0]
    system_prompt_content = ""
    for msg in called_chat_request.messages:
        if msg.role == "system":
            system_prompt_content = msg.content
            break
    assert "Available Tools:" in system_prompt_content
    assert "read_file(filepath: str)" in system_prompt_content
    assert "write_file(filepath: str, content: str)" in system_prompt_content
    assert "list_files(directory_path: str = \"\")" in system_prompt_content

async def test_code_agent_tool_call_read_file(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    project_dir = "/test/project"
    request = CodeAgentRequest(instruction="Read my_file.txt", project_directory=project_dir)

    # LLM first responds with a tool call
    llm_response_tool_call = '{"tool_name": "read_file", "parameters": {"filepath": "my_file.txt"}}'
    mock_chat_response_1 = ChatCompletionResponse(
        id="resp1", model="m1", provider="p1", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=llm_response_tool_call), finish_reason="tool_calls")
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )

    # LLM then responds with final answer after getting tool output
    final_code = "print('File content was: File content from mock_read_file')"
    mock_chat_response_2 = ChatCompletionResponse(
        id="resp2", model="m1", provider="p1", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=final_code), finish_reason="stop")
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )
    mock_llm_aggregator.chat_completion.side_effect = [mock_chat_response_1, mock_chat_response_2]
    mock_fs_tools.read_file.return_value = "File content from mock_read_file"

    response = await code_agent.generate_code(request)

    mock_fs_tools.read_file.assert_called_once_with(project_dir, "my_file.txt")
    assert mock_llm_aggregator.chat_completion.call_count == 2

    # Check history passed to second LLM call
    history_for_second_call = mock_llm_aggregator.chat_completion.call_args_list[1][0][0].messages
    assert history_for_second_call[-1].role == "tool"
    assert history_for_second_call[-1].content == "File content from mock_read_file"
    assert history_for_second_call[-2].role == "assistant" # LLM's previous response (tool call)
    assert history_for_second_call[-2].content == llm_response_tool_call

    assert response.generated_code == final_code

async def test_code_agent_tool_call_write_file(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    project_dir = "/test/project"
    request = CodeAgentRequest(instruction="Write 'hello' to output.txt", project_directory=project_dir)

    llm_response_tool_call = '{"tool_name": "write_file", "parameters": {"filepath": "output.txt", "content": "hello"}}'
    mock_chat_response_1 = ChatCompletionResponse(id="r1",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=llm_response_tool_call),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    final_explanation = "Successfully wrote to output.txt"
    mock_chat_response_2 = ChatCompletionResponse(id="r2",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=f"{CODE_EXPLANATION_SEPARATOR}{final_explanation}"),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    mock_llm_aggregator.chat_completion.side_effect = [mock_chat_response_1, mock_chat_response_2]
    mock_fs_tools.write_file.return_value = "File 'output.txt' written successfully."

    response = await code_agent.generate_code(request)

    mock_fs_tools.write_file.assert_called_once_with(project_dir, "output.txt", "hello")
    assert response.explanation == final_explanation
    assert response.generated_code == "" # No code in final response, just explanation

async def test_code_agent_tool_call_list_files(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    project_dir = "/test/project"
    request = CodeAgentRequest(instruction="List files in src/", project_directory=project_dir)

    llm_response_tool_call = '{"tool_name": "list_files", "parameters": {"directory_path": "src/"}}'
    mock_chat_response_1 = ChatCompletionResponse(id="r1",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=llm_response_tool_call),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    final_explanation = "Files in src/ are:\nmock_file.txt\nmock_subdir/"
    mock_chat_response_2 = ChatCompletionResponse(id="r2",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=f"{CODE_EXPLANATION_SEPARATOR}{final_explanation}"),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    mock_llm_aggregator.chat_completion.side_effect = [mock_chat_response_1, mock_chat_response_2]
    mock_fs_tools.list_files.return_value = ["mock_file.txt", "mock_subdir/"] # This is what the tool returns

    response = await code_agent.generate_code(request)

    mock_fs_tools.list_files.assert_called_once_with(project_dir, "src/")
    history_for_second_call = mock_llm_aggregator.chat_completion.call_args_list[1][0][0].messages
    assert history_for_second_call[-1].content == "mock_file.txt\nmock_subdir/" # Check tool output stringification
    assert response.explanation == final_explanation

async def test_code_agent_tool_error_feedback(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    project_dir = "/test/project"
    request = CodeAgentRequest(instruction="Read non_existent.txt", project_directory=project_dir)

    llm_response_tool_call = '{"tool_name": "read_file", "parameters": {"filepath": "non_existent.txt"}}'
    mock_chat_response_1 = ChatCompletionResponse(id="r1",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=llm_response_tool_call),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    final_explanation = "Could not read the file as it does not exist."
    mock_chat_response_2 = ChatCompletionResponse(id="r2",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=f"{CODE_EXPLANATION_SEPARATOR}{final_explanation}"),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    mock_llm_aggregator.chat_completion.side_effect = [mock_chat_response_1, mock_chat_response_2]
    mock_fs_tools.read_file.return_value = "Error: File not found at path 'non_existent.txt'." # Tool returns error

    response = await code_agent.generate_code(request)

    mock_fs_tools.read_file.assert_called_once_with(project_dir, "non_existent.txt")
    history_for_second_call = mock_llm_aggregator.chat_completion.call_args_list[1][0][0].messages
    assert "Error: File not found" in history_for_second_call[-1].content # Tool error passed to LLM
    assert response.explanation == final_explanation

async def test_code_agent_unknown_tool_call(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    project_dir = "/test/project"
    request = CodeAgentRequest(instruction="Use a fake tool", project_directory=project_dir)

    llm_response_tool_call = '{"tool_name": "fake_tool", "parameters": {}}'
    mock_chat_response_1 = ChatCompletionResponse(id="r1",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=llm_response_tool_call),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    final_explanation = "Okay, I will not use fake_tool."
    mock_chat_response_2 = ChatCompletionResponse(id="r2",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=f"{CODE_EXPLANATION_SEPARATOR}{final_explanation}"),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    mock_llm_aggregator.chat_completion.side_effect = [mock_chat_response_1, mock_chat_response_2]

    response = await code_agent.generate_code(request)

    assert mock_llm_aggregator.chat_completion.call_count == 2
    history_for_second_call = mock_llm_aggregator.chat_completion.call_args_list[1][0][0].messages
    assert "Error: Unknown tool 'fake_tool'." in history_for_second_call[-1].content
    assert response.explanation == final_explanation
    mock_fs_tools.read_file.assert_not_called()
    mock_fs_tools.write_file.assert_not_called()
    mock_fs_tools.list_files.assert_not_called()


async def test_code_agent_max_iterations_exceeded(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    project_dir = "/test/project"
    request = CodeAgentRequest(instruction="Loop forever", project_directory=project_dir)

    # LLM always responds with a tool call
    llm_response_tool_call = '{"tool_name": "list_files", "parameters": {}}'
    mock_chat_response_loop = ChatCompletionResponse(id="r_loop",model="m_loop",provider="p_loop",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=llm_response_tool_call),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    mock_llm_aggregator.chat_completion.return_value = mock_chat_response_loop # Always return this
    mock_fs_tools.list_files.return_value = ["some_file.txt"] # Tool always succeeds

    response = await code_agent.generate_code(request)

    from src.agents.code_agent import MAX_TOOL_ITERATIONS # Get the constant
    assert mock_llm_aggregator.chat_completion.call_count == MAX_TOOL_ITERATIONS
    assert mock_fs_tools.list_files.call_count == MAX_TOOL_ITERATIONS # Called in each iteration
    assert "Error: Agent exceeded maximum tool iterations." in response.generated_code

async def test_code_agent_tool_call_without_project_directory(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    request = CodeAgentRequest(instruction="Read a file without project context") # No project_directory

    # LLM tries to call a tool (even though it shouldn't have been told about them)
    llm_response_tool_call = '{"tool_name": "read_file", "parameters": {"filepath": "a_file.txt"}}'
    mock_chat_response_1 = ChatCompletionResponse(id="r1",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=llm_response_tool_call),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    final_explanation = "I understand I cannot use tools without a project directory."
    mock_chat_response_2 = ChatCompletionResponse(id="r2",model="m",provider="p",choices=[ChatCompletionChoice(message=ChatMessage(role="assistant",content=f"{CODE_EXPLANATION_SEPARATOR}{final_explanation}"),index=0)],usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2))

    mock_llm_aggregator.chat_completion.side_effect = [mock_chat_response_1, mock_chat_response_2]

    response = await code_agent.generate_code(request)

    mock_fs_tools.read_file.assert_not_called() # Filesystem tool should not be called

    history_for_second_call = mock_llm_aggregator.chat_completion.call_args_list[1][0][0].messages
    assert "Error: Project directory not specified. Cannot use file system tools." in history_for_second_call[-1].content
    assert response.explanation == final_explanation

async def test_code_agent_malformed_tool_json(code_agent: CodeAgent, mock_llm_aggregator: AsyncMock, mock_fs_tools: MagicMock):
    project_dir = "/test/project"
    request = CodeAgentRequest(instruction="Test malformed JSON", project_directory=project_dir)

    # LLM responds with malformed JSON for tool call
    llm_response_malformed_json = 'This is my reasoning. ```json {"tool_name": "read_file", "parameters": {"filepath": "my_file.txt" ``` <- oops, cut off'
    mock_chat_response_1 = ChatCompletionResponse(
        id="resp1_malformed", model="m1", provider="p1", choices=[
            ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=llm_response_malformed_json), finish_reason="stop") # finish_reason might be stop if LLM thinks it's done
        ], usage=ChatCompletionUsage(prompt_tokens=1,completion_tokens=1,total_tokens=2)
    )

    # In this case, _parse_llm_for_tool_call should return None, so it's treated as a final response.
    mock_llm_aggregator.chat_completion.return_value = mock_chat_response_1

    response = await code_agent.generate_code(request)

    mock_fs_tools.read_file.assert_not_called() # No tool call should be successfully parsed or executed
    assert mock_llm_aggregator.chat_completion.call_count == 1 # Only one call, as no valid tool was found

    # The response will be the LLM's attempt, parsed as code/explanation
    expected_code, expected_explanation = code_agent._parse_final_response(llm_response_malformed_json)
    assert response.generated_code == expected_code
    assert response.explanation == expected_explanation
```
