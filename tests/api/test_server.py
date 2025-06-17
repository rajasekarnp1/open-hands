import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

# We need to make sure the FastAPI app instance is created correctly for testing
# This might involve importing the 'app' from server.py or a factory function.
# For this test, let's assume 'app' can be imported from src.api.server
# and that global instances (like code_agent_instance) are patched during tests.

from src.api.server import app # main FastAPI app
from src.models import CodeAgentResponse, HumanInputRequest, SessionState, Message

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture
def mock_code_agent_generate_code():
    # This mock will be used to patch 'code_agent_instance.generate_code'
    # 'code_agent_instance' is a global in src.api.server
    return AsyncMock()

@pytest.fixture
def mock_checkpoint_manager_load_state():
    return AsyncMock()

@pytest.fixture
def mock_checkpoint_manager_save_state():
    return AsyncMock()

@pytest.fixture
def mock_planning_agent_generate_plan():
    return AsyncMock()

@pytest.fixture(autouse=True)
def patch_agent_and_checkpoint_manager(
    mock_code_agent_generate_code: AsyncMock,
    mock_planning_agent_generate_plan: AsyncMock, # Added
    mock_checkpoint_manager_load_state: AsyncMock,
    mock_checkpoint_manager_save_state: AsyncMock
):
    # Patch the global instances within src.api.server directly for the duration of tests
    with patch('src.api.server.code_agent_instance') as mock_code_agent_global_ref, \
         patch('src.api.server.planning_agent_instance') as mock_planning_agent_global_ref, \
         patch('src.api.server.checkpoint_manager_instance') as mock_cm_global_ref:

        if mock_code_agent_global_ref:
             mock_code_agent_global_ref.generate_code = mock_code_agent_generate_code

        if mock_planning_agent_global_ref:
            mock_planning_agent_global_ref.generate_plan = mock_planning_agent_generate_plan
            # Ensure the planning_agent_instance has a code_agent attribute for when it's called by planning_agent
            # This can be a simple mock or the actual mocked code_agent_instance if needed for deeper tests
            if hasattr(mock_planning_agent_global_ref, 'code_agent'):
                 mock_planning_agent_global_ref.code_agent = mock_code_agent_global_ref


        if mock_cm_global_ref:
            mock_cm_global_ref.load_state = mock_checkpoint_manager_load_state
            mock_cm_global_ref.save_state = mock_checkpoint_manager_save_state

        yield # Test runs with patches

# --- Tests for /v1/agents/code/invoke ---

def test_invoke_agent_simple_request(client: TestClient, mock_code_agent_generate_code: AsyncMock):
    # Mock the agent's response for this specific call
    mock_code_agent_generate_code.return_value = CodeAgentResponse(
        generated_code="print('Hello')",
        agent_status="completed"
    )

    response = client.post("/v1/agents/code/invoke", json={
        "instruction": "Say hello"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["generated_code"] == "print('Hello')"
    assert data["agent_status"] == "completed"
    mock_code_agent_generate_code.assert_called_once()
    # Assert that project_directory and thread_id are None or not present in call if not sent

def test_invoke_agent_with_thread_id(client: TestClient, mock_code_agent_generate_code: AsyncMock):
    thread_id = "my_test_thread"
    mock_code_agent_generate_code.return_value = CodeAgentResponse(
        generated_code="Updated code",
        agent_status="completed"
    )

    response = client.post("/v1/agents/code/invoke", json={
        "instruction": "Continue task",
        "thread_id": thread_id
    })

    assert response.status_code == 200
    mock_code_agent_generate_code.assert_called_once()
    # Check that thread_id was passed to the agent
    called_request = mock_code_agent_generate_code.call_args[0][0]
    assert called_request.thread_id == thread_id

def test_invoke_agent_returns_hitl_response(client: TestClient, mock_code_agent_generate_code: AsyncMock):
    hitl_request_details = HumanInputRequest(
        tool_call_id="hitl_abc",
        question_for_human="Should I proceed?"
    )
    mock_code_agent_generate_code.return_value = CodeAgentResponse(
        agent_status="requires_human_input",
        human_input_request=hitl_request_details,
        explanation="Paused for your input."
    )

    response = client.post("/v1/agents/code/invoke", json={
        "instruction": "Do something that needs human input",
        "thread_id": "hitl_thread_test"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["agent_status"] == "requires_human_input"
    assert data["human_input_request"]["tool_call_id"] == "hitl_abc"
    assert data["human_input_request"]["question_for_human"] == "Should I proceed?"
    assert data["explanation"] == "Paused for your input."

def test_invoke_agent_with_invalid_project_directory(client: TestClient):
    # This tests the validation added in the API endpoint itself
    response = client.post("/v1/agents/code/invoke", json={
        "instruction": "Test",
        "project_directory": "/path/that/does/not/exist/for/sure"
    })
    assert response.status_code == 400 # Bad request due to invalid path
    assert "does not exist" in response.json()["detail"]

    # Create a file to test the "is not a directory" case
    with open("temp_file_for_test.txt", "w") as f:
        f.write("test")

    response_file_path = client.post("/v1/agents/code/invoke", json={
        "instruction": "Test file path",
        "project_directory": "temp_file_for_test.txt"
    })
    assert response_file_path.status_code == 400
    assert "is not a directory" in response_file_path.json()["detail"]

    import os
    os.remove("temp_file_for_test.txt")


# --- Tests for /v1/agents/resume ---

def test_resume_agent_success(
    client: TestClient,
    mock_code_agent_generate_code: AsyncMock,
    mock_checkpoint_manager_load_state: AsyncMock,
    mock_checkpoint_manager_save_state: AsyncMock # To assert it's called
):
    thread_id = "resume_thread_ok"
    tool_call_id = "hitl_call_id_for_resume"
    human_response_text = "Yes, proceed."

    # Mock loading state
    mock_session_state = SessionState(
        thread_id=thread_id,
        conversation_history=[Message(role="user", content="Initial instruction")],
        original_request_info={"project_directory": "/test", "language": "python"} # Mocked original info
    )
    mock_checkpoint_manager_load_state.return_value = mock_session_state

    # Mock the agent's response after resumption
    mock_code_agent_generate_code.return_value = CodeAgentResponse(
        generated_code="Resumed and completed.",
        agent_status="completed"
    )

    response = client.post("/v1/agents/resume", json={
        "thread_id": thread_id,
        "tool_call_id": tool_call_id,
        "human_response": human_response_text
    })

    assert response.status_code == 200
    data = response.json()
    assert data["agent_status"] == "completed"
    assert data["generated_code"] == "Resumed and completed."

    mock_checkpoint_manager_load_state.assert_called_once_with(thread_id)

    # Check that save_state was called after adding human response to history
    # This save happens in the /resume endpoint before calling generate_code
    mock_checkpoint_manager_save_state.assert_any_call(thread_id, mock_session_state)

    # Verify the history now includes the human's response as a tool message
    assert len(mock_session_state.conversation_history) == 2
    last_message = mock_session_state.conversation_history[-1]
    assert last_message.role == "tool"
    assert last_message.name == "ask_human_for_input" # Name of the tool that was 'called'
    assert last_message.content == human_response_text
    assert last_message.tool_call_id == tool_call_id

    # Check that planning_agent.generate_plan was called for resumption
    mock_planning_agent_generate_plan.assert_called_once()
    resumed_call_request_arg = mock_planning_agent_generate_plan.call_args[0][0]
    from src.models import PlanningAgentRequest # For type check
    assert isinstance(resumed_call_request_arg, PlanningAgentRequest)
    assert resumed_call_request_arg.thread_id == thread_id
    assert resumed_call_request_arg.goal == "Initial instruction" # Goal from plan/original_request_info
    assert resumed_call_request_arg.project_directory == "/test"


def test_resume_agent_thread_not_found(client: TestClient, mock_checkpoint_manager_load_state: AsyncMock):
    mock_checkpoint_manager_load_state.return_value = None # Simulate no state found

    response = client.post("/v1/agents/resume", json={
        "thread_id": "non_existent_thread_for_resume",
        "tool_call_id": "any_id",
        "human_response": "doesn't matter"
    })

    assert response.status_code == 404
    assert "No session state found" in response.json()["detail"]

def test_resume_agent_missing_original_request_info(
    client: TestClient,
    mock_checkpoint_manager_load_state: AsyncMock
):
    thread_id = "resume_thread_no_orig_info"
    # State is loaded, but original_request_info is missing (shouldn't happen with current agent logic)
    mock_session_state_no_orig = SessionState(thread_id=thread_id, conversation_history=[])
    mock_session_state_no_orig.original_request_info = None # Explicitly set to None
    mock_checkpoint_manager_load_state.return_value = mock_session_state_no_orig

    response = client.post("/v1/agents/resume", json={
        "thread_id": thread_id,
        "tool_call_id": "any_id",
        "human_response": "Test"
    })
    assert response.status_code == 500 # Internal error because original_request_info is vital
    assert "Missing original request info" in response.json()["detail"]


# --- Tests for /v1/agents/plan/invoke ---

def test_invoke_planning_agent_success(client: TestClient, mock_planning_agent_generate_plan: AsyncMock):
    from src.agents.planning_models import Plan # For constructing mock response
    mock_plan = Plan(goal="Test goal", steps=[], plan_status="ready")
    mock_planning_agent_generate_plan.return_value = PlanningAgentResponse(
        plan=mock_plan,
        agent_status="plan_generated",
        thread_id="plan_thread_1"
    )

    response = client.post("/v1/agents/plan/invoke", json={
        "goal": "Test goal",
        "project_directory": None, # Assuming project_directory is validated and resolved by endpoint if provided
        "thread_id": "plan_thread_1"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["agent_status"] == "plan_generated"
    assert data["plan"]["goal"] == "Test goal"
    assert data["thread_id"] == "plan_thread_1"
    mock_planning_agent_generate_plan.assert_called_once()
    called_request = mock_planning_agent_generate_plan.call_args[0][0]
    assert called_request.goal == "Test goal"
    assert called_request.thread_id == "plan_thread_1"

def test_invoke_planning_agent_hitl_relay(client: TestClient, mock_planning_agent_generate_plan: AsyncMock):
    hitl_details = HumanInputRequest(tool_call_id="sub_agent_hitl_id", question_for_human="Approve sub-step?")
    mock_planning_agent_generate_plan.return_value = PlanningAgentResponse(
        agent_status="paused_for_human_input",
        human_input_request=hitl_details,
        thread_id="plan_hitl_thread",
        plan=MagicMock(spec=Plan) # Plan object would exist
    )

    response = client.post("/v1/agents/plan/invoke", json={
        "goal": "Goal that triggers HITL in sub-agent",
        "thread_id": "plan_hitl_thread"
    })

    assert response.status_code == 200
    data = response.json()
    assert data["agent_status"] == "paused_for_human_input"
    assert data["human_input_request"]["tool_call_id"] == "sub_agent_hitl_id"
    assert data["human_input_request"]["question_for_human"] == "Approve sub-step?"
    assert data["thread_id"] == "plan_hitl_thread"

def test_invoke_planning_agent_invalid_project_dir(client: TestClient):
    response = client.post("/v1/agents/plan/invoke", json={
        "goal": "Test with bad project dir",
        "project_directory": "/path/that/definitely/does/not/exist"
    })
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]
```
