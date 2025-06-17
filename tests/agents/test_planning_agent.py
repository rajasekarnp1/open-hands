import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.planning_agent import PlanningAgent
from src.agents.code_agent import CodeAgent # For mocking
from src.agents.checkpoint import BaseCheckpointManager, InMemoryCheckpointManager # For mocking
from src.models import (
    PlanningAgentRequest,
    PlanningAgentResponse,
    CodeAgentRequest,
    CodeAgentResponse,
    HumanInputRequest
)
from src.agents.planning_models import Plan, Step
from src.agents.state import SessionState, Message
from src.core.aggregator import LLMAggregator

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_llm_aggregator() -> AsyncMock:
    return AsyncMock(spec=LLMAggregator)

@pytest.fixture
def mock_checkpoint_manager() -> MagicMock: # Using MagicMock for sync/async flexibility if Base methods aren't all async
    return MagicMock(spec=BaseCheckpointManager)

@pytest.fixture
def mock_code_agent() -> AsyncMock:
    return AsyncMock(spec=CodeAgent)

@pytest.fixture
def planning_agent(
    mock_llm_aggregator: AsyncMock,
    mock_checkpoint_manager: MagicMock,
    mock_code_agent: AsyncMock
) -> PlanningAgent:
    return PlanningAgent(
        llm_aggregator=mock_llm_aggregator,
        checkpoint_manager=mock_checkpoint_manager,
        code_agent=mock_code_agent
    )

# --- Test Cases ---

async def test_planning_agent_generate_new_plan_success(
    planning_agent: PlanningAgent,
    mock_llm_aggregator: AsyncMock,
    mock_checkpoint_manager: MagicMock,
    mock_code_agent: AsyncMock # To ensure it's NOT called during initial planning
):
    thread_id = "plan_thread_new"
    goal = "Create a snake game."
    request = PlanningAgentRequest(goal=goal, thread_id=thread_id, project_directory="/test_project")

    mock_checkpoint_manager.load_state = AsyncMock(return_value=None) # No existing state

    # Mock LLM response for plan generation
    llm_plan_json_str = json.dumps([
        {"description": "Step 1: Setup game window", "type": "coding"},
        {"description": "Step 2: Implement snake movement", "type": "coding"}
    ])
    mock_llm_aggregator.chat_completion = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=llm_plan_json_str))]
    ))

    # Mock CodeAgent response for the first step execution (assuming auto-start)
    mock_code_agent.generate_code = AsyncMock(return_value=CodeAgentResponse(
        generated_code="pass # Step 1 done", agent_status="completed"
    ))

    response = await planning_agent.generate_plan(request)

    assert response.agent_status == "executing_step" # Because it auto-executes first step
    assert response.plan is not None
    assert response.plan.goal == goal
    assert len(response.plan.steps) == 2
    assert response.plan.steps[0].description == "Step 1: Setup game window"
    assert response.plan.steps[0].agent_to_use == "CodeAgent"
    assert response.plan.steps[0].status == "completed" # First step executed
    assert response.plan.steps[1].description == "Step 2: Implement snake movement"
    assert response.plan.steps[1].status == "pending" # Second step should be next

    assert mock_checkpoint_manager.load_state.call_count == 1
    # Save is called: 1. After new state init, 2. After plan generation, 3. After first step in_progress, 4. After first step completed
    assert mock_checkpoint_manager.save_state.call_count >= 3 # Could be more if execute_next_step calls it multiple times

    mock_code_agent.generate_code.assert_called_once() # First step was called

async def test_planning_agent_retrieve_existing_plan(
    planning_agent: PlanningAgent,
    mock_checkpoint_manager: MagicMock
):
    thread_id = "plan_thread_existing"
    goal = "An existing goal."
    existing_plan = Plan(goal=goal, steps=[Step(description="Existing step 1", status="pending")], plan_status="ready")
    existing_state = SessionState(thread_id=thread_id, current_plan=existing_plan)
    mock_checkpoint_manager.load_state = AsyncMock(return_value=existing_state)

    request = PlanningAgentRequest(goal="A new goal, but should use existing plan", thread_id=thread_id)

    response = await planning_agent.generate_plan(request) # generate_plan should detect and return existing

    assert response.agent_status == "plan_retrieved_existing"
    assert response.plan == existing_plan
    mock_checkpoint_manager.load_state.assert_called_once_with(thread_id)

async def test_planning_agent_execute_step_code_agent_hitl_relay(
    planning_agent: PlanningAgent,
    mock_checkpoint_manager: MagicMock,
    mock_code_agent: AsyncMock
):
    thread_id = "plan_thread_hitl"
    goal = "Goal leading to HITL"
    project_dir = "/test_hitl_project"

    # Plan where first step will trigger HITL via CodeAgent
    plan = Plan(
        goal=goal,
        steps=[Step(description="Do something requiring human input", agent_to_use="CodeAgent", status="pending")],
        plan_status="ready" # Plan is ready, about to execute first step
    )
    initial_state = SessionState(
        thread_id=thread_id,
        current_plan=plan,
        original_request_info={"goal": goal, "project_directory": project_dir}
    )
    # This state is what execute_next_step would receive or what generate_plan would set up before calling execute_next_step

    # Mock CodeAgent to return "requires_human_input"
    human_input_req_details = HumanInputRequest(tool_call_id="tc_hitl_01", question_for_human="Approve this step?")
    mock_code_agent.generate_code = AsyncMock(return_value=CodeAgentResponse(
        agent_status="requires_human_input",
        human_input_request=human_input_req_details
    ))

    # We are directly testing execute_next_step here by calling it after generate_plan has setup the plan.
    # In a real flow, generate_plan would call execute_next_step.
    # Here, we simulate that the plan is ready and we are about to execute its first step.
    planning_request = PlanningAgentRequest(goal=goal, thread_id=thread_id, project_directory=project_dir)
    response = await planning_agent.execute_next_step(planning_request, initial_state)

    assert response.agent_status == "paused_for_human_input"
    assert response.plan is not None
    assert response.plan.plan_status == "paused_for_human_input"
    assert response.plan.steps[0].status == "paused_for_human_input"
    assert response.human_input_request == human_input_req_details

    mock_code_agent.generate_code.assert_called_once()
    # Check CodeAgentRequest passed
    code_agent_call_args = mock_code_agent.generate_code.call_args[0][0]
    assert isinstance(code_agent_call_args, CodeAgentRequest)
    assert code_agent_call_args.instruction == "Do something requiring human input"
    assert code_agent_call_args.thread_id == thread_id
    assert code_agent_call_args.project_directory == project_dir

    # Check save_state calls: 1 for step in_progress, 1 for plan paused by HITL
    assert mock_checkpoint_manager.save_state.call_count == 2


async def test_planning_agent_plan_completion(
    planning_agent: PlanningAgent,
    mock_checkpoint_manager: MagicMock,
    mock_code_agent: AsyncMock
):
    thread_id = "plan_thread_completion"
    goal = "A two-step goal"
    project_dir = "/test_completion_project"

    # Initial request to generate the plan
    initial_request = PlanningAgentRequest(goal=goal, thread_id=thread_id, project_directory=project_dir)

    # Mock LLM to generate a two-step plan
    llm_plan_json_str = json.dumps([
        {"description": "Step 1: Code task", "type": "coding"},
        {"description": "Step 2: Another code task", "type": "coding"}
    ])
    planning_agent.llm_aggregator.chat_completion = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=llm_plan_json_str))]
    ))

    # Mock CodeAgent to complete steps successfully
    mock_code_agent.generate_code.side_effect = [
        CodeAgentResponse(generated_code="Step 1 done", agent_status="completed", model_used="m1"),
        CodeAgentResponse(generated_code="Step 2 done", agent_status="completed", model_used="m2")
    ]

    response = await planning_agent.generate_plan(initial_request)

    assert response.agent_status == "plan_completed"
    assert response.plan is not None
    assert response.plan.plan_status == "completed"
    assert len(response.plan.steps) == 2
    assert response.plan.steps[0].status == "completed"
    assert response.plan.steps[0].result == "Step 1 done"
    assert response.plan.steps[1].status == "completed"
    assert response.plan.steps[1].result == "Step 2 done"

    assert mock_code_agent.generate_code.call_count == 2
    # Check that save_state was called multiple times (init, plan_ready, step1_inprogress, step1_done, step2_inprogress, step2_done, plan_completed)
    assert mock_checkpoint_manager.save_state.call_count >= 7

async def test_planning_agent_step_failure(
    planning_agent: PlanningAgent,
    mock_checkpoint_manager: MagicMock,
    mock_code_agent: AsyncMock
):
    thread_id = "plan_thread_failure"
    goal = "Goal with a failing step"
    project_dir = "/test_failure_project"
    initial_request = PlanningAgentRequest(goal=goal, thread_id=thread_id, project_directory=project_dir)

    llm_plan_json_str = json.dumps([{"description": "Failing code task", "type": "coding"}])
    planning_agent.llm_aggregator.chat_completion = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content=llm_plan_json_str))]
    ))

    mock_code_agent.generate_code.return_value = CodeAgentResponse(
        agent_status="error", error_details="CodeAgent failed spectacularly"
    )

    response = await planning_agent.generate_plan(initial_request)

    assert response.agent_status == "failed" # Plan status reflects step failure
    assert response.plan is not None
    assert response.plan.plan_status == "failed"
    assert response.plan.steps[0].status == "failed"
    assert response.plan.steps[0].error_details == "CodeAgent failed spectacularly"

    mock_code_agent.generate_code.assert_called_once()
    assert mock_checkpoint_manager.save_state.call_count >= 3 # init, plan_ready, step_inprogress, step_failed/plan_failed

```
