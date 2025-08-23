"""
Planning Agent implementation.
Decomposes a high-level goal into a sequence of actionable steps and can orchestrate CodeAgent.
"""
import json
import logging
import re # For parsing LLM plan string
from typing import Optional, List, Dict, Any

from ..models import (
    PlanningAgentRequest,
    PlanningAgentResponse,
    ChatCompletionRequest,
    ChatCompletionResponse, # For type hint
    CodeAgentRequest,      # For calling CodeAgent
    CodeAgentResponse,     # For handling CodeAgent's response
    HumanInputRequest      # For relaying HITL if CodeAgent pauses
)
from .planning_models import Plan, Step
from .state import SessionState, Message
from .checkpoint import BaseCheckpointManager
from ..core.aggregator import LLMAggregator
from ..agents.code_agent import CodeAgent # Import CodeAgent

logger = logging.getLogger(__name__)

class PlanningAgent:
    """
    Agent responsible for decomposing a goal into a plan and orchestrating execution
    of steps, starting with CodeAgent steps.
    """

    def __init__(self,
                 llm_aggregator: LLMAggregator,
                 checkpoint_manager: BaseCheckpointManager,
                 code_agent: CodeAgent): # Added CodeAgent dependency
        self.llm_aggregator = llm_aggregator
        self.checkpoint_manager = checkpoint_manager
        self.code_agent = code_agent # Store CodeAgent instance

    def _construct_system_prompt(self) -> str:
        return (
            "You are an expert project planner. Your task is to decompose a given goal into a "
            "sequence of actionable steps. Each step should be a clear, concise action that "
            "contributes to achieving the overall goal.\n"
            "Consider the context of a software development project where tasks might involve "
            "coding, research, file system operations, or requiring human decisions.\n"
            "For each step, provide a 'description' of the action to be taken. "
            "Also, suggest a 'type' for the step, which can be one of: "
            "'coding' (if it primarily involves writing or modifying code), "
            "'filesystem' (if it's a direct file operation like read, write, list - these will be handled by CodeAgent tools), "
            "'research' (if it involves information gathering), "
            "'human_input' (if it requires a decision or input from a human - this will be handled by CodeAgent's ask_human_for_input tool), "
            "or 'general' (for other types of tasks or if unsure).\n"
            "If the step is 'coding' or 'filesystem', the description should be a clear instruction for a coding agent. "
            "If 'human_input', the description should be the question to ask the human.\n"
            "Output the plan as a JSON list of objects, where each object represents a step "
            "and has 'description' and 'type' fields. Ensure the JSON is well-formed."
            "\nExample JSON output structure: \n"
            "[\n"
            "  {\"description\": \"Write a Python function to parse CSV data from 'input.csv'\", \"type\": \"coding\"},\n"
            "  {\"description\": \"Read the 'requirements.txt' file to understand dependencies\", \"type\": \"filesystem\"},\n"
            "  {\"description\": \"Decide if backward compatibility is needed for version X\", \"type\": \"human_input\"}\n"
            "]"
        )

    def _construct_user_prompt_for_planning(self, request: PlanningAgentRequest) -> str:
        prompt_parts = [f"Goal: {request.goal}"]
        if request.project_directory:
            prompt_parts.append(
                f"\nProject Context: The user is working within a project located at "
                f"'{request.project_directory}'. Consider this when creating steps, "
                f"especially for file-related tasks. Assume file paths in steps will be relative to this root."
            )
        return "\n".join(prompt_parts)

    async def _get_llm_plan_response(
        self,
        conversation_history: List[Message],
        request: PlanningAgentRequest
    ) -> Optional[str]:
        messages_for_llm = [msg.model_dump(exclude_none=True) for msg in conversation_history]
        model_quality = "best_quality"
        chat_request = ChatCompletionRequest(
            messages=messages_for_llm, # type: ignore
            model="auto",
            model_quality=model_quality,
            temperature=0.2,
        )
        try:
            logger.debug(f"PlanningAgent sending request to LLMAggregator: {messages_for_llm}")
            llm_response = await self.llm_aggregator.chat_completion(chat_request)
            if llm_response.choices and llm_response.choices[0].message and \
               isinstance(llm_response.choices[0].message.content, str):
                content = llm_response.choices[0].message.content.strip()
                logger.debug(f"PlanningAgent received raw plan response: {content}")
                return content
            logger.warning("Planning LLM response was empty or malformed.")
            return None
        except Exception as e:
            logger.error(f"Error calling LLM for planning: {e}", exc_info=True)
            return None

    def _parse_llm_plan_to_steps(self, llm_response_content: str) -> List[Step]:
        steps = []
        try:
            match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```|(\[[\s\S]*?\])$", llm_response_content, re.DOTALL | re.MULTILINE)
            json_str = ""
            if match: json_str = match.group(1) if match.group(1) else match.group(2)
            if not json_str: json_str = llm_response_content

            parsed_steps_json = json.loads(json_str)
            if not isinstance(parsed_steps_json, list):
                logger.warning(f"LLM plan response is not a list: {parsed_steps_json}")
                return []

            for step_data in parsed_steps_json:
                if not isinstance(step_data, dict) or "description" not in step_data or "type" not in step_data:
                    logger.warning(f"Skipping malformed step data: {step_data}")
                    continue

                agent_to_use: Optional[str] = None
                tool_to_use: Optional[str] = None
                llm_step_type = step_data.get("type", "general").lower()

                if llm_step_type == "coding" or llm_step_type == "filesystem":
                    agent_to_use = "CodeAgent"
                elif llm_step_type == "human_input":
                    agent_to_use = "CodeAgent" # CodeAgent handles ask_human_for_input tool
                    tool_to_use = "ask_human_for_input"
                elif llm_step_type == "research": # Placeholder
                    agent_to_use = "ResearchAgent"

                steps.append(Step(
                    description=step_data["description"],
                    agent_to_use=agent_to_use,
                    tool_to_use=tool_to_use,
                ))
            return steps
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON plan from LLM: {e}\nRaw: {llm_response_content}")
            return []
        except Exception as e: # pylint: disable=broad-except
            logger.error(f"Error parsing LLM plan: {e}\nRaw: {llm_response_content}", exc_info=True)
            return []

    async def execute_next_step(self, request: PlanningAgentRequest, current_session_state: SessionState) -> PlanningAgentResponse:
        if not current_session_state.current_plan or not current_session_state.current_plan.steps:
            logger.warning(f"execute_next_step called for thread {request.thread_id} but no plan or steps found.")
            return PlanningAgentResponse(plan=current_session_state.current_plan, agent_status="error", error_details="No plan or steps to execute.", thread_id=request.thread_id)

        plan = current_session_state.current_plan

        next_pending_step: Optional[Step] = None
        for step in plan.steps:
            if step.status == "pending":
                next_pending_step = step
                break

        if not next_pending_step:
            plan.plan_status = "completed"
            plan.final_output = "All steps completed successfully." # Or gather results from steps
            logger.info(f"Plan completed for thread_id: {request.thread_id}")
            if request.thread_id:
                await self.checkpoint_manager.save_state(request.thread_id, current_session_state)
            return PlanningAgentResponse(plan=plan, agent_status="plan_completed", thread_id=request.thread_id)

        logger.info(f"Executing step '{next_pending_step.description}' (ID: {next_pending_step.step_id}) for thread: {request.thread_id}")
        next_pending_step.status = "in_progress"
        plan.plan_status = "in_progress"
        if request.thread_id:
            await self.checkpoint_manager.save_state(request.thread_id, current_session_state)

        if next_pending_step.agent_to_use == "CodeAgent":
            code_agent_instruction = next_pending_step.description
            # If the step specifically dictates a tool, pass it in context or modify CodeAgentRequest
            # For now, CodeAgent infers tool use from instruction or uses its own tool loop.
            # If step type was human_input, the description IS the question.
            if next_pending_step.tool_to_use == "ask_human_for_input":
                 code_agent_instruction = (
                    f"Please ask the human user the following question to proceed: "
                    f"'{next_pending_step.description}'. "
                    f"Use the 'ask_human_for_input' tool."
                 )


            code_agent_req = CodeAgentRequest(
                instruction=code_agent_instruction,
                project_directory=current_session_state.original_request_info.get("project_directory") if current_session_state.original_request_info else request.project_directory,
                thread_id=request.thread_id, # Crucial: use the same thread_id for CodeAgent
                # model_quality, provider can be inherited from original_request_info or PlanningAgentRequest
                model_quality=current_session_state.original_request_info.get("model_quality") if current_session_state.original_request_info else None,
                provider=current_session_state.original_request_info.get("provider") if current_session_state.original_request_info else None,
                # language needs to be inferred or passed if available in original_request_info
                language=current_session_state.original_request_info.get("language") if current_session_state.original_request_info else None,
            )

            logger.debug(f"Calling CodeAgent for step {next_pending_step.step_id} with instruction: {code_agent_req.instruction}")
            code_agent_response: CodeAgentResponse = await self.code_agent.generate_code(code_agent_req)

            if code_agent_response.agent_status == "completed":
                next_pending_step.status = "completed"
                next_pending_step.result = code_agent_response.generated_code or code_agent_response.explanation
                logger.info(f"Step {next_pending_step.step_id} completed by CodeAgent.")
            elif code_agent_response.agent_status == "requires_human_input":
                next_pending_step.status = "paused_for_human_input" # Custom status for step
                plan.plan_status = "paused_for_human_input" # Plan is also paused
                logger.info(f"Plan paused for human input at step {next_pending_step.step_id} by CodeAgent.")
                if request.thread_id: # State already saved by CodeAgent before returning HITL
                    await self.checkpoint_manager.save_state(request.thread_id, current_session_state) # Save updated plan/step status
                return PlanningAgentResponse(
                    plan=plan,
                    agent_status="paused_for_human_input", # PlanningAgent itself is paused
                    human_input_request=code_agent_response.human_input_request,
                    thread_id=request.thread_id
                )
            elif code_agent_response.agent_status == "error":
                next_pending_step.status = "failed"
                next_pending_step.error_details = code_agent_response.error_details
                plan.plan_status = "failed" # Or a more granular error status
                logger.error(f"Step {next_pending_step.step_id} failed by CodeAgent: {code_agent_response.error_details}")

            if request.thread_id:
                await self.checkpoint_manager.save_state(request.thread_id, current_session_state)

            # If the step was completed and not paused, try to execute the next one immediately
            if next_pending_step.status == "completed" and plan.plan_status == "in_progress":
                 return await self.execute_next_step(request, current_session_state) # Recursive call for next step

        elif next_pending_step.agent_to_use == "ResearchAgent": # Placeholder
            next_pending_step.status = "failed"
            next_pending_step.error_details = "ResearchAgent not implemented yet."
            plan.plan_status = "failed"
            logger.warning("ResearchAgent not implemented.")
            if request.thread_id: await self.checkpoint_manager.save_state(request.thread_id, current_session_state)
        else:
            next_pending_step.status = "failed"
            next_pending_step.error_details = f"Unknown agent type '{next_pending_step.agent_to_use}' for step or step type not executable by PlanningAgent."
            plan.plan_status = "failed"
            logger.warning(f"Step {next_pending_step.step_id} has unhandled agent type: {next_pending_step.agent_to_use}")
            if request.thread_id: await self.checkpoint_manager.save_state(request.thread_id, current_session_state)

        return PlanningAgentResponse(plan=plan, agent_status=plan.plan_status, thread_id=request.thread_id, error_details=plan.error_details)


    async def generate_plan(self, request: PlanningAgentRequest) -> PlanningAgentResponse:
        current_session_state: Optional[SessionState] = None
        is_new_thread = True

        if request.thread_id:
            logger.info(f"PlanningAgent: Attempting to load state for thread_id: {request.thread_id}")
            current_session_state = await self.checkpoint_manager.load_state(request.thread_id)
            if current_session_state:
                is_new_thread = False
                logger.info(f"PlanningAgent: Loaded state for thread_id: {request.thread_id}")
                if current_session_state.current_plan:
                    if current_session_state.current_plan.plan_status == "paused_for_human_input":
                        logger.info(f"Plan for thread {request.thread_id} is paused. Resuming execution.")
                        # The /resume endpoint should have added the human response.
                        # execute_next_step will pick up from the paused step.
                        return await self.execute_next_step(request, current_session_state)
                    elif current_session_state.current_plan.plan_status not in ["completed", "failed"]:
                        logger.info(f"Returning existing, active plan for thread_id: {request.thread_id}")
                        return PlanningAgentResponse(plan=current_session_state.current_plan, agent_status="plan_retrieved_existing", thread_id=request.thread_id)

                # If plan was completed/failed, or no plan, start fresh for this new goal
                current_session_state.current_plan = None
                current_session_state.conversation_history = [] # Reset history for new planning session
                # Keep original_request_info if it's from the very first request of this thread_id
                # For a new goal with same thread_id, we might want to update original_request_info
                current_session_state.original_request_info = request.model_dump(exclude_none=True, exclude={"goal"})
                current_session_state.original_request_info["goal"] = request.goal # ensure goal is part of it

        if not current_session_state:
            current_session_state = SessionState(
                thread_id=request.thread_id,
                original_request_info=request.model_dump(exclude_none=True)
            )
            is_new_thread = True # Explicitly set for clarity if state was None
            logger.info(f"PlanningAgent: Initialized new session state for thread_id: {request.thread_id}")

        current_session_state.current_plan = Plan(goal=request.goal, plan_status="generating")
        if request.thread_id and is_new_thread: # Save state with new plan object and original_request_info
             await self.checkpoint_manager.save_state(request.thread_id, current_session_state)

        system_prompt = self._construct_system_prompt()
        user_prompt_for_planning = self._construct_user_prompt_for_planning(request)

        current_session_state.conversation_history = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt_for_planning)
        ]

        llm_plan_str = await self._get_llm_plan_response(current_session_state.conversation_history, request)

        if not llm_plan_str:
            current_session_state.current_plan.plan_status = "failed"
            current_session_state.current_plan.error_details = "LLM failed to generate a plan response."
            if request.thread_id: await self.checkpoint_manager.save_state(request.thread_id, current_session_state)
            return PlanningAgentResponse(plan=current_session_state.current_plan, agent_status="error", error_details="LLM response for plan generation was empty or failed.", thread_id=request.thread_id)

        parsed_steps = self._parse_llm_plan_to_steps(llm_plan_str)
        if not parsed_steps:
            current_session_state.current_plan.plan_status = "failed"
            current_session_state.current_plan.error_details = "Failed to parse LLM response into actionable steps."
            if request.thread_id: await self.checkpoint_manager.save_state(request.thread_id, current_session_state)
            return PlanningAgentResponse(plan=current_session_state.current_plan, agent_status="error", error_details="Failed to parse plan steps from LLM response.", thread_id=request.thread_id)

        current_session_state.current_plan.steps = parsed_steps
        current_session_state.current_plan.plan_status = "ready"
        current_session_state.add_message(role="assistant", content=llm_plan_str) # LLM's raw plan as assistant msg
        if request.thread_id: await self.checkpoint_manager.save_state(request.thread_id, current_session_state)

        logger.info(f"PlanningAgent: Plan generated for thread_id: {request.thread_id}. Proceeding to execute first step.")
        return await self.execute_next_step(request, current_session_state)

```
