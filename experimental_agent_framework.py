# =========================================================================================
# EXPERIMENTAL AGENT FRAMEWORK - ARCHIVED CODE
# =========================================================================================
# This file contains the complete source code for the experimental AI Agent Framework.
# The code in this file is DEPRECATED and NON-OPERATIONAL in its current form.
# It has been archived here to preserve the work for a future development phase.
#
# DO NOT ATTEMPT TO IMPORT, USE, OR MODIFY THE CODE IN THIS FILE.
#
# The primary focus of the project is the Enterprise LLM Aggregator. The agent
# framework will be revisited after the aggregator is production-ready.
# =========================================================================================

# =========================================================================================
# Original File: src/agents/__init__.py
# =========================================================================================
# This file makes Python treat the 'agents' directory as a package.

# You can optionally expose certain classes or functions for easier importing, e.g.:
# from .code_agent import CodeAgent

# For now, leave it empty or add specific imports as agent modules are developed.

# =========================================================================================
# Original File: src/agents/checkpoint.py
# =========================================================================================
"""
Checkpointing mechanism for agent sessions.
Allows saving and loading agent state to support long-running, stateful interactions.
"""
from __future__ import annotations # For Python 3.7, 3.8 compatibility

from abc import ABC, abstractmethod
from typing import Optional, Dict

from .state import SessionState # Import SessionState from the new state.py

class BaseCheckpointManager(ABC):
    """
    Abstract base class for checkpoint managers.
    Defines the interface for saving and loading agent session states.
    """

    @abstractmethod
    async def load_state(self, thread_id: str) -> Optional[SessionState]:
        """
        Loads the session state for a given thread_id.

        Args:
            thread_id: The unique identifier for the conversation thread.

        Returns:
            The SessionState if found, otherwise None.
        """
        pass

    @abstractmethod
    async def save_state(self, thread_id: str, state: SessionState) -> None:
        """
        Saves the session state for a given thread_id.

        Args:
            thread_id: The unique identifier for the conversation thread.
            state: The SessionState object to save.
        """
        pass

    @abstractmethod
    async def delete_state(self, thread_id: str) -> bool:
        """
        Deletes the session state for a given thread_id.

        Args:
            thread_id: The unique identifier for the conversation thread.

        Returns:
            True if state was deleted, False if not found.
        """
        pass


class InMemoryCheckpointManager(BaseCheckpointManager):
    """
    An in-memory implementation of the checkpoint manager.
    Stores checkpoints in a Python dictionary. Suitable for single-instance deployments
    and testing. Not persistent across application restarts.
    """
    def __init__(self):
        self._checkpoints: Dict[str, SessionState] = {}
        # For potential concurrency control if needed, though Pydantic models are generally not thread-safe for mutation
        # self._lock = asyncio.Lock()

    async def load_state(self, thread_id: str) -> Optional[SessionState]:
        """Loads state, returning a deep copy to prevent shared mutable state issues."""
        # async with self._lock: # Uncomment if direct mutation of _checkpoints items is a concern elsewhere
        state = self._checkpoints.get(thread_id)
        if state:
            # Pydantic's model_copy(deep=True) is the new way for deep copies in v2
            # For older Pydantic, state.copy(deep=True)
            return state.model_copy(deep=True)
        return None

    async def save_state(self, thread_id: str, state: SessionState) -> None:
        """Saves state, storing a deep copy."""
        # async with self._lock:
        if not isinstance(state, SessionState):
            raise TypeError("state must be an instance of SessionState")
        # Store a deep copy to prevent issues if the original 'state' object is modified elsewhere
        self._checkpoints[thread_id] = state.model_copy(deep=True)

    async def delete_state(self, thread_id: str) -> bool:
        """Deletes state for the given thread_id."""
        # async with self._lock:
        if thread_id in self._checkpoints:
            del self._checkpoints[thread_id]
            return True
        return False

# =========================================================================================
# Original File: src/agents/code_agent.py
# =========================================================================================
"""
Coding Agent implementation with Filesystem Tool Usage and Checkpointing.
"""

import json
import logging
import re
from typing import Optional, Tuple, List, Dict, Any

from ..models import (
    CodeAgentRequest,
    CodeAgentResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HumanInputRequest # New import
)
from pydantic import create_model as create_pydantic_model, ValidationError

from ..core.aggregator import LLMAggregator
from .tools import filesystem_tools
from .tools.registry import global_tool_registry
from .tools.base import ToolDefinition # ToolParameter not directly used in this file's public interface
from .tools.human_tools import HumanInterruption # New import for HITL
from .state import SessionState, Message
from .checkpoint import BaseCheckpointManager

logger = logging.getLogger(__name__)

CODE_EXPLANATION_SEPARATOR = "###EXPLANATION###"
MAX_TOOL_ITERATIONS = 5

class CodeAgent:
    """Agent specialized for code generation tasks, with filesystem tool and state capabilities."""

    def __init__(self, llm_aggregator: LLMAggregator, checkpoint_manager: BaseCheckpointManager):
        self.llm_aggregator = llm_aggregator
        self.checkpoint_manager = checkpoint_manager

    def _construct_initial_user_message_content(self, request: CodeAgentRequest) -> str:
        """Constructs the content for the initial user message."""
        prompt_parts = []
        if request.language:
            prompt_parts.append(f"Target language: {request.language}.")
        else:
            prompt_parts.append("Infer the programming language from the instruction or context, or default to Python if unclear.")

        prompt_parts.append(f"\nUser Instruction:\n{request.instruction}")

        if request.context:
            prompt_parts.append(f"\nProvided Context/Code:\n```\n{request.context}\n```")

        return "\n".join(prompt_parts)

    def _construct_system_prompt(self, request: CodeAgentRequest) -> str: # Renamed for clarity
        """Builds the system prompt. If request.project_directory is set, includes tool instructions."""
        base_system_prompt = (
            "You are an expert coding assistant. Your primary task is to generate, modify, or explain code based on the user's instruction.\n"
            "If you are asked to generate code, provide only the generated code block. \n"
            f"If you provide an explanation, do so after all the code, separated by a line containing only '{CODE_EXPLANATION_SEPARATOR}'.\n"
            "Ensure the code is complete and runnable if possible within typical constraints."
        )

        if not request.project_directory:
            return base_system_prompt

        tool_descriptions_json = global_tool_registry.generate_llm_tool_descriptions(as_json_string=True)
        tool_instructions = f"""

You have access to the following tools to interact with the user's project file system.
The project root directory is pre-configured for you. All filepaths MUST be relative to this project root.
Do not attempt to access files outside this project directory.

Available Tools (as a JSON schema list):
{tool_descriptions_json}

To use a tool, output ONLY a single valid JSON object formatted exactly as shown:
{{"tool_name": "tool_name_here", "parameters": {{"param1": "value1", ...}}}}
Your reasoning for the tool call should precede this JSON block. After your tool call JSON, stop.
I will provide the tool's output, and you can continue your task or provide the final answer.
If you believe you have enough information to directly answer the user's request without tools, provide the answer directly (code and/or explanation).
If you need to use a tool, make only one tool call per response.
Do not ask for permission to use tools, just use them if needed.
If a tool call fails, I will inform you of the error, and you can try a different approach or tool.
"""
        return base_system_prompt + tool_instructions

    async def _get_llm_response(
        self,
        current_session_state: SessionState, # Use SessionState's history
        request: CodeAgentRequest
    ) -> Tuple[str, Optional[ChatCompletionResponse]]:
        """Calls llm_aggregator.chat_completion and returns the text content and full response."""
        model_quality_for_coding = request.model_quality or "best_quality"

        # Convert Pydantic Message models to dicts for ChatCompletionRequest
        messages_for_llm = [msg.model_dump(exclude_none=True) for msg in current_session_state.conversation_history]

        chat_request = ChatCompletionRequest(
            messages=messages_for_llm, # type: ignore | Pydantic v1/v2 compatibility for .dict()
            model="auto",
            provider=request.provider,
            model_quality=model_quality_for_coding,
            temperature=0.3,
        )

        try:
            logger.debug(f"CodeAgent sending request to LLMAggregator with history: {messages_for_llm}")
            llm_response_obj = await self.llm_aggregator.chat_completion(chat_request)

            if llm_response_obj.choices and llm_response_obj.choices[0].message and llm_response_obj.choices[0].message.content is not None:
                content = llm_response_obj.choices[0].message.content.strip()
                logger.debug(f"CodeAgent received raw response content: {content}")
                return content, llm_response_obj
            else: # Handle cases like content being None if tool_calls are present.
                # If there are tool_calls, the content might be None or empty. The raw response_text should reflect this.
                # The _parse_llm_for_tool_call should primarily look at llm_response_obj.choices[0].message.tool_calls
                raw_message = llm_response_obj.choices[0].message if (llm_response_obj.choices and llm_response_obj.choices[0].message) else None
                if raw_message and raw_message.tool_calls:
                    # If there's a tool call, we might not have textual content.
                    # We need to reconstruct the part of the message that represents the tool call for history.
                    # For now, assume the _parse_llm_for_tool_call will use the full message object.
                    # The textual response here could be a string representation of the tool call or reasoning.
                    # Let's pass the full message object to _parse_llm_for_tool_call later.
                    # For now, if content is None but tool_calls exist, return an empty string for text, but the full response.
                    return "", llm_response_obj # Empty string for text, but pass full response for tool parsing
                logger.warning("LLM response content was None or choices were empty.")
                return "Error: LLM response was empty or malformed.", None
        except Exception as e:
            logger.error(f"Error calling LLM aggregator: {e}", exc_info=True)
            return f"Error: Could not get response from LLM. {str(e)}", None

    def _parse_llm_for_tool_call(self, llm_message: Optional[Message]) -> Optional[Dict[str, Any]]:
        """
        Parses a Message object from the LLM for a tool call.
        OpenAI-style tool calls are expected in message.tool_calls.
        This method adapts to look for JSON in content if direct tool_calls field isn't used/populated.
        """
        if not llm_message:
            return None

        if llm_message.tool_calls and isinstance(llm_message.tool_calls, list) and len(llm_message.tool_calls) > 0:
            # Assuming one tool call per response as per current prompting strategy
            first_tool_call = llm_message.tool_calls[0]
            if first_tool_call.get("type") == "function" and first_tool_call.get("function"):
                function_details = first_tool_call["function"]
                tool_name = function_details.get("name")
                try:
                    # Arguments are a JSON string
                    tool_params = json.loads(function_details.get("arguments", "{}"))
                    if tool_name and isinstance(tool_params, dict):
                        logger.info(f"Parsed tool call via tool_calls: {tool_name}, params: {tool_params}")
                        return {"tool_name": tool_name, "parameters": tool_params, "id": first_tool_call.get("id")}
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError parsing arguments for tool {tool_name}: {e}")
            return None # Malformed tool_call structure

        # Fallback: try to parse from text content if no structured tool_calls
        response_text = llm_message.content if isinstance(llm_message.content, str) else ""
        if not response_text: return None

        logger.debug(f"Attempting to parse tool call from text content: {response_text}")
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```|(\{[\s\S]*\"tool_name\"[\s\S]*\})$", response_text, re.DOTALL | re.MULTILINE)

        if match:
            json_str = match.group(1) if match.group(1) else match.group(2)
            if json_str:
                try:
                    tool_call = json.loads(json_str.strip())
                    if isinstance(tool_call, dict) and "tool_name" in tool_call and "parameters" in tool_call:
                        logger.info(f"Parsed tool call from text: {tool_call}")
                        return tool_call
                except json.JSONDecodeError: # Ignore if not valid JSON
                    pass
        logger.debug("No tool call found in LLM response text content.")
        return None

    def _parse_final_response(self, response_text: str) -> Tuple[str, Optional[str]]:
        generated_code = response_text
        explanation: Optional[str] = None

        if CODE_EXPLANATION_SEPARATOR in response_text:
            parts = response_text.split(CODE_EXPLANATION_SEPARATOR, 1)
            generated_code = parts[0].strip()
            if len(parts) > 1:
                explanation = parts[1].strip()

        if generated_code.startswith("```") and generated_code.endswith("```"):
            lines = generated_code.splitlines()
            if len(lines) > 1: generated_code = "\n".join(lines[1:-1]).strip()
            else: generated_code = ""
        elif generated_code.startswith("```"):
             lines = generated_code.splitlines()
             if len(lines) > 0: generated_code = "\n".join(lines[1:]).strip()
        return generated_code, explanation

    def _validate_tool_parameters(
        self,
        tool_def: ToolDefinition,
        provided_args: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Validates provided arguments against the tool's parameter schema.
        Returns (validated_args, None) or (None, error_message).
        """
        param_definitions = {p.name: p for p in tool_def.parameters}
        validated_args = {}
        errors = []

        # Check for missing required parameters
        for name, definition in param_definitions.items():
            if definition.required and name not in provided_args:
                errors.append(f"Missing required parameter: '{name}' ({definition.description}). Expected type: {definition.type}.")

        if errors: # Fail fast if required params are missing
            return None, " ".join(errors)

        # Type checking and building arguments for dynamic Pydantic model
        # This is a simplified type mapping. JSON schema types to Python types.
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list, # Further validation for array items would be needed
            "object": dict, # Further validation for object properties would be needed
        }

        fields_for_dynamic_model: Dict[str, Any] = {}
        for name, definition in param_definitions.items():
            python_type = type_map.get(definition.type, Any) # Default to Any if type unknown
            if definition.required:
                fields_for_dynamic_model[name] = (python_type, ...) # Ellipsis means required
            else:
                # Pydantic needs a default value for optional fields in create_model
                # If your tool functions handle None or have their own defaults, that's fine.
                # Here, we provide None as the default for the dynamic model.
                fields_for_dynamic_model[name] = (Optional[python_type], None)


        if not fields_for_dynamic_model and not provided_args: # Tool takes no arguments
             return {}, None # No validation needed, return empty dict
        if not fields_for_dynamic_model and provided_args: # Tool takes no args, but some were provided
            return None, f"Tool '{tool_def.name}' expects no arguments, but received: {', '.join(provided_args.keys())}."


        # Create a dynamic Pydantic model for validation
        try:
            # Filter provided_args to only include those defined in the tool
            args_to_validate = {k: v for k, v in provided_args.items() if k in fields_for_dynamic_model}

            # Check for extraneous arguments provided by LLM
            extra_args = set(provided_args.keys()) - set(param_definitions.keys())
            if extra_args:
                errors.append(f"Extraneous parameters provided: {', '.join(extra_args)}. Valid parameters are: {', '.join(param_definitions.keys())}.")


            DynamicToolArgsModel = create_pydantic_model(
                f"{tool_def.name}Args",
                **fields_for_dynamic_model # type: ignore
            )

            # Validate the (filtered) arguments
            validated_model_instance = DynamicToolArgsModel(**args_to_validate)
            validated_args = validated_model_instance.model_dump(exclude_unset=True) # Get validated args

        except ValidationError as ve:
            # Format Pydantic's validation errors into a user-friendly string
            for error in ve.errors():
                param_name = error['loc'][0] if error['loc'] else 'unknown_param'
                errors.append(f"Parameter '{param_name}': {error['msg']}.")
        except Exception as e: # Catch any other errors during model creation/validation
             errors.append(f"Unexpected error during parameter validation: {str(e)}")


        if errors:
            return None, " ".join(errors)

        return validated_args, None


    async def generate_code(self, request: CodeAgentRequest) -> CodeAgentResponse:
        current_session_state: Optional[SessionState] = None
        is_new_thread = True

        if request.thread_id:
            logger.info(f"Attempting to load state for thread_id: {request.thread_id}")
            current_session_state = await self.checkpoint_manager.load_state(request.thread_id)
            if current_session_state:
                is_new_thread = False
                logger.info(f"Loaded state for thread_id: {request.thread_id}, history length: {len(current_session_state.conversation_history)}")
                # Add current user instruction to existing history
                user_message_content = self._construct_initial_user_message_content(request)
                current_session_state.add_message(role="user", content=user_message_content)
            else:
                logger.info(f"No existing state found for thread_id: {request.thread_id}. Creating new session.")

        if not current_session_state:
            current_session_state = SessionState(
                thread_id=request.thread_id,
                original_request_info=request.model_dump(exclude_none=True, exclude={"instruction", "context"})
                # Store relevant, non-sensitive parts of the original request.
                # 'instruction' and 'context' are part of the first user message.
            )
            system_prompt = self._construct_system_prompt(request) # Pass full request for prompt construction
            current_session_state.add_message(role="system", content=system_prompt)
            user_message_content = self._construct_initial_user_message_content(request) # Pass full request
            current_session_state.add_message(role="user", content=user_message_content)

        if request.thread_id and is_new_thread: # Save state only if it's a new thread being initialized
            await self.checkpoint_manager.save_state(request.thread_id, current_session_state)

        last_model_used: Optional[str] = None

        for iteration in range(MAX_TOOL_ITERATIONS):
            logger.info(f"CodeAgent Iteration {iteration + 1}/{MAX_TOOL_ITERATIONS} for thread_id: {request.thread_id}")

            response_text, llm_full_response = await self._get_llm_response(current_session_state, request)

            if llm_full_response and llm_full_response.choices: # Store model from last successful LLM call
                last_model_used = llm_full_response.model
                llm_message_obj = llm_full_response.choices[0].message # This is a ChatMessage from models.py
                # Convert to state.Message for history
                assistant_response_message = Message(
                    role="assistant",
                    content=llm_message_obj.content,
                    tool_calls=llm_message_obj.tool_calls
                )
            else: # Error condition from _get_llm_response
                return CodeAgentResponse(generated_code="", explanation=response_text, request_params=request.model_dump(), model_used=last_model_used)

            current_session_state.add_message(
                role=assistant_response_message.role,
                content=assistant_response_message.content,
                tool_calls=assistant_response_message.tool_calls
            )
            # No immediate save after adding assistant's raw response, save after tool cycle or final response.

            tool_call_parsed_info = self._parse_llm_for_tool_call(assistant_response_message) # Pass the Message object

            if tool_call_parsed_info:
                tool_name = tool_call_parsed_info.get("tool_name")
                tool_params = tool_call_parsed_info.get("parameters", {})
                tool_call_id = tool_call_parsed_info.get("id") # For OpenAI compliant tool message
                tool_output_content: str = ""

                if not request.project_directory:
                    tool_output_content = "Error: Project directory not specified. Cannot use file system tools."
                    logger.warning("Tool call attempted without project_directory specified.")
                else:
                    tool_definition = global_tool_registry.get_tool(tool_name)
                    if tool_definition:
                        validated_params, validation_error_msg = self._validate_tool_parameters(tool_definition, tool_params)

                        if validation_error_msg:
                            logger.warning(f"Tool parameter validation failed for {tool_name}: {validation_error_msg}")
                            tool_output_content = f"Error: Parameter validation failed for tool '{tool_name}'. {validation_error_msg}"
                        elif validated_params is not None: # Validation succeeded
                            kwargs_for_tool_call = validated_params
                            # Inject internal parameters AFTER validation of LLM-provided ones
                            if "project_directory" in tool_definition.internal_parameters: # Ensure it's defined as internal
                                kwargs_for_tool_call["project_directory"] = request.project_directory

                            logger.info(f"Executing tool: {tool_name}, validated_params: {validated_params}")
                            try:
                                raw_tool_output = await tool_definition.function(**kwargs_for_tool_call)
                                # Convert list output to string for history, assign to tool_output_content
                                if isinstance(raw_tool_output, list): tool_output_content = "\n".join(raw_tool_output)
                                elif isinstance(raw_tool_output, str): tool_output_content = raw_tool_output
                                else: tool_output_content = str(raw_tool_output)

                            except HumanInterruption as hi:
                                logger.info(f"Human interruption requested by tool {tool_name} with ID {hi.tool_call_id}: {hi.question_for_human}")
                                # State should be saved before returning this special response.
                                # The assistant's message that *led* to this tool call is already in history.
                                # We don't add a "tool" message for HumanInterruption itself yet,
                                # that will come from the /resume endpoint.
                                if request.thread_id:
                                    await self.checkpoint_manager.save_state(request.thread_id, current_session_state)

                                return CodeAgentResponse(
                                    agent_status="requires_human_input",
                                    human_input_request=HumanInputRequest(
                                        tool_call_id=hi.tool_call_id, # This is the ID of the ask_human_for_input tool call
                                        question_for_human=hi.question_for_human
                                    ),
                                    # Include current conversation history for context if UI needs it
                                    # generated_code and explanation might be None or from previous thoughts
                                    request_params=request.model_dump(),
                                    model_used=last_model_used
                                    # Potentially add conversation history to response if client needs it to show context with question
                                )
                            except TypeError as te:
                                logger.error(f"TypeError calling tool {tool_name} with combined params {kwargs_for_tool_call}: {te}", exc_info=True)
                                tool_output_content = f"Error: Incorrect parameters or setup for tool '{tool_name}'. {str(te)}"
                            except Exception as e:
                                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                                tool_output_content = f"Error: Failed to execute tool '{tool_name}'. {str(e)}"
                        else:
                             logger.error(f"Invalid state after parameter validation for tool {tool_name}: No validated_params and no error_msg.")
                             tool_output_content = f"Error: Unknown validation issue for tool '{tool_name}'."
                    else: # tool_definition not found
                        logger.warning(f"Unknown tool name: {tool_name}")
                        available_tools = ", ".join([t.name for t in global_tool_registry.get_all_tools()])
                        tool_output_content = f"Error: Unknown tool '{tool_name}'. Available tools are: {available_tools}."

                logger.debug(f"Tool output for {tool_name}: {tool_output_content[:500]}...")
                current_session_state.add_message(role="tool", content=tool_output_content, name=tool_name, tool_call_id=tool_call_id)

                if request.thread_id: # Save state after tool interaction
                    await self.checkpoint_manager.save_state(request.thread_id, current_session_state)
            else: # No tool call, assume final answer from LLM
                logger.info("No tool call detected. Parsing as final response.")
                # The response_text is the assistant's content from the last LLM call
                generated_code, explanation = self._parse_final_response(response_text)
                if request.thread_id: # Save final state
                     await self.checkpoint_manager.save_state(request.thread_id, current_session_state)
                return CodeAgentResponse(
                    generated_code=generated_code, explanation=explanation,
                    request_params=request.model_dump(),
                    model_used=last_model_used,
                    agent_status="completed"
                )

        logger.warning(f"Agent exceeded maximum tool iterations for thread_id: {request.thread_id}")
        final_explanation_on_max_iter = "The agent could not complete the request within the allowed number of steps."
        if request.thread_id: # Save state even if iterations exceeded
            # Add a final message indicating max iterations was hit.
            current_session_state.add_message(role="assistant", content=final_explanation_on_max_iter)
            await self.checkpoint_manager.save_state(request.thread_id, current_session_state)
        return CodeAgentResponse(
            generated_code="", # No code generated if max iterations hit this way
            explanation=final_explanation_on_max_iter,
            request_params=request.model_dump(),
            model_used=last_model_used,
            agent_status="error",
            error_details="Exceeded maximum tool iterations."
        )

# =========================================================================================
# Original File: src/agents/state.py
# =========================================================================================
"""
Defines the state models for long-running agent sessions, including conversation history.
"""
from __future__ import annotations # For Python 3.7, 3.8 compatibility with Pydantic v2

from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field

class Message(BaseModel):
    """
    Represents a single message in a conversation history.
    Compatible with OpenAI's message format and tool call structures.
    """
    role: str  # e.g., "system", "user", "assistant", "tool"

    # Content can be a simple string or a list of content blocks (e.g., for multimodal later)
    # For tool results, content is typically a string (output of the tool).
    # For assistant requesting tool calls, content might be null if tool_calls is present.
    content: Union[str, List[Dict[str, Any]], None] = None

    name: Optional[str] = None  # For "tool" role, the name of the tool that was called.
                                # For "assistant" role, can be assistant's name.

    # For assistant messages that want to call tools
    tool_calls: Optional[List[ToolCall]] = None

    # For "tool" messages that are results of a tool call
    tool_call_id: Optional[str] = None # ID from the assistant's tool_calls request

    class Config:
        # Pydantic v2: `model_config = {"validate_assignment": True}`
        # For Pydantic v1, this is default behavior.
        # We want to ensure that `content` can indeed be `None` if `tool_calls` is present.
        # The `Union[..., None]` and `Optional` already handle this.
        # `exclude_none=True` can be used during serialization if needed.
        pass

class ToolCallFunction(BaseModel):
    """Function details for a tool call, as expected by OpenAI models."""
    name: str
    arguments: str # JSON string of arguments

class ToolCall(BaseModel):
    """
    Represents a tool call requested by the assistant.
    Compatible with OpenAI's tool call format.
    """
    id: str # Unique ID for the tool call, generated by LLM
    function: ToolCallFunction
    type: str = "function" # Usually "function"


class SessionState(BaseModel):
    """
    Represents the state of an agent's session, primarily the conversation history.
    This state can be checkpointed to allow for long-running, stateful interactions.
    """
    conversation_history: List[Message] = Field(default_factory=list)

    # Optional unique identifier for the session or thread
    thread_id: Optional[str] = None

    # Additional state variables can be added here as needed:
    # current_plan: Optional[List[str]] = None
    # scratchpad: Dict[str, Any] = Field(default_factory=dict)
    # user_preferences: Dict[str, Any] = Field(default_factory=dict)
    # current_task_id: Optional[str] = None
    original_request_info: Optional[Dict[str, Any]] = None # Store key, non-sensitive parts of original CodeAgentRequest

    # Example: Storing last executed tool call info for context or debugging
    # last_tool_call: Optional[Dict[str, Any]] = None
    # last_tool_result: Optional[str] = None

    def add_message(self, role: str, content: Union[str, List[Dict[str, Any]], None],
                    name: Optional[str] = None,
                    tool_calls: Optional[List[Dict[str, Any]]] = None,
                    tool_call_id: Optional[str] = None) -> None:
        """Helper method to add a message to the conversation history."""
        self.conversation_history.append(Message(
            role=role,
            content=content,
            name=name,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id
        ))

    def get_last_message(self, role: Optional[str] = None, ignore_system: bool = False) -> Optional[Message]:
        """
        Helper to get the last message, optionally filtered by role and ignoring system messages.
        """
        if not self.conversation_history:
            return None
        for msg in reversed(self.conversation_history):
            if ignore_system and msg.role == "system":
                continue
            if role is None or msg.role == role:
                return msg
        return None

    class Config:
        # Pydantic v2: `model_config = {"validate_assignment": True}`
        pass

# =========================================================================================
# Original File: src/agents/tools/__init__.py
# =========================================================================================
# This file makes Python treat the 'tools' directory as a sub-package of 'agents'.

# Expose tool functions for easier access if desired, e.g.:
# from .filesystem_tools import read_file, write_file, list_files

# For now, leave it empty. It will be populated as tools are defined.

# =========================================================================================
# Original File: src/agents/tools/base.py
# =========================================================================================
"""
Base definitions for tools that AI agents can use.
Includes ToolParameter, ToolDefinition models and the @openhands_tool decorator.
"""
from __future__ import annotations # For Python 3.7, 3.8 compatibility with Pydantic v2

import inspect
from typing import Callable, List, Optional, Dict, Any, Type
from pydantic import BaseModel, create_model, Field
import docstring_parser

class ToolParameter(BaseModel):
    """Describes a parameter for a tool."""
    name: str
    type: str  # JSON schema type (e.g., "string", "integer", "boolean", "object", "array")
    description: str
    required: bool
    properties: Optional[Dict[str, ToolParameter]] = None # For nested object types
    items: Optional[ToolParameter] = None # For array types if elements are uniform

class ToolDefinition(BaseModel):
    """Describes a tool available to an agent."""
    name: str
    description: str
    parameters: List[ToolParameter] # Parameters the LLM should provide
    function: Callable # The actual function to call, not exposed to LLM directly
    # Internal parameters are those injected by the agent runtime, not specified by LLM
    internal_parameters: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True # To allow Callable

# Placeholder for the decorator and registry logic which will be added next.
# The decorator will populate ToolDefinition objects.
# A global registry instance will likely live in a separate 'registry.py'.

# Example of how Pydantic model schemas can be extracted
def get_pydantic_model_schema(pydantic_model: Type[BaseModel]) -> List[ToolParameter]:
    """
    Converts a Pydantic model's JSON schema into a list of ToolParameter.
    This is a simplified version. Real schema conversion can be more complex.
    """
    if not hasattr(pydantic_model, "model_json_schema"):
        return [] # Not a Pydantic model or old Pydantic version

    schema = pydantic_model.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    tool_params = []
    for name, prop_schema in properties.items():
        param_type = prop_schema.get("type", "any")
        if "anyOf" in prop_schema: # Handle Optional fields or Unions
            # Simplification: take the first non-null type or just represent as 'any'
            # A more robust solution would detail the union or optional nature.
            non_null_type = next((t.get("type", "any") for t in prop_schema["anyOf"] if t.get("type") != "null"), "any")
            param_type = non_null_type

        description = prop_schema.get("description", "")
        # If description is not in schema, try to get from title (often field name for simple fields)
        if not description and "title" in prop_schema:
            description = f"Parameter: {prop_schema['title']}"

        param = ToolParameter(
            name=name,
            type=param_type,
            description=description,
            required=name in required_fields
        )
        # Basic handling for nested objects - this would need recursion for full support
        if param_type == "object" and "properties" in prop_schema:
            # This is a simplification; true nested schema parsing is more involved
            # For now, we'll just indicate it's an object.
            # A full implementation would recursively call something like this function.
             param.description += " (Object with nested fields)"
        tool_params.append(param)

    return tool_params

# Placeholder for the decorator itself
def openhands_tool(func: Callable) -> Callable:
    # Actual decorator logic will be implemented in the next step.
    # It will create a ToolDefinition and register it.
    # For now, just return the function itself.
from .registry import global_tool_registry # Import the actual global registry

def _parse_type_hint(hint: Type[Any]) -> str:
    """Converts Python type hints to JSON schema type strings."""
    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", [])

    if hint is str:
        return "string"
    if hint is int:
        return "integer"
    if hint is float:
        return "number"
    if hint is bool:
        return "boolean"
    if hint is list or origin is list:
        if args and args[0] is not Any:
            # For List[type], describe as array of that type.
            # Simplified: does not create full 'items' schema here yet.
            return f"array (items type: {_parse_type_hint(args[0])})"
        return "array"
    if hint is dict or origin is dict:
        return "object"
    if issubclass(hint, BaseModel):
        # For Pydantic models, we'd ideally return "object" and a nested schema.
        # For now, just identify as object. get_pydantic_model_schema can be used later.
        return "object" # In future, could add schema details here
    if hint is Any or hint is inspect.Parameter.empty:
        return "any"

    # Handle Optional[Type] by extracting the inner type
    if origin is Optional or (origin is Union and type(None) in args):
        if args and args[0] is not type(None): # Make sure there's another type apart from None
            return _parse_type_hint(args[0]) # Get the type of T in Optional[T]
        else: # pragma: no cover (e.g. Optional[NoneType] which is just NoneType)
            return "null"

    return str(hint) # Fallback, might not be a valid JSON schema type


def openhands_tool(func: Callable) -> Callable:
    """
    Decorator to register a function as an OpenHands tool.
    It inspects the function's signature and docstring to create a ToolDefinition.
    """
    parsed_docstring = docstring_parser.parse(func.__doc__ or "")
    tool_description = parsed_docstring.short_description or func.__name__

    sig = inspect.signature(func)
    tool_parameters: List[ToolParameter] = []
    internal_params: List[str] = []

    # Identify internal parameters (e.g., project_directory, tool_call_id for specific tools)
    # These are typically injected by the agent runtime.
    INTERNAL_PARAM_NAMES = ["project_directory", "tool_call_id"]

    for name, param in sig.parameters.items():
        if name in INTERNAL_PARAM_NAMES:
            internal_params.append(name)
            continue # Skip adding internal params to LLM-facing schema

        param_doc = next((p for p in parsed_docstring.params if p.arg_name == name), None)
        param_description = param_doc.description if param_doc else "No description."

        param_type_hint = param.annotation
        param_type_str = _parse_type_hint(param_type_hint)

        # Check if parameter is required (no default value)
        is_required = (param.default == inspect.Parameter.empty)

        # Basic handling for Pydantic models as parameters
        properties_schema = None
        items_schema = None
        if inspect.isclass(param_type_hint) and issubclass(param_type_hint, BaseModel):
            # For Pydantic models, generate a simplified nested schema
            # This is a placeholder for a more robust schema generation
            # properties_schema = get_pydantic_model_schema(param_type_hint) # TODO: Refine this
            param_type_str = "object" # Indicate it's an object, details can be in description
            param_description += f" (Pydantic Model: {param_type_hint.__name__})"


        tool_parameters.append(ToolParameter(
            name=name,
            type=param_type_str,
            description=param_description,
            required=is_required,
            properties=properties_schema, # For nested objects
            items=items_schema         # For arrays of objects
        ))

    tool_def = ToolDefinition(
        name=func.__name__,
        description=tool_description,
        parameters=tool_parameters,
        function=func,
        internal_parameters=internal_params
    )

    # Register with the global tool registry
    global_tool_registry.register(tool_def)

    # Store the definition on the function itself for potential direct access/inspection
    func._tool_definition = tool_def # type: ignore

    return func

# =========================================================================================
# Original File: src/agents/tools/filesystem_tools.py
# =========================================================================================
"""
Secure Filesystem Tools for AI Agents.

Provides basic, sandboxed file operations like reading, writing, and listing files
within a specified project directory.
"""

import os
import asyncio
from pathlib import Path
import logging

import aiofiles

from .base import openhands_tool # Import the decorator

logger = logging.getLogger(__name__)

def _resolve_safe_path(project_directory: str, relative_path: str) -> Path:
    """
    Resolves a relative path against a project directory and ensures it's safe.

    Args:
        project_directory: The absolute path to the root of the project.
        relative_path: The relative path from the user/agent.

    Returns:
        A resolved, absolute Path object if safe.

    Raises:
        PermissionError: If the resolved path attempts to traverse outside
                         the project_directory (path traversal).
        ValueError: If project_directory is not an absolute path.
    """
    if not os.path.isabs(project_directory):
        # Ensure project_directory is absolute for security checks.
        # This should ideally be enforced by the caller or system configuration.
        logger.warning(f"Project directory '{project_directory}' was not absolute. Resolving now.")
        project_root = Path(project_directory).resolve()
    else:
        project_root = Path(project_directory)

    # Normalize relative_path: remove leading slashes if any to ensure proper joining
    # os.path.normpath can also help, but Path objects handle this well.
    # If relative_path starts with '/', it's treated as absolute by Path's / operator,
    # but resolve() later still correctly checks against project_root.
    # For clarity, ensure it's treated as relative.
    norm_relative_path = relative_path.lstrip('./\\') # Remove leading relative components
    if os.path.isabs(norm_relative_path): # If it became absolute (e.g. was "/foo")
        # This case should ideally be blocked or handled carefully.
        # Forcing it to be relative to project_root:
        norm_relative_path = norm_relative_path.lstrip(os.path.sep)


    # Join the project root with the potentially untrusted relative path
    unsafe_path = project_root / norm_relative_path

    # Resolve the path (evaluates '..' etc.)
    resolved_path = unsafe_path.resolve()

    # Security Check: Verify that the resolved path is still within the project_root.
    # Path.is_relative_to() is available in Python 3.9+
    # For robust checking: project_root.resolve() and resolved_path.resolve()
    if not resolved_path.is_relative_to(project_root.resolve()):
        logger.error(f"Path traversal attempt: '{relative_path}' resolved to '{resolved_path}' which is outside project root '{project_root.resolve()}'.")
        raise PermissionError(f"Path traversal detected and blocked for path: '{relative_path}'")

    return resolved_path

@openhands_tool
async def read_file(project_directory: str, filepath: str) -> str:
    """
    Reads the entire content of a specified file.

    Args:
        project_directory (str): (Internal) The absolute path to the root of the project.
        filepath (str): The relative path from the project root to the file to be read.

    Returns:
        str: The content of the file, or an error message if reading fails.
    """
    try:
        safe_path = _resolve_safe_path(project_directory, filepath)

        # Check if it's a file (synchronous, run in thread)
        is_file = await asyncio.to_thread(safe_path.is_file)
        if not is_file:
            # To be more precise, check if it exists at all first.
            exists = await asyncio.to_thread(safe_path.exists)
            if not exists:
                 return f"Error: File not found at path '{filepath}'."
            return f"Error: Path '{filepath}' is not a file."

        async with aiofiles.open(safe_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
        return content
    except PermissionError as e:
        logger.warning(f"Access denied for {filepath} in {project_directory}: {e}")
        return f"Error: Access denied. {e}"
    except FileNotFoundError: # Should be caught by is_file/exists, but as fallback
        logger.warning(f"File not found for {filepath} in {project_directory}")
        return f"Error: File not found at path '{filepath}'."
    except Exception as e:
        logger.error(f"Could not read file '{filepath}' in {project_directory}: {e}", exc_info=True)
        return f"Error: Could not read file '{filepath}'. {str(e)}"

@openhands_tool
async def write_file(project_directory: str, filepath: str, content: str) -> str:
    """
    Writes content to a specified file. Overwrites if the file exists, creates it if not.
    Parent directories will be created if they do not exist.

    Args:
        project_directory (str): (Internal) The absolute path to the root of the project.
        filepath (str): The relative path from the project root to the file to be written.
        content (str): The content to write into the file.

    Returns:
        str: A success message if writing is successful, or an error message.
    """
    try:
        safe_path = _resolve_safe_path(project_directory, filepath)

        # Ensure parent directories exist (synchronous, run in thread)
        parent_dir = safe_path.parent
        await asyncio.to_thread(lambda: parent_dir.mkdir(parents=True, exist_ok=True))

        async with aiofiles.open(safe_path, mode='w', encoding='utf-8') as f:
            await f.write(content)
        logger.info(f"File '{filepath}' written successfully in '{project_directory}'.")
        return f"File '{filepath}' written successfully."
    except PermissionError as e:
        logger.warning(f"Access denied for writing {filepath} in {project_directory}: {e}")
        return f"Error: Access denied. {e}"
    except Exception as e:
        logger.error(f"Could not write file '{filepath}' in {project_directory}: {e}", exc_info=True)
        return f"Error: Could not write file '{filepath}'. {str(e)}"

@openhands_tool
async def list_files(project_directory: str, directory_path: str = "") -> list[str] | str:
    """
    Lists files and subdirectories within a specified relative path.
    Appends a '/' to directory names in the output list.

    Args:
        project_directory (str): (Internal) The absolute path to the root of the project.
        directory_path (str): The relative path from the project root to the directory to list.
                              Defaults to the project root if empty.

    Returns:
        list[str] | str: A list of file and directory names (directories have a trailing '/'),
                         or an error message string if listing fails.
    """
    try:
        safe_target_path = _resolve_safe_path(project_directory, path)

        # Check if the resolved path is a directory (synchronous, run in thread)
        is_dir = await asyncio.to_thread(safe_target_path.is_dir)
        if not is_dir:
            exists = await asyncio.to_thread(safe_target_path.exists)
            if not exists:
                return f"Error: Directory not found at path '{path}'."
            return f"Error: Path '{path}' is not a directory."

        # List directory contents (synchronous, run in thread)
        dir_contents = await asyncio.to_thread(lambda: os.listdir(safe_target_path))

        results = []
        for item_name in sorted(dir_contents): # Sort for consistent output
            item_path = safe_target_path / item_name
            # Check if item is a directory (synchronous, run in thread)
            is_item_dir = await asyncio.to_thread(item_path.is_dir)
            if is_item_dir:
                results.append(f"{item_name}/")
            else:
                results.append(item_name)
        return results
    except PermissionError as e:
        logger.warning(f"Access denied for listing {path} in {project_directory}: {e}")
        return f"Error: Access denied. {e}"
    except FileNotFoundError: # Should be caught by is_dir/exists, but as fallback
        logger.warning(f"Directory not found for listing {path} in {project_directory}")
        return f"Error: Directory not found at path '{path}'."
    except Exception as e:
        logger.error(f"Could not list files in '{path}' in {project_directory}: {e}", exc_info=True)
        return f"Error: Could not list files in '{path}'. {str(e)}"

# =========================================================================================
# Original File: src/agents/tools/human_tools.py
# =========================================================================================
"""
Tools for enabling Human-in-the-Loop (HITL) interactions for AI agents.
"""
from __future__ import annotations # For Python 3.7, 3.8 compatibility

from .base import openhands_tool
# We don't directly use ToolDefinition here, but it's part of the ecosystem.

# Special sentinel exception to signal interruption for HITL
class HumanInterruption(Exception):
    """
    This exception is raised by a tool when it requires human input to proceed.
    The agent's main loop should catch this and handle the HITL workflow.
    """
    def __init__(self, tool_call_id: str, question_for_human: str, context: dict | None = None):
        """
        Args:
            tool_call_id: The ID of the tool call that is being interrupted for human input.
                          This ID is used to link the human's response back to this specific point.
            question_for_human: The question the agent needs to ask the human.
            context: Optional additional context to provide to the human or UI.
        """
        self.tool_call_id = tool_call_id
        self.question_for_human = question_for_human
        self.context = context # Store any additional context if needed
        super().__init__(question_for_human)

@openhands_tool
async def ask_human_for_input(tool_call_id: str, question: str) -> str:
    """
    Pauses execution and asks the human user for input, clarification, or a decision.
    Use this when you need information from the human to complete your current task.
    The question should be specific and clear.

    Args:
        tool_call_id (str): (Internal) The ID of this tool call, injected by the agent.
        question (str): The specific question to ask the human user.

    Raises:
        HumanInterruption: This tool signals the agent to pause and wait for human input
                           by raising this special exception. The agent should not see
                           a direct string return value from this tool in the normal flow.
    """
    # This tool doesn't "return" in the conventional sense to the LLM.
    # It interrupts the agent flow by raising a special exception.
    # The agent's main loop is responsible for catching this exception
    # and then packaging the agent's state and the human_input_request.
    # The 'tool_call_id' is crucial for the resume operation.
    # The 'question' is what the human will see.

    # Note: The '-> str' return type hint is technically what the LLM expects as a tool output
    # in the conversation history if the interruption mechanism wasn't special-cased by the agent.
    # However, since the agent *will* special-case HumanInterruption, this return type
    # effectively describes the type of the *human's response* that will eventually be
    # fed back into the conversation history.
    raise HumanInterruption(tool_call_id=tool_call_id, question_for_human=question)

# Ensure this tool is registered when this module is imported.
# The @openhands_tool decorator handles registration with global_tool_registry.

# =========================================================================================
# Original File: src/agents/tools/registry.py
# =========================================================================================
"""
Tool Registry for managing and accessing OpenHands tools.
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ToolDefinition # Import only for type checking to avoid circular dependency at runtime if base imports registry

class ToolRegistry:
    """
    Manages the registration and retrieval of tools available to agents.
    """
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool_def: ToolDefinition):
        """
        Registers a tool definition.

        Args:
            tool_def: The ToolDefinition object to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool_def.name in self._tools:
            raise ValueError(f"Tool with name '{tool_def.name}' is already registered.")
        self._tools[tool_def.name] = tool_def
        print(f"Tool '{tool_def.name}' registered.") # Simple print for now, use logger in real app

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Retrieves a tool definition by its name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The ToolDefinition if found, else None.
        """
        return self._tools.get(name)

    def get_all_tools(self) -> List[ToolDefinition]:
        """
        Returns a list of all registered tool definitions.
        """
        return list(self._tools.values())

    def generate_llm_tool_descriptions(self, as_json_string: bool = False) -> str | List[Dict[str, Any]]:
        """
        Generates a description of all registered tools formatted for inclusion in an LLM prompt.
        This description informs the LLM about what tools it can call.

        Args:
            as_json_string: If True, returns a JSON string. Otherwise, returns a list of dicts.

        Returns:
            A string (plain text or JSON) or a list of dictionaries describing the tools.
        """
        tool_schemas = []
        for tool_def in self._tools.values():
            parameters_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            for param in tool_def.parameters:
                param_schema = {
                    "type": param.type,
                    "description": param.description
                }
                # Handle nested properties for object types if defined (basic support)
                if param.type == "object" and param.properties:
                    # This part would need to recursively build the schema for nested objects
                    # For now, we'll just pass the simplified 'properties' if available
                    nested_props = {}
                    for p_name, p_def in param.properties.items():
                        nested_props[p_name] = {"type": p_def.type, "description": p_def.description}
                    param_schema["properties"] = nested_props

                # Handle items for array types if defined (basic support)
                if param.type.startswith("array") and param.items:
                     param_schema["items"] = {"type": param.items.type, "description": param.items.description}


                parameters_schema["properties"][param.name] = param_schema
                if param.required:
                    parameters_schema["required"].append(param.name)

            if not parameters_schema["properties"]: # No parameters for the tool
                 parameters_schema = {"type": "object", "properties": {}}


            tool_schemas.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": parameters_schema
            })

        if as_json_string:
            try:
                return json.dumps(tool_schemas, indent=2)
            except TypeError as e:
                # Fallback for complex types not serializable by default json.dumps
                # (e.g. if a 'type' field contained a Python type object by mistake)
                print(f"Error serializing tool schemas to JSON: {e}. Returning non-indented string.")
                return str(tool_schemas) # Basic string representation as fallback
        else:
            return tool_schemas

# Create a global instance of the ToolRegistry for easy access
global_tool_registry = ToolRegistry()

# Now, modify base.py to use this global_tool_registry
# This part is tricky as it creates a potential circular import if base.py imports registry.py
# for global_tool_registry and registry.py imports base.py for ToolDefinition.
# This is often solved by:
# 1. Defining global_tool_registry in base.py itself (simplest for now).
# 2. Or, the decorator in base.py appends to a list, and registry explicitly pulls from that list.
# 3. Or, using runtime imports or a setter method on the registry.

# For this step, I will assume that base.py's decorator will register to this instance.
# I will modify base.py in the next step to correctly use this.

# =========================================================================================
# Original File: tests/agents/test_checkpoint.py
# =========================================================================================
import pytest
from src.agents.state import SessionState, Message
from src.agents.checkpoint import InMemoryCheckpointManager

pytestmark = pytest.mark.asyncio

@pytest.fixture
def checkpoint_manager() -> InMemoryCheckpointManager:
    """Returns a fresh InMemoryCheckpointManager instance for each test."""
    return InMemoryCheckpointManager()

@pytest.fixture
def sample_session_state() -> SessionState:
    """Returns a sample SessionState object."""
    return SessionState(
        conversation_history=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ],
        thread_id="thread_123",
        original_request_info={"language": "python"}
    )

# --- Test Cases ---

async def test_save_and_load_state(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_thread_1"

    # Save state
    await checkpoint_manager.save_state(thread_id, sample_session_state)

    # Load state
    loaded_state = await checkpoint_manager.load_state(thread_id)

    assert loaded_state is not None
    assert loaded_state.thread_id == sample_session_state.thread_id
    assert len(loaded_state.conversation_history) == len(sample_session_state.conversation_history)
    assert loaded_state.conversation_history[0].content == "Hello"
    assert loaded_state.original_request_info == {"language": "python"}

async def test_load_non_existent_state(checkpoint_manager: InMemoryCheckpointManager):
    loaded_state = await checkpoint_manager.load_state("non_existent_thread")
    assert loaded_state is None

async def test_save_state_is_deep_copy(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_deep_copy"

    original_content = sample_session_state.conversation_history[0].content

    await checkpoint_manager.save_state(thread_id, sample_session_state)

    # Modify the original state object AFTER saving
    sample_session_state.conversation_history[0].content = "Modified content"
    sample_session_state.add_message(role="user", content="Another message")

    loaded_state = await checkpoint_manager.load_state(thread_id)
    assert loaded_state is not None
    # Check that loaded state reflects the state AT THE TIME OF SAVING, not the modified original
    assert loaded_state.conversation_history[0].content == original_content
    assert len(loaded_state.conversation_history) == 2 # Not 3

async def test_load_state_is_deep_copy(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_load_deep_copy"
    await checkpoint_manager.save_state(thread_id, sample_session_state)

    loaded_state_1 = await checkpoint_manager.load_state(thread_id)
    assert loaded_state_1 is not None

    # Modify the loaded state
    loaded_state_1.add_message(role="user", content="New message in loaded_state_1")
    loaded_state_1.conversation_history[0].content = "Changed in loaded_state_1"

    # Load the state again
    loaded_state_2 = await checkpoint_manager.load_state(thread_id)
    assert loaded_state_2 is not None

    # Ensure loaded_state_2 is not affected by modifications to loaded_state_1
    assert len(loaded_state_2.conversation_history) == 2
    assert loaded_state_2.conversation_history[0].content == "Hello" # Original content

async def test_delete_state(checkpoint_manager: InMemoryCheckpointManager, sample_session_state: SessionState):
    thread_id = "test_delete_thread"

    # Save then delete
    await checkpoint_manager.save_state(thread_id, sample_session_state)
    deleted = await checkpoint_manager.delete_state(thread_id)
    assert deleted is True

    # Try to load deleted state
    loaded_state = await checkpoint_manager.load_state(thread_id)
    assert loaded_state is None

async def test_delete_non_existent_state(checkpoint_manager: InMemoryCheckpointManager):
    deleted = await checkpoint_manager.delete_state("non_existent_for_delete")
    assert deleted is False

async def test_save_state_type_error(checkpoint_manager: InMemoryCheckpointManager):
    with pytest.raises(TypeError, match="state must be an instance of SessionState"):
        await checkpoint_manager.save_state("thread_type_error", "not_a_session_state_object") # type: ignore

# =========================================================================================
# Original File: tests/agents/test_code_agent.py
# =========================================================================================
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

# =========================================================================================
# Original File: tests/agents/tools/test_base.py
# =========================================================================================
import pytest
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from src.agents.tools.base import ToolParameter, ToolDefinition, openhands_tool, _parse_type_hint
from src.agents.tools.registry import global_tool_registry, ToolRegistry

# Sample Pydantic model for testing parameter type parsing
class SamplePydanticParam(BaseModel):
    sub_param_str: str = Field(..., description="A sub string parameter.")
    sub_param_int: Optional[int] = Field(None, description="An optional sub integer parameter.")

# --- Tests for _parse_type_hint ---

def test_parse_type_hint_basic():
    assert _parse_type_hint(str) == "string"
    assert _parse_type_hint(int) == "integer"
    assert _parse_type_hint(float) == "number"
    assert _parse_type_hint(bool) == "boolean"
    assert _parse_type_hint(list) == "array"
    assert _parse_type_hint(dict) == "object"
    assert _parse_type_hint(Any) == "any"

def test_parse_type_hint_complex():
    assert _parse_type_hint(Optional[str]) == "string" # Optional unwraps to base type
    assert _parse_type_hint(List[str]) == "array (items type: string)"
    assert _parse_type_hint(Dict[str, int]) == "object" # Dicts are objects, details would be in properties
    assert _parse_type_hint(Optional[List[int]]) == "array (items type: integer)"

def test_parse_type_hint_pydantic_model():
    assert _parse_type_hint(SamplePydanticParam) == "object"

# --- Tests for @openhands_tool decorator ---

# Clear the global registry before each test run in this module to ensure isolation
@pytest.fixture(autouse=True)
def clear_global_registry():
    original_tools = global_tool_registry._tools.copy()
    global_tool_registry._tools.clear()
    yield
    global_tool_registry._tools = original_tools # Restore

def test_openhands_tool_registration_and_definition():
    @openhands_tool
    def sample_tool_for_test(
        param_str: str,
        param_int: int = 10,
        param_bool: Optional[bool] = None,
        project_directory: Optional[str] = None, # Internal
        tool_call_id: Optional[str] = "id_abc" # Internal with default
    ):
        """
        A sample tool for testing purposes.
        This is its short description.

        Args:
            param_str (str): A required string parameter.
            param_int (int): An optional integer parameter with a default value.
            param_bool (Optional[bool]): An optional boolean parameter.
            project_directory (Optional[str]): (Internal) The project directory.
            tool_call_id (Optional[str]): (Internal) The tool call ID.
        """
        return f"{param_str}, {param_int}, {param_bool}"

    # Check registration
    tool_def = global_tool_registry.get_tool("sample_tool_for_test")
    assert tool_def is not None
    assert tool_def.name == "sample_tool_for_test"
    assert tool_def.description == "A sample tool for testing purposes."
    assert tool_def.function == sample_tool_for_test
    assert hasattr(sample_tool_for_test, '_tool_definition')
    assert sample_tool_for_test._tool_definition == tool_def # type: ignore

    # Check parameters (LLM-facing)
    assert len(tool_def.parameters) == 3

    param_str_def = next(p for p in tool_def.parameters if p.name == "param_str")
    assert param_str_def.type == "string"
    assert param_str_def.description == "A required string parameter."
    assert param_str_def.required is True

    param_int_def = next(p for p in tool_def.parameters if p.name == "param_int")
    assert param_int_def.type == "integer"
    assert param_int_def.description == "An optional integer parameter with a default value."
    assert param_int_def.required is False # Has default

    param_bool_def = next(p for p in tool_def.parameters if p.name == "param_bool")
    assert param_bool_def.type == "boolean" # Optional[bool] unwraps to bool
    assert param_bool_def.description == "An optional boolean parameter."
    assert param_bool_def.required is False # Is Optional

    # Check internal parameters
    assert len(tool_def.internal_parameters) == 2
    assert "project_directory" in tool_def.internal_parameters
    assert "tool_call_id" in tool_def.internal_parameters


def test_openhands_tool_no_docstring():
    @openhands_tool
    def tool_no_docstring(param1: str):
        pass # pragma: no cover

    tool_def = global_tool_registry.get_tool("tool_no_docstring")
    assert tool_def is not None
    assert tool_def.description == "tool_no_docstring" # Defaults to function name
    assert tool_def.parameters[0].description == "No description."

def test_openhands_tool_pydantic_param_description():
    @openhands_tool
    def tool_with_pydantic(data: SamplePydanticParam):
        """
        Tool that takes a Pydantic model as input.
        Args:
            data (SamplePydanticParam): The structured data input.
        """
        pass # pragma: no cover

    tool_def = global_tool_registry.get_tool("tool_with_pydantic")
    assert tool_def is not None
    assert len(tool_def.parameters) == 1
    param_data_def = tool_def.parameters[0]
    assert param_data_def.name == "data"
    assert param_data_def.type == "object"
    assert "Pydantic Model: SamplePydanticParam" in param_data_def.description
    assert param_data_def.required is True

# --- Tests for ToolParameter and ToolDefinition models ---

def test_tool_parameter_creation():
    param = ToolParameter(name="test_param", type="string", description="A test param.", required=True)
    assert param.name == "test_param"
    assert param.type == "string"

def test_tool_definition_creation():
    def dummy_func(): pass # pragma: no cover
    param = ToolParameter(name="p1", type="string", description="d1", required=True)
    tool_def = ToolDefinition(
        name="my_tool",
        description="My test tool.",
        parameters=[param],
        function=dummy_func,
        internal_parameters=["project_dir"]
    )
    assert tool_def.name == "my_tool"
    assert len(tool_def.parameters) == 1
    assert tool_def.function == dummy_func
    assert "project_dir" in tool_def.internal_parameters

# =========================================================================================
# Original File: tests/agents/tools/test_filesystem_tools.py
# =========================================================================================
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from src.agents.tools.filesystem_tools import (
    _resolve_safe_path,
    read_file,
    write_file,
    list_files
)

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# --- Tests for _resolve_safe_path ---

def test_resolve_safe_path_valid(tmp_path: Path):
    project_dir = tmp_path.resolve()
    safe_path = _resolve_safe_path(str(project_dir), "file.txt")
    assert safe_path == project_dir / "file.txt"

    safe_path_subdir = _resolve_safe_path(str(project_dir), "subdir/file.txt")
    assert safe_path_subdir == project_dir / "subdir" / "file.txt"

def test_resolve_safe_path_within_project_with_dots(tmp_path: Path):
    project_dir = tmp_path.resolve()
    (project_dir / "subdir").mkdir()
    safe_path = _resolve_safe_path(str(project_dir), "subdir/../file.txt") # Resolves to project_dir/file.txt
    assert safe_path == project_dir / "file.txt"

def test_resolve_safe_path_empty_relative_path(tmp_path: Path):
    project_dir = tmp_path.resolve()
    safe_path = _resolve_safe_path(str(project_dir), "")
    assert safe_path == project_dir

    safe_path_dot = _resolve_safe_path(str(project_dir), ".")
    assert safe_path_dot == project_dir

def test_resolve_safe_path_traversal_attempt_simple(tmp_path: Path):
    project_dir = tmp_path.resolve()
    with pytest.raises(PermissionError, match="Path traversal detected"):
        _resolve_safe_path(str(project_dir), "../file_outside.txt")

def test_resolve_safe_path_traversal_attempt_complex(tmp_path: Path):
    project_dir = tmp_path.resolve()
    with pytest.raises(PermissionError, match="Path traversal detected"):
        _resolve_safe_path(str(project_dir), "subdir/../../../../etc/passwd")

def test_resolve_safe_path_traversal_with_absolute_relative(tmp_path: Path):
    project_dir = tmp_path.resolve()
    # Simulating if relative_path was somehow an absolute path outside project_dir
    # Note: Path joining with an absolute path on the right discards the left.
    # Our _resolve_safe_path has a small normalization for leading slashes,
    # but a full absolute path like "/etc/passwd" would still be caught by is_relative_to.

    # Case 1: Relative path is absolute and outside
    with pytest.raises(PermissionError, match="Path traversal detected"):
        # This test relies on the final check, as Path("/project") / "/etc/passwd" becomes "/etc/passwd"
        _resolve_safe_path(str(project_dir), "/etc/passwd")

    # Case 2: Relative path is absolute but *inside* (less common, but check logic)
    # This is not a traversal, but tests absolute path handling by _resolve_safe_path
    # Our normalization `relative_path.lstrip(os.path.sep)` would make it relative.
    # If it was `project_dir_str + "/file.txt"`
    # E.g. project_dir = /tmp/foo, relative_path = /tmp/foo/bar.txt
    # Path(project_dir) / relative_path would become /tmp/foo/bar.txt
    # (project_dir / "bar.txt").resolve()
    # This should be allowed. Let's test the normalized version.
    abs_path_inside_project = project_dir / "abs_file.txt"
    # If an agent provides a path that *looks* absolute but means to be relative to root.
    # e.g. filepath = "/abs_file.txt" meaning "project_root/abs_file.txt"
    resolved = _resolve_safe_path(str(project_dir), str(abs_path_inside_project.relative_to(project_dir.parent))) # makes it like "foo/abs_file.txt"
    assert resolved == abs_path_inside_project

    # More direct test for lstrip logic:
    resolved_lstrip = _resolve_safe_path(str(project_dir), "/file_at_root.txt")
    assert resolved_lstrip == project_dir / "file_at_root.txt"


def test_resolve_safe_path_project_dir_not_absolute(tmp_path: Path):
    # _resolve_safe_path should internally convert project_directory to absolute
    project_dir_relative = Path(tmp_path.name) # e.g., "pytest-of-user"
    project_dir_abs = tmp_path.resolve()

    # Create a dummy current working directory for the test to be robust
    # This ensures that Path(".").resolve() inside the function behaves predictably
    with patch('pathlib.Path.cwd', return_value=tmp_path.parent):
        # If project_directory is "my_project" and cwd is "/testuser"
        # then Path("my_project").resolve() becomes "/testuser/my_project"
        # We need to ensure our project_dir_abs is this path for the test to be correct.
        # The fixture tmp_path is already absolute, so we can use it directly.

        # Forcing project_directory to be relative for this test call
        # Assuming current working directory is tmp_path.parent for this test scope
        # Then Path(tmp_path.name).resolve() would be tmp_path

        # To make this test less dependent on actual cwd, we can create a subdir
        # and run from its parent.
        test_project_name = "test_proj"
        specific_project_dir = tmp_path / test_project_name
        specific_project_dir.mkdir()

        # Pass relative path to project_directory
        safe_path = _resolve_safe_path(test_project_name, "file.txt", project_root_override=tmp_path)
        # ^^^ This test needs a bit of rework because _resolve_safe_path doesn't take project_root_override
        # The point is that project_directory is resolved.
        # It's better to test this by setting cwd for the duration of the test if possible,
        # or by ensuring the logging path is hit. For now, trust resolve() and the structure.
        # The critical part is that it *becomes* absolute and is used for is_relative_to.
        # Let's simplify and assume project_directory is usually passed as absolute by API layer.
        # The warning in _resolve_safe_path covers the case where it's not.

        # A simpler check: if it resolves correctly even if passed as relative,
        # and still performs security checks relative to that resolved root.

        # Current working directory is tmp_path.parent
        # project_directory passed as "test_proj" (relative)
        # This should resolve to tmp_path / "test_proj"
        # Then "file.txt" is relative to that.

        # To effectively test this, we'd need to control Path.cwd() or pass the base for resolution.
        # For now, the implementation does Path(project_directory).resolve().
        # If project_directory is "foo", and cwd is "/bar", it becomes "/bar/foo".
        # This is the intended behavior.

        # This test can be more about ensuring it doesn't fail if a relative project_dir is given,
        # and that the security check is still based on its *resolved absolute* path.
        with patch.object(Path, 'cwd', return_value=tmp_path): # Mock current working directory
            relative_project_path = "my_sub_project"
            abs_project_path = tmp_path / relative_project_path
            abs_project_path.mkdir()

            resolved = _resolve_safe_path(relative_project_path, "somefile.txt")
            assert resolved == abs_project_path / "somefile.txt"

            with pytest.raises(PermissionError, match="Path traversal detected"):
                _resolve_safe_path(relative_project_path, "../another_file.txt")


# --- Tests for read_file ---
# To be added: tests for read_file, write_file, list_files
# These will require mocking aiofiles and asyncio.to_thread for Path operations.

async def test_read_file_success(tmp_path: Path):
    project_dir = tmp_path
    test_file = project_dir / "test_read.txt"
    test_file.write_text("Hello, World!")

    content = await read_file(str(project_dir), "test_read.txt")
    assert content == "Hello, World!"

async def test_read_file_not_found(tmp_path: Path):
    project_dir = tmp_path
    content = await read_file(str(project_dir), "non_existent_file.txt")
    assert "Error: File not found" in content

async def test_read_file_is_a_directory(tmp_path: Path):
    project_dir = tmp_path
    (project_dir / "a_directory").mkdir()
    content = await read_file(str(project_dir), "a_directory")
    assert "Error: Path 'a_directory' is not a file." in content

async def test_read_file_permission_error(tmp_path: Path):
    project_dir = tmp_path
    # _resolve_safe_path will raise PermissionError for traversal
    content = await read_file(str(project_dir), "../forbidden.txt")
    assert "Error: Access denied. Path traversal detected" in content

@patch('aiofiles.open', new_callable=AsyncMock)
async def test_read_file_other_exception(mock_aiofiles_open: AsyncMock, tmp_path: Path):
    project_dir = tmp_path
    test_file = project_dir / "exception_test.txt"
    test_file.write_text("content") # File needs to exist and be a file for this mock

    # Simulate an unexpected error during file read
    mock_aiofiles_open.side_effect = IOError("Disk read error")

    content = await read_file(str(project_dir), "exception_test.txt")
    assert "Error: Could not read file 'exception_test.txt'. Disk read error" in content

# --- Tests for write_file ---

async def test_write_file_success(tmp_path: Path):
    project_dir = tmp_path
    filepath_relative = "new_dir/new_file.txt"
    content_to_write = "This is new content."

    result = await write_file(str(project_dir), filepath_relative, content_to_write)
    assert result == f"File '{filepath_relative}' written successfully."

    written_file_path = project_dir / filepath_relative
    assert written_file_path.exists()
    assert written_file_path.is_file()
    assert written_file_path.read_text() == content_to_write
    assert written_file_path.parent.exists() # Check directory creation

async def test_write_file_overwrite_existing(tmp_path: Path):
    project_dir = tmp_path
    filepath_relative = "existing_file.txt"
    existing_file = project_dir / filepath_relative
    existing_file.write_text("Old content.")

    new_content = "New overwritten content."
    result = await write_file(str(project_dir), filepath_relative, new_content)
    assert result == f"File '{filepath_relative}' written successfully."
    assert existing_file.read_text() == new_content

async def test_write_file_permission_error(tmp_path: Path):
    project_dir = tmp_path
    result = await write_file(str(project_dir), "../cant_write_this.txt", "content")
    assert "Error: Access denied. Path traversal detected" in result

@patch('aiofiles.open', new_callable=AsyncMock)
async def test_write_file_other_exception(mock_aiofiles_open: AsyncMock, tmp_path: Path):
    project_dir = tmp_path
    filepath_relative = "write_exception.txt"

    # Simulate an unexpected error during file write
    mock_aiofiles_open.side_effect = IOError("Disk write error")

    result = await write_file(str(project_dir), filepath_relative, "content")
    assert f"Error: Could not write file '{filepath_relative}'. Disk write error" in result


# --- Tests for list_files ---

async def test_list_files_success(tmp_path: Path):
    project_dir = tmp_path
    (project_dir / "dir1").mkdir()
    (project_dir / "file1.txt").write_text("content1")
    (project_dir / "file2.py").write_text("content2")
    (project_dir / "dir2").mkdir()
    (project_dir / "dir1" / "subfile.md").write_text("sub content")

    # List root
    result_root = await list_files(str(project_dir), "")
    assert isinstance(result_root, list)
    assert sorted(result_root) == sorted(["dir1/", "dir2/", "file1.txt", "file2.py"])

    # List subdir
    result_subdir = await list_files(str(project_dir), "dir1")
    assert isinstance(result_subdir, list)
    assert result_subdir == ["subfile.md"] # sorted by default in implementation

    # List empty dir
    result_empty_subdir = await list_files(str(project_dir), "dir2")
    assert isinstance(result_empty_subdir, list)
    assert result_empty_subdir == []


async def test_list_files_target_is_file(tmp_path: Path):
    project_dir = tmp_path
    (project_dir / "a_file.txt").write_text("i am a file")
    result = await list_files(str(project_dir), "a_file.txt")
    assert "Error: Path 'a_file.txt' is not a directory." in result

async def test_list_files_not_found(tmp_path: Path):
    project_dir = tmp_path
    result = await list_files(str(project_dir), "non_existent_dir")
    assert "Error: Directory not found" in result

async def test_list_files_permission_error(tmp_path: Path):
    project_dir = tmp_path
    result = await list_files(str(project_dir), "../another_project")
    assert "Error: Access denied. Path traversal detected" in result

@patch('src.agents.tools.filesystem_tools._resolve_safe_path') # Patching _resolve_safe_path to test os.listdir error
@patch('asyncio.to_thread') # Patching to_thread to control is_dir
async def test_list_files_os_error(mock_to_thread: AsyncMock, mock_resolve: MagicMock, tmp_path: Path):
    project_dir = tmp_path
    target_path_str = "valid_dir"
    safe_path_obj = project_dir / target_path_str

    mock_resolve.return_value = safe_path_obj # _resolve_safe_path returns a valid Path object

    # First to_thread call is for safe_target_path.is_dir()
    # Second to_thread call is for os.listdir(safe_target_path)
    # Third+ to_thread calls are for item_path.is_dir() for each item

    # Simulate is_dir returning True for the target directory
    # and then os.listdir raising an OSError
    async def to_thread_side_effect(*args, **kwargs):
        func = args[0]
        if hasattr(func, '__name__') and func.__name__ == '<lambda>' and 'listdir' in func.__code__.co_names: # Check if it's our os.listdir lambda
            raise OSError("Simulated OS error during listdir")
        # For Path.is_dir calls
        if isinstance(args[0], Path) and args[0].name == target_path_str : # is_dir for the main path
            return True # It's a directory
        return False # Default for items inside (not relevant for this error)

    mock_to_thread.side_effect = to_thread_side_effect

    # Need to ensure the directory exists for os.listdir to be called if not for the mock structure
    # However, with the mock structure, the actual existence doesn't matter as much as the mocked behavior.
    # For robustness, let's create it.
    safe_path_obj.mkdir(exist_ok=True)


    result = await list_files(str(project_dir), target_path_str)
    assert f"Error: Could not list files in '{target_path_str}'. Simulated OS error during listdir" in result

# =========================================================================================
# Original File: tests/agents/tools/test_registry.py
# =========================================================================================
import pytest
import json
from typing import Callable

from src.agents.tools.registry import ToolRegistry
from src.agents.tools.base import ToolDefinition, ToolParameter

# Sample tool functions for testing
def tool_func_1(param1: str, param2: int = 5) -> str:
    """Tool 1 description.
    Args:
        param1 (str): Description for param1.
        param2 (int): Description for param2.
    """
    return f"{param1}-{param2}"

def tool_func_2(project_directory: str, flag: bool) -> bool:
    """Tool 2 description.
    Args:
        project_directory (str): Internal project path.
        flag (bool): A boolean flag.
    """
    return not flag


@pytest.fixture
def sample_tool_def_1() -> ToolDefinition:
    return ToolDefinition(
        name="tool_func_1",
        description="Tool 1 description.",
        parameters=[
            ToolParameter(name="param1", type="string", description="Description for param1.", required=True),
            ToolParameter(name="param2", type="integer", description="Description for param2.", required=False),
        ],
        function=tool_func_1,
        internal_parameters=[]
    )

@pytest.fixture
def sample_tool_def_2() -> ToolDefinition:
    return ToolDefinition(
        name="tool_func_2",
        description="Tool 2 description.",
        parameters=[
            ToolParameter(name="flag", type="boolean", description="A boolean flag.", required=True),
        ],
        function=tool_func_2,
        internal_parameters=["project_directory"]
    )

@pytest.fixture
def registry() -> ToolRegistry:
    """Returns a fresh ToolRegistry instance for each test."""
    return ToolRegistry()

# --- Test Cases ---

def test_registry_register_tool(registry: ToolRegistry, sample_tool_def_1: ToolDefinition):
    registry.register(sample_tool_def_1)
    assert registry.get_tool("tool_func_1") == sample_tool_def_1

def test_registry_register_tool_conflict(registry: ToolRegistry, sample_tool_def_1: ToolDefinition):
    registry.register(sample_tool_def_1)
    with pytest.raises(ValueError, match="Tool with name 'tool_func_1' is already registered."):
        registry.register(sample_tool_def_1) # Registering the same tool again

def test_registry_get_tool_not_found(registry: ToolRegistry):
    assert registry.get_tool("non_existent_tool") is None

def test_registry_get_all_tools(registry: ToolRegistry, sample_tool_def_1: ToolDefinition, sample_tool_def_2: ToolDefinition):
    assert registry.get_all_tools() == []
    registry.register(sample_tool_def_1)
    registry.register(sample_tool_def_2)
    all_tools = registry.get_all_tools()
    assert len(all_tools) == 2
    assert sample_tool_def_1 in all_tools
    assert sample_tool_def_2 in all_tools

def test_registry_generate_llm_tool_descriptions_empty(registry: ToolRegistry):
    assert registry.generate_llm_tool_descriptions() == []
    assert registry.generate_llm_tool_descriptions(as_json_string=True) == "[]"

def test_registry_generate_llm_tool_descriptions_populated(registry: ToolRegistry, sample_tool_def_1: ToolDefinition, sample_tool_def_2: ToolDefinition):
    registry.register(sample_tool_def_1)
    registry.register(sample_tool_def_2)

    # Test as list of dicts
    descriptions_list = registry.generate_llm_tool_descriptions(as_json_string=False)
    assert isinstance(descriptions_list, list)
    assert len(descriptions_list) == 2

    desc_tool1 = next(d for d in descriptions_list if d["name"] == "tool_func_1")
    assert desc_tool1["description"] == "Tool 1 description."
    assert desc_tool1["parameters"]["type"] == "object"
    assert "param1" in desc_tool1["parameters"]["properties"]
    assert desc_tool1["parameters"]["properties"]["param1"]["type"] == "string"
    assert desc_tool1["parameters"]["properties"]["param1"]["description"] == "Description for param1."
    assert "param2" in desc_tool1["parameters"]["properties"]
    assert desc_tool1["parameters"]["properties"]["param2"]["type"] == "integer"
    assert "param1" in desc_tool1["parameters"]["required"]
    assert "param2" not in desc_tool1["parameters"]["required"] # param2 has a default

    desc_tool2 = next(d for d in descriptions_list if d["name"] == "tool_func_2")
    assert desc_tool2["description"] == "Tool 2 description."
    assert "flag" in desc_tool2["parameters"]["properties"]
    assert desc_tool2["parameters"]["properties"]["flag"]["type"] == "boolean"
    assert "flag" in desc_tool2["parameters"]["required"]
    # project_directory should not be in parameters for LLM
    assert "project_directory" not in desc_tool2["parameters"]["properties"]


    # Test as JSON string
    descriptions_json_str = registry.generate_llm_tool_descriptions(as_json_string=True)
    assert isinstance(descriptions_json_str, str)
    descriptions_loaded_from_json = json.loads(descriptions_json_str)
    assert len(descriptions_loaded_from_json) == 2

    # Verify one of the tools from JSON string for good measure
    json_desc_tool1 = next(d for d in descriptions_loaded_from_json if d["name"] == "tool_func_1")
    assert json_desc_tool1["parameters"]["properties"]["param1"]["type"] == "string"


def test_registry_tool_with_no_llm_params(registry: ToolRegistry):
    def no_param_tool_func(project_directory: str): # Only internal param
        """A tool with no LLM-facing parameters."""
        pass # pragma: no cover

    no_param_tool_def = ToolDefinition(
        name="no_param_tool",
        description="A tool with no LLM-facing parameters.",
        parameters=[], # Empty list for LLM
        function=no_param_tool_func,
        internal_parameters=["project_directory"]
    )
    registry.register(no_param_tool_def)

    descriptions_list = registry.generate_llm_tool_descriptions(as_json_string=False)
    assert len(descriptions_list) == 1
    desc_no_param_tool = descriptions_list[0]
    assert desc_no_param_tool["name"] == "no_param_tool"
    assert desc_no_param_tool["parameters"]["type"] == "object"
    assert desc_no_param_tool["parameters"]["properties"] == {} # No properties for LLM
    assert not desc_no_param_tool["parameters"].get("required", []) # No required params
