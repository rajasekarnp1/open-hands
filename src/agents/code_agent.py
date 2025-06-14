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

```
