"""
Coding Agent implementation with Filesystem Tool Usage.
"""

import json
import logging
import re
from typing import Optional, Tuple, List, Dict, Any

from ..models import (
    CodeAgentRequest,
    CodeAgentResponse,
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse # For _get_llm_response
)
from ..core.aggregator import LLMAggregator
from .tools import filesystem_tools # Import the new tools

logger = logging.getLogger(__name__)

CODE_EXPLANATION_SEPARATOR = "###EXPLANATION###"
MAX_TOOL_ITERATIONS = 5 # Prevent infinite loops

class CodeAgent:
    """Agent specialized for code generation tasks, with filesystem tool capabilities."""

    def __init__(self, llm_aggregator: LLMAggregator):
        self.llm_aggregator = llm_aggregator

    def _construct_initial_user_prompt(self, request: CodeAgentRequest) -> str:
        """Constructs the initial user-facing prompt including instruction, context, language."""
        prompt_parts = []
        if request.language:
            prompt_parts.append(f"Target language: {request.language}.")
        else:
            prompt_parts.append("Infer the programming language from the instruction or context, or default to Python if unclear.")

        prompt_parts.append(f"\nUser Instruction:\n{request.instruction}")

        if request.context:
            prompt_parts.append(f"\nProvided Context/Code:\n```\n{request.context}\n```")

        return "\n".join(prompt_parts)

    def _construct_system_prompt_with_tools_if_applicable(self, request: CodeAgentRequest) -> str:
        """Builds the system prompt. If request.project_directory is set, includes tool instructions."""
        base_system_prompt = (
            "You are an expert coding assistant. Your primary task is to generate, modify, or explain code based on the user's instruction.\n"
            "If you are asked to generate code, provide only the generated code block. \n"
            f"If you provide an explanation, do so after all the code, separated by a line containing only '{CODE_EXPLANATION_SEPARATOR}'.\n"
            "Ensure the code is complete and runnable if possible within typical constraints."
        )

        if not request.project_directory: # No project directory, no tools.
            return base_system_prompt

        tool_instructions = f"""

You have access to the following tools to interact with the user's project file system.
The project root directory is pre-configured for you. All filepaths MUST be relative to this project root.
Do not attempt to access files outside this project directory.

Available Tools:
1. read_file(filepath: str) -> str:
   Reads the entire content of a specified file.
   Example JSON: {{"tool_name": "read_file", "parameters": {{"filepath": "src/main.py"}}}}

2. write_file(filepath: str, content: str) -> str:
   Writes content to a specified file. Overwrites if exists, creates if not.
   Example JSON: {{"tool_name": "write_file", "parameters": {{"filepath": "docs/new_feature.md", "content": "# New Feature\\nDetails..."}}}}

3. list_files(directory_path: str = "") -> list[str] | str:
   Lists files and directories within a specified relative path. Defaults to project root.
   Example JSON: {{"tool_name": "list_files", "parameters": {{"directory_path": "src/utils"}}}}

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
        conversation_history: List[Dict[str, str]],
        request: CodeAgentRequest
    ) -> Tuple[str, Optional[ChatCompletionResponse]]:
        """Calls llm_aggregator.chat_completion and returns the text content and full response."""
        model_quality_for_coding = request.model_quality or "best_quality"

        chat_request = ChatCompletionRequest(
            messages=[ChatMessage(role=msg["role"], content=msg["content"]) for msg in conversation_history],
            model="auto",
            provider=request.provider,
            model_quality=model_quality_for_coding,
            temperature=0.3, # Suitable for coding
        )

        try:
            logger.debug(f"CodeAgent sending request to LLMAggregator with history: {conversation_history}")
            llm_response = await self.llm_aggregator.chat_completion(chat_request)

            if llm_response.choices and llm_response.choices[0].message:
                content = llm_response.choices[0].message.content.strip()
                logger.debug(f"CodeAgent received raw response content: {content}")
                return content, llm_response
            else:
                logger.warning("LLM response was empty or malformed.")
                return "Error: LLM response was empty.", None
        except Exception as e:
            logger.error(f"Error calling LLM aggregator: {e}", exc_info=True)
            return f"Error: Could not get response from LLM. {str(e)}", None


    def _parse_llm_for_tool_call(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to parse a JSON tool call from the response_text.
        The JSON should be the last significant part of the text, possibly after some reasoning.
        It might be enclosed in backticks.
        """
        logger.debug(f"Attempting to parse tool call from: {response_text}")
        # Regex to find JSON block, possibly enclosed in ```json ... ``` or ``` ... ```
        # It tries to find the JSON block that is most likely a tool call.
        # This regex looks for a JSON object, possibly with leading/trailing whitespace or markdown code fences.
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```|(\{[\s\S]*\"tool_name\"[\s\S]*\})$", response_text, re.DOTALL | re.MULTILINE)

        if match:
            json_str = match.group(1) if match.group(1) else match.group(2)
            if json_str:
                try:
                    tool_call = json.loads(json_str.strip())
                    if isinstance(tool_call, dict) and "tool_name" in tool_call and "parameters" in tool_call:
                        logger.info(f"Parsed tool call: {tool_call}")
                        return tool_call
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError parsing tool call '{json_str}': {e}")
                    # Attempt to strip potential non-JSON explanation after the JSON block if LLM didn't stop
                    # This is a basic attempt, more robust parsing might be needed for complex LLM outputs
                    try:
                        # Find the last valid JSON object in the string
                        potential_json_objects = re.findall(r'\{.*?\}', response_text)
                        if potential_json_objects:
                            for obj_str in reversed(potential_json_objects):
                                try:
                                    tool_call_candidate = json.loads(obj_str)
                                    if isinstance(tool_call_candidate, dict) and \
                                       "tool_name" in tool_call_candidate and \
                                       "parameters" in tool_call_candidate:
                                        logger.info(f"Successfully parsed tool call after cleanup: {tool_call_candidate}")
                                        return tool_call_candidate
                                except json.JSONDecodeError:
                                    continue # Try previous JSON object
                        logger.warning("No valid tool call JSON found even after regex match.")
                        return None
                    except Exception as final_e: # pylint: disable=broad-except
                        logger.error(f"Exception during final parse attempt for tool call: {final_e}")
                        return None
        logger.debug("No tool call found in LLM response.")
        return None

    def _parse_final_response(self, response_text: str) -> Tuple[str, Optional[str]]:
        """Uses existing logic to parse code and explanation (e.g., ###EXPLANATION### separator)."""
        generated_code = response_text
        explanation: Optional[str] = None

        if CODE_EXPLANATION_SEPARATOR in response_text:
            parts = response_text.split(CODE_EXPLANATION_SEPARATOR, 1)
            generated_code = parts[0].strip()
            if len(parts) > 1:
                explanation = parts[1].strip()

        # Remove potential backticks or language specifiers if LLM wraps code in markdown
        # This should be applied only to the code part
        if generated_code.startswith("```") and generated_code.endswith("```"):
            lines = generated_code.splitlines()
            if len(lines) > 1:
                generated_code = "\n".join(lines[1:-1]).strip()
            else:
                generated_code = "" # Only backticks or empty
        elif generated_code.startswith("```"): # Check if LLM forgot closing backticks
             lines = generated_code.splitlines()
             if len(lines) > 0:
                generated_code = "\n".join(lines[1:]).strip()


        return generated_code, explanation

    async def generate_code(self, request: CodeAgentRequest) -> CodeAgentResponse:
        """
        Generates code based on the user's instruction, potentially using filesystem tools
        if a project_directory is provided.
        """
        system_prompt = self._construct_system_prompt_with_tools_if_applicable(request)
        initial_user_prompt = self._construct_initial_user_prompt(request)

        conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_prompt}
        ]

        last_model_used: Optional[str] = None

        for iteration in range(MAX_TOOL_ITERATIONS):
            logger.info(f"CodeAgent Iteration {iteration + 1}/{MAX_TOOL_ITERATIONS}")

            llm_response_text, llm_full_response = await self._get_llm_response(conversation_history, request)
            if llm_full_response: # Store model from last successful LLM call
                last_model_used = llm_full_response.model

            if "Error: Could not get response from LLM" in llm_response_text: # Critical LLM error
                return CodeAgentResponse(generated_code="", explanation=llm_response_text, request_params=request.dict(), model_used=last_model_used)

            tool_call = self._parse_llm_for_tool_call(llm_response_text)

            if tool_call:
                # Append LLM's response that includes the tool call reasoning and the call itself
                conversation_history.append({"role": "assistant", "content": llm_response_text})

                tool_name = tool_call.get("tool_name")
                tool_params = tool_call.get("parameters", {})
                tool_output: str = ""

                if not request.project_directory:
                    tool_output = "Error: Project directory not specified. Cannot use file system tools."
                    logger.warning("Tool call attempted without project_directory specified.")
                elif tool_name == "read_file":
                    filepath = tool_params.get("filepath", "")
                    logger.info(f"Executing tool: read_file, params: {{'filepath': '{filepath}'}}")
                    tool_output = await filesystem_tools.read_file(request.project_directory, filepath)
                elif tool_name == "write_file":
                    filepath = tool_params.get("filepath", "")
                    content = tool_params.get("content", "")
                    logger.info(f"Executing tool: write_file, params: {{'filepath': '{filepath}', 'content_len': {len(content)}}}")
                    tool_output = await filesystem_tools.write_file(request.project_directory, filepath, content)
                elif tool_name == "list_files":
                    dir_path = tool_params.get("directory_path", "")
                    logger.info(f"Executing tool: list_files, params: {{'directory_path': '{dir_path}'}}")
                    # The tool returns list[str] | str. We need to convert list to string for history.
                    raw_tool_output = await filesystem_tools.list_files(request.project_directory, dir_path)
                    if isinstance(raw_tool_output, list):
                        tool_output = "\n".join(raw_tool_output)
                    else: # It's already an error string
                        tool_output = raw_tool_output
                else:
                    logger.warning(f"Unknown tool name: {tool_name}")
                    tool_output = f"Error: Unknown tool '{tool_name}'. Available tools are: read_file, write_file, list_files."

                logger.debug(f"Tool output for {tool_name}: {tool_output[:200]}...") # Log snippet
                conversation_history.append({"role": "tool", "content": tool_output})
            else:
                # No tool call, assume final answer from LLM
                logger.info("No tool call detected. Parsing as final response.")
                generated_code, explanation = self._parse_final_response(llm_response_text)
                return CodeAgentResponse(
                    generated_code=generated_code,
                    explanation=explanation,
                    request_params=request.dict(),
                    model_used=last_model_used
                )

        logger.warning("Agent exceeded maximum tool iterations.")
        return CodeAgentResponse(
            generated_code="Error: Agent exceeded maximum tool iterations.",
            explanation="The agent could not complete the request within the allowed number of steps.",
            request_params=request.dict(),
            model_used=last_model_used
        )
