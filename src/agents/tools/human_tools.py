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
```
