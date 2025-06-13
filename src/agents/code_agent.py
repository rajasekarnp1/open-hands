"""
Coding Agent implementation.
"""

import logging
from typing import Optional

from ..models import CodeAgentRequest, CodeAgentResponse, ChatMessage, ChatCompletionRequest
from ..core.aggregator import LLMAggregator

logger = logging.getLogger(__name__)

CODE_EXPLANATION_SEPARATOR = "###EXPLANATION###"

class CodeAgent:
    """Agent specialized for code generation tasks."""

    def __init__(self, llm_aggregator: LLMAggregator):
        self.llm_aggregator = llm_aggregator

    async def generate_code(self, request: CodeAgentRequest) -> CodeAgentResponse:
        """
        Generates code based on the user's instruction, context, and language preference.
        """
        prompt_parts = []
        prompt_parts.append("You are an expert coding assistant. Your task is to generate code based on the following instruction.")

        if request.language:
            prompt_parts.append(f"Generate the code in {request.language}.")
        else:
            prompt_parts.append("Infer the programming language from the instruction or context, or default to Python if unclear.")

        prompt_parts.append(f"\nInstruction:\n{request.instruction}")

        if request.context:
            prompt_parts.append(f"\nHere is some existing code or context to consider:\n```\n{request.context}\n```")

        prompt_parts.append(f"\nPlease provide only the generated code. If you want to add an explanation, do so after the code, separated by a line containing only '{CODE_EXPLANATION_SEPARATOR}'.")
        prompt_parts.append("Ensure the code is complete and runnable if possible within typical constraints.")

        full_prompt = "\n".join(prompt_parts)

        # Prepare ChatCompletionRequest for the aggregator
        # We might want to guide the aggregator towards a code-capable model.
        # This can be done via model_quality, specific model name, or custom routing rules.
        # For now, let's rely on existing router capabilities and potentially model_quality.

        # Determine model quality for coding tasks. "best_quality" is often preferred for code generation.
        # If the user specified a quality, use that, otherwise default to something suitable for coding.
        model_quality_for_coding = request.model_quality or "best_quality" # Default to best for code

        chat_request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=full_prompt)],
            model="auto", # Let the aggregator/router decide the specific model
            provider=request.provider, # User can specify a provider
            model_quality=model_quality_for_coding,
            # Consider adding a stop sequence for the explanation separator if models tend to overrun
            # stop=[CODE_EXPLANATION_SEPARATOR] # This might be too aggressive initially
            # Temperature might be lower for code, e.g., 0.2-0.5
            temperature=0.3, # A reasonable default for code generation
        )

        try:
            logger.debug(f"CodeAgent sending request to LLMAggregator: {chat_request.messages}")
            llm_response = await self.llm_aggregator.chat_completion(chat_request)

            generated_text = ""
            if llm_response.choices and llm_response.choices[0].message:
                generated_text = llm_response.choices[0].message.content.strip()

            logger.debug(f"CodeAgent received raw response: {generated_text}")

            generated_code = generated_text
            explanation: Optional[str] = None

            if CODE_EXPLANATION_SEPARATOR in generated_text:
                parts = generated_text.split(CODE_EXPLANATION_SEPARATOR, 1)
                generated_code = parts[0].strip()
                if len(parts) > 1:
                    explanation = parts[1].strip()

            # Remove potential backticks or language specifiers if LLM wraps code in markdown
            if generated_code.startswith("```") and generated_code.endswith("```"):
                # Remove the first line (e.g., ```python) and the last line (```)
                lines = generated_code.splitlines()
                if len(lines) > 1: # Ensure there's content between the backticks
                    generated_code = "\n".join(lines[1:-1]).strip()
                else: # Only backticks or empty
                    generated_code = ""


            return CodeAgentResponse(
                generated_code=generated_code,
                explanation=explanation,
                request_params=request.dict(), # For traceability
                model_used=llm_response.model # Get the actual model used
            )

        except Exception as e:
            logger.error(f"CodeAgent error during code generation: {e}", exc_info=True)
            # Re-raise or return an error response
            # For now, let's return a response with error message in generated_code
            return CodeAgentResponse(
                generated_code=f"Error generating code: {str(e)}",
                explanation="An error occurred while processing your request."
            )
