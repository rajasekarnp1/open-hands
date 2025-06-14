"""
FastAPI server for the LLM API Aggregator.
"""

import asyncio
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from pathlib import Path # New import for project_directory validation
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from src.config import settings # Centralized settings

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    AccountCredentials,
    ProviderConfig,
    SystemConfig,
    CodeAgentRequest,
    CodeAgentResponse,
    ResumeAgentRequest # New import
)
from pydantic import BaseModel

from ..core.aggregator import LLMAggregator
from ..agents.code_agent import CodeAgent
from ..agents.checkpoint import InMemoryCheckpointManager, BaseCheckpointManager # New imports
from ..core.account_manager import AccountManager
from ..core.router import ProviderRouter
from ..core.rate_limiter import RateLimiter, RateLimitExceeded
# Specific provider imports are now within lifespan or not needed at top level if dynamically loaded


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Security configuration using centralized settings
# ADMIN_TOKEN and ALLOWED_ORIGINS are now accessed via settings object

if not settings.ADMIN_TOKEN:
    logger.warning("ADMIN_TOKEN not set in environment or .env file. Admin endpoints will be disabled if called.")

# Global instances
aggregator_instance: Optional[LLMAggregator] = None # Renamed for clarity
code_agent_instance: Optional[CodeAgent] = None
checkpoint_manager_instance: Optional[BaseCheckpointManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global aggregator_instance, code_agent_instance, checkpoint_manager_instance
    
    logger.info("Starting application lifespan...")
    
    # Initialize components
    account_manager = AccountManager()
    rate_limiter = RateLimiter()
    checkpoint_manager_instance = InMemoryCheckpointManager() # Create checkpoint manager instance

    # Dynamically load providers (conceptual example, actual loading might be more complex)
    # This part should ideally discover all provider modules in src/providers/
    # and call their respective create_<provider_name>_provider functions.
    providers_list = []
    provider_names = ["openrouter", "groq", "cerebras", "anthropic"] # Example list
    for name in provider_names:
        try:
            module = __import__(f"src.providers.{name}", fromlist=[f"create_{name}_provider"])
            create_func = getattr(module, f"create_{name}_provider")
            providers_list.append(create_func([])) # Initialize with empty credentials
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load provider {name}: {e}")

    if not providers_list:
        logger.warning("No providers were loaded. LLM Aggregator might not function correctly.")

    provider_configs = {provider.name: provider.config for provider in providers_list}
    router = ProviderRouter(provider_configs=provider_configs) # Pass configs correctly

    aggregator_instance = LLMAggregator(
        providers=providers_list,
        account_manager=account_manager,
        router=router,
        rate_limiter=rate_limiter
    )

    # Initialize CodeAgent with the aggregator and checkpoint manager
    code_agent_instance = CodeAgent(
        llm_aggregator=aggregator_instance,
        checkpoint_manager=checkpoint_manager_instance
    )
    
    logger.info("LLM API Aggregator and Agents initialized successfully.")
    
    yield
    
    logger.info("Shutting down application lifespan...")
    if aggregator_instance:
        await aggregator_instance.close()
    logger.info("Application shut down successfully.")


# Create FastAPI app
app = FastAPI(
    title="LLM API Aggregator",
    description="Multi-provider LLM API with intelligent routing and account management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, # Use settings
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def get_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Extract user ID from authorization header."""
    if credentials:
        return credentials.credentials  # Use token as user ID for simplicity
    return None


async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify admin token for admin endpoints."""
    if not settings.ADMIN_TOKEN:
        logger.error("Attempt to access admin endpoint, but ADMIN_TOKEN is not configured.")
        raise HTTPException(status_code=503, detail="Admin functionality is not configured or disabled.")
    
    if not credentials or credentials.credentials != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    
    return credentials.credentials


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM API Aggregator",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    provider_health = await aggregator.health_check()
    
    return {
        "status": "healthy",
        "providers": provider_health,
        "timestamp": time.time()
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(get_user_id)
):
    """OpenAI-compatible chat completions endpoint."""
    
    if not aggregator_instance: # Use updated global name
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if request.stream:
            async def generate():
                async for chunk in aggregator_instance.chat_completion_stream(request, user_id): # Use updated global name
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream", # Changed to text/event-stream from text/plain
                headers={"Cache-Control": "no-cache"}
            )
        else:
            response = await aggregator_instance.chat_completion(request, user_id) # Use updated global name
            return response
            
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for the new contextual chat endpoint
class ContextualChatRequest(BaseModel):
    prompt: str
    selected_text: Optional[str] = None
    active_file_content: Optional[str] = None
    model: Optional[str] = "auto"
    provider: Optional[str] = None
    model_quality: Optional[str] = None
    stream: bool = False
    # Add other relevant parameters from ChatCompletionRequest if needed
    # e.g., max_tokens, temperature, etc.
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None # Assuming stop is List[str] based on ChatCompletionRequest


@app.post("/v1/contextual_chat/completions", response_model=ChatCompletionResponse)
async def contextual_chat_completions(
    request: ContextualChatRequest,
    background_tasks: BackgroundTasks, # Keep if needed for background tasks
    user_id: Optional[str] = Depends(get_user_id) # Keep for consistency
):
    """Endpoint for contextual chat, combining prompt with editor context."""

    if not aggregator_instance: # Use updated global name
        raise HTTPException(status_code=503, detail="Service not ready")

    full_prompt = f"{request.prompt}"
    if request.selected_text:
        full_prompt += f"\n\n## Selected Code Context:\n```\n{request.selected_text}\n```"
    if request.active_file_content:
        # Limit full file context to avoid overly large prompts, e.g., first/last N lines or a section around selection
        # For now, let's include a placeholder for this idea. A simple truncation might be:
        MAX_FILE_CONTEXT_LEN = 10000 # Characters
        file_context_to_send = request.active_file_content
        if len(file_context_to_send) > MAX_FILE_CONTEXT_LEN:
            # Basic truncation, could be smarter (e.g. keep start and end)
            file_context_to_send = request.active_file_content[:MAX_FILE_CONTEXT_LEN] + "\n... (truncated)"
        full_prompt += f"\n\n## Full File Context (partial):\n```\n{file_context_to_send}\n```"

    chat_request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=full_prompt)],
        model=request.model if request.model else "auto",
        provider=request.provider,
        model_quality=request.model_quality,
        stream=request.stream,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop
        # user=user_id # Can pass user_id if your aggregator/providers use it
    )

    try:
        if chat_request.stream:
            async def generate_stream():
                async for chunk in aggregator_instance.chat_completion_stream(chat_request, user_id): # Use updated global name
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            response = await aggregator_instance.chat_completion(chat_request, user_id) # Use updated global name
            return response

    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Contextual chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/agents/code/invoke", response_model=CodeAgentResponse)
async def code_agent_invoke(
    request: CodeAgentRequest,
    user_id: Optional[str] = Depends(get_user_id) # For potential future use (e.g. user-specific limits)
):
    """Endpoint to invoke the CodeAgent."""
    if not code_agent_instance:
        raise HTTPException(status_code=503, detail="CodeAgent not ready")

    # Validate project_directory if provided
    if request.project_directory:
        logger.info(f"Received request for CodeAgent with project_directory: {request.project_directory}")
        try:
            # Security: Resolve the path to prevent certain types of manipulation,
            # though _resolve_safe_path in the tool itself is the primary defense.
            path_obj = Path(request.project_directory).resolve()
            if not path_obj.exists():
                logger.warning(f"Project directory '{request.project_directory}' (resolved: '{path_obj}') does not exist.")
                raise HTTPException(status_code=400, detail=f"Project directory '{request.project_directory}' does not exist.")
            if not path_obj.is_dir():
                logger.warning(f"Project directory '{request.project_directory}' (resolved: '{path_obj}') is not a directory.")
                raise HTTPException(status_code=400, detail=f"Project directory '{request.project_directory}' is not a directory.")

            # Update request.project_directory to the resolved, absolute path for consistency downstream
            # This ensures that the agent and tools always work with a verified absolute path.
            request.project_directory = str(path_obj)
            logger.info(f"Validated project_directory: {request.project_directory}")

        except SecurityException as se: # Assuming Path.resolve() might raise SecurityException for some odd paths
            logger.error(f"Security exception resolving project directory '{request.project_directory}': {se}")
            raise HTTPException(status_code=400, detail=f"Invalid project directory path (security concern): {request.project_directory}")
        except Exception as path_e: # Catch other potential path errors during validation
            logger.error(f"Error validating project directory '{request.project_directory}': {path_e}")
            raise HTTPException(status_code=400, detail=f"Error validating project directory: {str(path_e)}")

    try:
        response = await code_agent_instance.generate_code(request)
        return response
    except Exception as e:
        logger.error(f"CodeAgent invocation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/agents/resume", response_model=CodeAgentResponse)
async def resume_agent_execution(
    resume_request: ResumeAgentRequest,
    user_id: Optional[str] = Depends(get_user_id) # For consistency and potential future use
):
    """Endpoint to resume CodeAgent execution after human input."""
    if not code_agent_instance or not checkpoint_manager_instance:
        logger.error("CodeAgent or CheckpointManager not initialized for resume.")
        raise HTTPException(status_code=503, detail="Agent components not ready.")

    logger.info(f"Resuming agent for thread_id: {resume_request.thread_id}, tool_call_id: {resume_request.tool_call_id}")

    # 1. Load state
    current_session_state = await checkpoint_manager_instance.load_state(resume_request.thread_id)
    if not current_session_state:
        logger.warning(f"No session state found for thread_id: {resume_request.thread_id} on resume.")
        raise HTTPException(status_code=404, detail=f"No session state found for thread_id: {resume_request.thread_id}")

    # 2. Add human response as a "tool" message to history
    # The 'name' should match the tool that asked for human input, typically 'ask_human_for_input'.
    # The 'tool_call_id' must match the ID of the tool call that was interrupted.
    current_session_state.add_message(
        role="tool",
        name="ask_human_for_input", # Assuming this was the tool name that paused
        content=resume_request.human_response,
        tool_call_id=resume_request.tool_call_id
    )

    # 3. Save updated state
    await checkpoint_manager_instance.save_state(resume_request.thread_id, current_session_state)

    # 4. Re-construct the original CodeAgentRequest parameters for the agent to continue
    # The agent's generate_code method will load the state (including the new human response)
    # and continue the loop. The original instruction is already part of the history.
    # We need to provide other parameters like project_directory, model_quality, etc.
    # These are now stored in current_session_state.original_request_info.

    if not current_session_state.original_request_info:
        logger.error(f"Original request info not found in session state for thread_id: {resume_request.thread_id}")
        raise HTTPException(status_code=500, detail="Internal error: Missing original request info for resumption.")

    # Create a new CodeAgentRequest for the resumption call.
    # The 'instruction' and 'context' are not strictly needed here as the conversation
    # history drives the resumption, but they are required fields in CodeAgentRequest.
    # We can use placeholders or retrieve them if stored separately (currently not).
    # For now, the agent's _construct_initial_user_message_content is only called for new threads.
    # When resuming, it adds the *new user message* which isn't what we want here.
    # The generate_code method needs to be robust to being called with a history
    # that already includes the initial user instruction.

    # The most important part is the thread_id and other parameters that affect agent execution.
    # The `instruction` in this resumed request will be added to history by `generate_code`
    # if `is_new_thread` is true, which it won't be here. So we can pass a generic one.
    # The `code_agent.generate_code` will load the history which now includes the human's answer.

    original_params = current_session_state.original_request_info
    resumed_agent_request = CodeAgentRequest(
        instruction="Continue based on human feedback.", # This instruction is for this specific call, history is main driver
        project_directory=original_params.get("project_directory"),
        thread_id=resume_request.thread_id,
        model_quality=original_params.get("model_quality"),
        provider=original_params.get("provider"),
        language=original_params.get("language") # Ensure language is also carried over if set
        # Other fields from original_request_info could be added here too
    )

    logger.debug(f"Calling generate_code for resumption with request: {resumed_agent_request.model_dump(exclude_none=True)}")
    try:
        # The agent will load the state (which includes the new human response) and continue.
        response = await code_agent_instance.generate_code(resumed_agent_request)
        return response
    except Exception as e:
        logger.error(f"CodeAgent resumption error for thread_id {resume_request.thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models across all providers."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        models_by_provider = await aggregator.list_available_models()
        
        # Flatten into OpenAI-compatible format
        models = []
        for provider_name, provider_models in models_by_provider.items():
            for model in provider_models:
                models.append({
                    "id": f"{provider_name}/{model.name}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider_name,
                    "permission": [],
                    "root": model.name,
                    "parent": None,
                    "display_name": model.display_name,
                    "capabilities": [cap.value for cap in model.capabilities],
                    "context_length": model.context_length,
                    "is_free": model.is_free
                })
        
        return {"object": "list", "data": models}
        
    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/credentials")
async def add_credentials(
    provider: str,
    account_id: str,
    api_key: str,
    additional_headers: Optional[Dict[str, str]] = None,
    _admin_token: str = Depends(verify_admin_token)
):
    """Add API credentials for a provider."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        credentials = await aggregator.account_manager.add_credentials(
            provider=provider,
            account_id=account_id,
            api_key=api_key,
            additional_headers=additional_headers
        )
        
        return {
            "message": f"Credentials added for {provider}:{account_id}",
            "provider": provider,
            "account_id": account_id
        }
        
    except Exception as e:
        logger.error(f"Add credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/credentials")
async def list_credentials(_admin_token: str = Depends(verify_admin_token)):
    """List all credentials (without sensitive data)."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        credentials = await aggregator.account_manager.list_credentials()
        return credentials
        
    except Exception as e:
        logger.error(f"List credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/credentials/{provider}/{account_id}")
async def remove_credentials(provider: str, account_id: str, _admin_token: str = Depends(verify_admin_token)):
    """Remove credentials for a specific account."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        removed = await aggregator.account_manager.remove_credentials(provider, account_id)
        
        if removed:
            return {"message": f"Credentials removed for {provider}:{account_id}"}
        else:
            raise HTTPException(status_code=404, detail="Credentials not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/providers")
async def get_provider_status(_admin_token: str = Depends(verify_admin_token)):
    """Get status of all providers."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        status = await aggregator.get_provider_status()
        return status
        
    except Exception as e:
        logger.error(f"Get provider status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/rate-limits")
async def get_rate_limit_status(user_id: Optional[str] = Depends(get_user_id), _admin_token: str = Depends(verify_admin_token)):
    """Get rate limit status."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        status = aggregator.rate_limiter.get_rate_limit_status(user_id)
        return status
        
    except Exception as e:
        logger.error(f"Get rate limit status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/usage-stats")
async def get_usage_stats(_admin_token: str = Depends(verify_admin_token)):
    """Get usage statistics."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Get account usage stats
        account_stats = await aggregator.account_manager.get_usage_stats()
        
        # Get rate limiter stats
        rate_limit_stats = aggregator.rate_limiter.get_user_stats()
        
        # Get provider scores
        provider_scores = aggregator.router.get_provider_scores()
        
        return {
            "account_usage": account_stats,
            "rate_limits": rate_limit_stats,
            "provider_scores": provider_scores,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Get usage stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/rotate-credentials/{provider}")
async def rotate_credentials(provider: str, _admin_token: str = Depends(verify_admin_token)):
    """Rotate credentials for a provider."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        await aggregator.account_manager.rotate_credentials(provider)
        return {"message": f"Credentials rotated for {provider}"}
        
    except Exception as e:
        logger.error(f"Rotate credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the server."""
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.LOG_LEVEL.lower() # Use settings for log level
    )


if __name__ == "__main__":
    # Basic logging configuration for startup messages
    logging.basicConfig(level=settings.LOG_LEVEL.upper())
    run_server()