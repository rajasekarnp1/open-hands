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
    CodeAgentRequest, # New import
    CodeAgentResponse # New import
)
from pydantic import BaseModel

from ..core.aggregator import LLMAggregator
from ..agents.code_agent import CodeAgent # New import
from ..core.account_manager import AccountManager
from ..core.router import ProviderRouter
from ..core.rate_limiter import RateLimiter, RateLimitExceeded
from ..providers.openrouter import create_openrouter_provider
from ..providers.groq import create_groq_provider
from ..providers.cerebras import create_cerebras_provider


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Security configuration using centralized settings
# ADMIN_TOKEN and ALLOWED_ORIGINS are now accessed via settings object

if not settings.ADMIN_TOKEN:
    logger.warning("ADMIN_TOKEN not set in environment or .env file. Admin endpoints will be disabled if called.")

# Global instances
aggregator: Optional[LLMAggregator] = None
code_agent_instance: Optional[CodeAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global aggregator, code_agent_instance
    
    # Startup
    logger.info("Starting LLM API Aggregator and Agents...")
    
    # Initialize components
    account_manager = AccountManager() # Consider making this part of settings or a singleton
    rate_limiter = RateLimiter()
    
    # Create providers
    # TODO: Dynamically discover and load providers based on config
    providers_list = []

    # Initialize providers with empty credentials (will be populated by AccountManager)
    from ..providers.openrouter import create_openrouter_provider
    from ..providers.groq import create_groq_provider
    from ..providers.cerebras import create_cerebras_provider
    # Import Anthropic provider factory function
    from ..providers.anthropic import create_anthropic_provider

    providers_list.append(create_openrouter_provider([]))
    providers_list.append(create_groq_provider([]))
    providers_list.append(create_cerebras_provider([]))
    providers_list.append(create_anthropic_provider([])) # Add Anthropic

    # Create provider configs dict for the router
    # Assuming provider config is loaded from a central place or default is used.
    # For now, router might rely on default configs within each provider if not overridden.
    provider_configs = {provider.name: provider.config for provider in providers_list}
    
    # Initialize router
    router = ProviderRouter(provider_configs) # router needs provider configurations
    
    # Initialize aggregator
    aggregator = LLMAggregator(
        providers=providers_list,
        account_manager=account_manager,
        router=router,
        rate_limiter=rate_limiter
    )

    # Initialize CodeAgent
    code_agent_instance = CodeAgent(llm_aggregator=aggregator)
    
    logger.info("LLM API Aggregator and Agents started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM API Aggregator and Agents...")
    if aggregator:
        await aggregator.close()
    # No specific close needed for CodeAgent unless it holds resources
    logger.info("LLM API Aggregator and Agents shut down")


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
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in aggregator.chat_completion_stream(request, user_id):
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Return regular response
            response = await aggregator.chat_completion(request, user_id)
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

    if not aggregator:
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
                async for chunk in aggregator.chat_completion_stream(chat_request, user_id):
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream", # Correct media type for SSE
                headers={"Cache-Control": "no-cache"}
            )
        else:
            response = await aggregator.chat_completion(chat_request, user_id)
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