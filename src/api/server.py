"""
FastAPI server for the LLM API Aggregator.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    AccountCredentials,
    ProviderConfig,
    SystemConfig
)
from ..core.aggregator import LLMAggregator
from ..core.account_manager import AccountManager
from ..core.router import ProviderRouter
from ..core.rate_limiter import RateLimiter, RateLimitExceeded
from ..providers.openrouter import create_openrouter_provider
from ..providers.groq import create_groq_provider
from ..providers.cerebras import create_cerebras_provider


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Global aggregator instance
aggregator: Optional[LLMAggregator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global aggregator
    
    # Startup
    logger.info("Starting LLM API Aggregator...")
    
    # Initialize components
    account_manager = AccountManager()
    rate_limiter = RateLimiter()
    
    # Create providers
    providers = []
    
    # Initialize providers with empty credentials (will be added via API)
    openrouter = create_openrouter_provider([])
    groq = create_groq_provider([])
    cerebras = create_cerebras_provider([])
    
    providers.extend([openrouter, groq, cerebras])
    
    # Create provider configs dict
    provider_configs = {provider.name: provider.config for provider in providers}
    
    # Initialize router
    router = ProviderRouter(provider_configs)
    
    # Initialize aggregator
    aggregator = LLMAggregator(
        providers=providers,
        account_manager=account_manager,
        router=router,
        rate_limiter=rate_limiter
    )
    
    logger.info("LLM API Aggregator started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM API Aggregator...")
    if aggregator:
        await aggregator.close()
    logger.info("LLM API Aggregator shut down")


# Create FastAPI app
app = FastAPI(
    title="LLM API Aggregator",
    description="Multi-provider LLM API with intelligent routing and account management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Extract user ID from authorization header."""
    if credentials:
        return credentials.credentials  # Use token as user ID for simplicity
    return None


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
    additional_headers: Optional[Dict[str, str]] = None
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
async def list_credentials():
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
async def remove_credentials(provider: str, account_id: str):
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
async def get_provider_status():
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
async def get_rate_limit_status(user_id: Optional[str] = Depends(get_user_id)):
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
async def get_usage_stats():
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
async def rotate_credentials(provider: str):
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
        log_level="info"
    )


if __name__ == "__main__":
    run_server()