"""
FastAPI server for the LLM API Aggregator.
"""

import asyncio
import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
import uvicorn
import psutil

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    AccountCredentials,
    ProviderConfig,
    SystemConfig
)

# Enhanced WebUI models
from pydantic import BaseModel
from typing import Union

class ImprovementRequest(BaseModel):
    mode: str
    areas: List[str]

class ModelConfigRequest(BaseModel):
    primaryProvider: str
    fallbackStrategy: str

class StabilityRequest(BaseModel):
    level: str
    autoRollback: bool

class PerformanceRequest(BaseModel):
    cpuLimit: int
    memoryLimit: int
    maxConcurrent: int

class IdleSettingsRequest(BaseModel):
    systemIdle: bool
    vmIdle: bool
    acPower: bool
    lightningVM: bool

class AccountAddRequest(BaseModel):
    provider: str
    apiKey: str

class RotationRequest(BaseModel):
    interval: int

class SystemMetrics(BaseModel):
    cpu: Union[float, str]
    memory: str
    temperature: Union[float, str]
    uptime: str
from ..core.aggregator import LLMAggregator
from ..core.account_manager import AccountManager
from ..core.router import ProviderRouter
from ..core.rate_limiter import RateLimiter, RateLimitExceeded
from ..providers.openrouter import create_openrouter_provider
from ..providers.groq import create_groq_provider
from ..providers.cerebras import create_cerebras_provider


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# Security configuration
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

if not ADMIN_TOKEN:
    logger.warning("ADMIN_TOKEN not set. Admin endpoints will be disabled.")

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

# Add CORS middleware with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin functionality disabled")
    
    if not credentials or credentials.credentials != ADMIN_TOKEN:
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
    _: str = Depends(verify_admin_token)
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
async def list_credentials(_: str = Depends(verify_admin_token)):
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
async def remove_credentials(provider: str, account_id: str, _: str = Depends(verify_admin_token)):
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
async def get_provider_status(_: str = Depends(verify_admin_token)):
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
async def get_usage_stats(_: str = Depends(verify_admin_token)):
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
async def rotate_credentials(provider: str, _: str = Depends(verify_admin_token)):
    """Rotate credentials for a provider."""
    
    if not aggregator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        await aggregator.account_manager.rotate_credentials(provider)
        return {"message": f"Credentials rotated for {provider}"}
        
    except Exception as e:
        logger.error(f"Rotate credentials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced WebUI Endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_webui():
    """Serve the enhanced WebUI."""
    webui_path = os.path.join(os.path.dirname(__file__), "..", "..", "webui", "index.html")
    if os.path.exists(webui_path):
        return FileResponse(webui_path)
    else:
        return HTMLResponse("""
        <html>
            <head><title>OpenHands Enhanced</title></head>
            <body>
                <h1>OpenHands Enhanced</h1>
                <p>WebUI not found. Please run the setup script to install the enhanced WebUI.</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """)

@app.post("/api/v1/improvement/start")
async def start_improvement(request: ImprovementRequest):
    """Start the improvement process."""
    try:
        # Implementation for starting improvement process
        logger.info(f"Starting improvement with mode: {request.mode}, areas: {request.areas}")
        
        # Here you would integrate with the self-improvement system
        # For now, return a success response
        return {
            "status": "started",
            "mode": request.mode,
            "areas": request.areas,
            "message": "Improvement process initiated successfully"
        }
    except Exception as e:
        logger.error(f"Start improvement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/models/config")
async def update_model_config(request: ModelConfigRequest):
    """Update model configuration."""
    try:
        if not aggregator:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Update router configuration
        aggregator.router.set_primary_provider(request.primaryProvider)
        aggregator.router.set_fallback_strategy(request.fallbackStrategy)
        
        return {
            "status": "updated",
            "primaryProvider": request.primaryProvider,
            "fallbackStrategy": request.fallbackStrategy
        }
    except Exception as e:
        logger.error(f"Update model config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/system/stability")
async def update_stability(request: StabilityRequest):
    """Update system stability level."""
    try:
        # Update stability configuration
        os.environ["STABILITY_LEVEL"] = request.level
        os.environ["AUTO_ROLLBACK"] = str(request.autoRollback).lower()
        
        return {
            "status": "updated",
            "level": request.level,
            "autoRollback": request.autoRollback
        }
    except Exception as e:
        logger.error(f"Update stability error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/system/performance")
async def apply_performance(request: PerformanceRequest):
    """Apply performance settings."""
    try:
        # Update performance configuration
        os.environ["CPU_LIMIT"] = str(request.cpuLimit)
        os.environ["MEMORY_LIMIT"] = str(request.memoryLimit)
        os.environ["MAX_CONCURRENT"] = str(request.maxConcurrent)
        
        if aggregator:
            # Update rate limiter if available
            aggregator.rate_limiter.max_requests = request.maxConcurrent
        
        return {
            "status": "applied",
            "cpuLimit": request.cpuLimit,
            "memoryLimit": request.memoryLimit,
            "maxConcurrent": request.maxConcurrent
        }
    except Exception as e:
        logger.error(f"Apply performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/system/idle")
async def update_idle_settings(request: IdleSettingsRequest):
    """Update idle improvement settings."""
    try:
        # Update idle configuration
        os.environ["SYSTEM_IDLE"] = str(request.systemIdle).lower()
        os.environ["VM_IDLE"] = str(request.vmIdle).lower()
        os.environ["AC_POWER"] = str(request.acPower).lower()
        os.environ["LIGHTNING_VM"] = str(request.lightningVM).lower()
        
        return {
            "status": "updated",
            "systemIdle": request.systemIdle,
            "vmIdle": request.vmIdle,
            "acPower": request.acPower,
            "lightningVM": request.lightningVM
        }
    except Exception as e:
        logger.error(f"Update idle settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/accounts/add")
async def add_account(request: AccountAddRequest):
    """Add a new API account."""
    try:
        if not aggregator:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Add account to account manager
        credentials = AccountCredentials(
            provider=request.provider,
            api_key=request.apiKey
        )
        
        await aggregator.account_manager.add_credentials(request.provider, credentials)
        
        return {
            "status": "added",
            "provider": request.provider,
            "message": f"Account added for {request.provider}"
        }
    except Exception as e:
        logger.error(f"Add account error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/accounts/rotation")
async def update_rotation(request: RotationRequest):
    """Update account rotation settings."""
    try:
        if not aggregator:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Update rotation interval
        aggregator.account_manager.rotation_interval = request.interval * 60  # Convert to seconds
        
        return {
            "status": "updated",
            "interval": request.interval,
            "message": f"Rotation interval set to {request.interval} minutes"
        }
    except Exception as e:
        logger.error(f"Update rotation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get real-time system metrics."""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Get temperature if available
        temperature = 0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        temperature = entries[0].current
                        break
        except:
            temperature = 42  # Default value
        
        # Calculate uptime
        uptime_seconds = time.time() - psutil.boot_time()
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{uptime_hours:02d}:{uptime_minutes:02d}"
        
        return SystemMetrics(
            cpu=f"{cpu_percent:.1f}",
            memory=f"{memory.used / (1024**3):.1f}GB",
            temperature=f"{temperature:.0f}",
            uptime=uptime_str
        )
    except Exception as e:
        logger.error(f"Get metrics error: {e}")
        # Return default values on error
        return SystemMetrics(
            cpu="--",
            memory="--",
            temperature="--",
            uptime="--:--"
        )

@app.get("/api/v1/logs")
async def get_logs():
    """Get system logs."""
    try:
        log_file = os.getenv("LOG_FILE", "logs/openhands_enhanced.log")
        if os.path.exists(log_file):
            return FileResponse(log_file, media_type="text/plain")
        else:
            return {"message": "Log file not found"}
    except Exception as e:
        logger.error(f"Get logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/config/export")
async def export_config():
    """Export current configuration."""
    try:
        config = {
            "install_type": os.getenv("INSTALL_TYPE", "local"),
            "stability_level": os.getenv("STABILITY_LEVEL", "stable"),
            "performance_level": os.getenv("PERFORMANCE_LEVEL", "70"),
            "features": {
                "idle_improvement": os.getenv("ENABLE_IDLE_IMPROVEMENT", "false"),
                "vm_mode": os.getenv("ENABLE_VM_MODE", "false"),
                "scifi_ui": os.getenv("ENABLE_SCIFI_UI", "false"),
                "multi_api": os.getenv("ENABLE_MULTI_API", "true")
            },
            "performance": {
                "cpu_limit": os.getenv("CPU_LIMIT", "70"),
                "memory_limit": os.getenv("MEMORY_LIMIT", "4"),
                "max_concurrent": os.getenv("MAX_CONCURRENT", "10")
            }
        }
        
        return config
    except Exception as e:
        logger.error(f"Export config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/system/restart")
async def restart_system():
    """Restart the OpenHands system."""
    try:
        # In a real implementation, you would gracefully restart the service
        # For now, just return a success message
        return {"message": "System restart initiated", "status": "success"}
    except Exception as e:
        logger.error(f"Restart system error: {e}")
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