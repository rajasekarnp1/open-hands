"""
Data models for the LLM API Aggregator.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ProviderStatus(str, Enum):
    """Provider status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ModelCapability(str, Enum):
    """Model capability enumeration."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"


class ProviderType(str, Enum):
    """Provider type enumeration."""
    FREE = "free"
    TRIAL_CREDIT = "trial_credit"
    PAID = "paid"


class RateLimit(BaseModel):
    """Rate limit configuration."""
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    concurrent_requests: Optional[int] = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    display_name: str
    provider: str
    capabilities: List[ModelCapability]
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    cost_per_token: Optional[float] = None
    is_free: bool = True
    rate_limit: Optional[RateLimit] = None


class ProviderConfig(BaseModel):
    """Provider configuration."""
    name: str
    display_name: str
    provider_type: ProviderType
    base_url: str
    api_key_required: bool = True
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"
    models: List[ModelInfo]
    rate_limit: RateLimit
    status: ProviderStatus = ProviderStatus.ACTIVE
    priority: int = 1  # Lower number = higher priority
    supports_streaming: bool = True
    supports_function_calling: bool = False
    max_retries: int = 3
    timeout: int = 30


class AccountCredentials(BaseModel):
    """Account credentials for a provider."""
    provider: str
    account_id: str
    api_key: str
    additional_headers: Optional[Dict[str, str]] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_reset: Optional[datetime] = None


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage]
    model: str = "auto"  # "auto" for automatic selection
    provider: Optional[str] = None  # Force specific provider
    model_quality: Optional[str] = None # "fastest", "best_quality", "balanced"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[ChatMessage] = None  # For streaming
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Chat completion usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    provider: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None


class ProviderMetrics(BaseModel):
    """Provider performance metrics."""
    provider: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    rate_limit_hits: int = 0
    error_count: int = 0
    uptime_percentage: float = 100.0


class RoutingRule(BaseModel):
    """Routing rule for provider selection."""
    name: str
    conditions: Dict[str, Any]  # Conditions for applying this rule
    provider_preferences: List[str]  # Ordered list of preferred providers
    fallback_chain: List[str]  # Fallback providers if preferred ones fail
    is_active: bool = True


class SystemConfig(BaseModel):
    """System configuration."""
    default_model: str = "auto"
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    retry_attempts: int = 3
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    enable_metrics: bool = True
    log_level: str = "INFO"
    encryption_key: Optional[str] = None


class RequestLog(BaseModel):
    """Request log entry."""
    id: str
    timestamp: datetime
    provider: str
    model: str
    user_id: Optional[str] = None
    request_tokens: int
    response_tokens: int
    response_time: float
    status: str  # "success", "error", "rate_limited"
    error_message: Optional[str] = None


# Agent-related Models
class CodeAgentRequest(BaseModel):
    instruction: str
    context: Optional[str] = None  # For existing code or other context
    language: Optional[str] = None  # e.g., "python", "javascript"
    project_directory: Optional[str] = None # Base directory for filesystem tools
    model_quality: Optional[str] = None # "fastest", "best_quality", "balanced"
    provider: Optional[str] = None
    # Potentially add other ChatCompletionRequest relevant params if needed by agent's LLM call


class CodeAgentResponse(BaseModel):
    generated_code: str
    explanation: Optional[str] = None
    # Potentially include original request details or model used for traceability
    request_params: Optional[dict] = None
    model_used: Optional[str] = None