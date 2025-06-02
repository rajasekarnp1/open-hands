"""
Base provider class and interfaces.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ProviderConfig,
    AccountCredentials,
    ProviderMetrics,
)


class ProviderError(Exception):
    """Base provider error."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Authentication error."""
    pass


class ModelNotFoundError(ProviderError):
    """Model not found error."""
    pass


class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: ProviderConfig, credentials: List[AccountCredentials]):
        self.config = config
        self.credentials = credentials
        self.current_credential_index = 0
        self.metrics = ProviderMetrics(provider=config.name)
        self._client = httpx.AsyncClient(timeout=config.timeout)
        self._rate_limiter = asyncio.Semaphore(config.rate_limit.concurrent_requests or 10)
    
    @property
    def name(self) -> str:
        """Get provider name."""
        return self.config.name
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return (
            self.config.status.value == "active" and
            len(self.credentials) > 0 and
            any(cred.is_active for cred in self.credentials)
        )
    
    def get_current_credentials(self) -> Optional[AccountCredentials]:
        """Get current active credentials."""
        active_creds = [cred for cred in self.credentials if cred.is_active]
        if not active_creds:
            return None
        
        # Rotate credentials to distribute load
        cred = active_creds[self.current_credential_index % len(active_creds)]
        self.current_credential_index = (self.current_credential_index + 1) % len(active_creds)
        return cred
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        for model in self.config.models:
            if model.name == model_name:
                return model
        return None
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        return self.config.models
    
    @abstractmethod
    async def chat_completion(
        self, 
        request: ChatCompletionRequest,
        credentials: AccountCredentials
    ) -> ChatCompletionResponse:
        """Perform chat completion."""
        pass
    
    @abstractmethod
    async def chat_completion_stream(
        self, 
        request: ChatCompletionRequest,
        credentials: AccountCredentials
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Perform streaming chat completion."""
        pass
    
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> httpx.Response:
        """Make HTTP request with error handling."""
        async with self._rate_limiter:
            start_time = time.time()
            
            try:
                if stream:
                    response = await self._client.stream(
                        method, url, headers=headers, json=data
                    )
                else:
                    response = await self._client.request(
                        method, url, headers=headers, json=data
                    )
                
                response_time = time.time() - start_time
                self._update_metrics(response_time, response.status_code)
                
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    raise RateLimitError(
                        f"Rate limit exceeded for {self.name}",
                        retry_after=int(retry_after) if retry_after else None
                    )
                elif response.status_code == 401:
                    raise AuthenticationError(f"Authentication failed for {self.name}")
                elif response.status_code >= 400:
                    error_text = await response.aread() if hasattr(response, 'aread') else response.text
                    raise ProviderError(f"HTTP {response.status_code}: {error_text}")
                
                return response
                
            except httpx.TimeoutException:
                self._update_metrics(time.time() - start_time, 408)
                raise ProviderError(f"Request timeout for {self.name}")
            except httpx.RequestError as e:
                self._update_metrics(time.time() - start_time, 500)
                raise ProviderError(f"Request error for {self.name}: {str(e)}")
    
    def _update_metrics(self, response_time: float, status_code: int):
        """Update provider metrics."""
        self.metrics.total_requests += 1
        
        if status_code < 400:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            if status_code == 429:
                self.metrics.rate_limit_hits += 1
            else:
                self.metrics.error_count += 1
        
        # Update average response time
        total_successful = self.metrics.successful_requests
        if total_successful > 0:
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (total_successful - 1) + response_time) 
                / total_successful
            )
        
        self.metrics.last_request_time = time.time()
        
        # Calculate uptime percentage
        if self.metrics.total_requests > 0:
            self.metrics.uptime_percentage = (
                self.metrics.successful_requests / self.metrics.total_requests * 100
            )
    
    def _prepare_headers(self, credentials: AccountCredentials) -> Dict[str, str]:
        """Prepare headers for API request."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LLM-Aggregator/1.0"
        }
        
        # Add authentication header
        if credentials.api_key:
            auth_value = f"{self.config.auth_prefix} {credentials.api_key}".strip()
            headers[self.config.auth_header] = auth_value
        
        # Add any additional headers
        if credentials.additional_headers:
            headers.update(credentials.additional_headers)
        
        return headers
    
    async def health_check(self) -> bool:
        """Perform health check on the provider."""
        try:
            credentials = self.get_current_credentials()
            if not credentials:
                return False
            
            # Simple test request
            test_request = ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                model=self.config.models[0].name if self.config.models else "default",
                max_tokens=1
            )
            
            await self.chat_completion(test_request, credentials)
            return True
            
        except Exception:
            return False
    
    async def close(self):
        """Close the provider and cleanup resources."""
        await self._client.aclose()


class OpenAICompatibleProvider(BaseProvider):
    """Base class for OpenAI-compatible providers."""
    
    async def chat_completion(
        self, 
        request: ChatCompletionRequest,
        credentials: AccountCredentials
    ) -> ChatCompletionResponse:
        """Perform chat completion using OpenAI-compatible API."""
        headers = self._prepare_headers(credentials)
        
        # Prepare request data
        data = {
            "model": request.model,
            "messages": [msg.dict() for msg in request.messages],
            "stream": False
        }
        
        # Add optional parameters
        if request.max_tokens:
            data["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stop:
            data["stop"] = request.stop
        if request.presence_penalty is not None:
            data["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            data["frequency_penalty"] = request.frequency_penalty
        if request.user:
            data["user"] = request.user
        
        url = f"{self.config.base_url}/chat/completions"
        response = await self._make_request("POST", url, headers, data)
        
        response_data = response.json()
        return self._parse_chat_response(response_data)
    
    async def chat_completion_stream(
        self, 
        request: ChatCompletionRequest,
        credentials: AccountCredentials
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Perform streaming chat completion."""
        headers = self._prepare_headers(credentials)
        
        # Prepare request data
        data = {
            "model": request.model,
            "messages": [msg.dict() for msg in request.messages],
            "stream": True
        }
        
        # Add optional parameters
        if request.max_tokens:
            data["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stop:
            data["stop"] = request.stop
        
        url = f"{self.config.base_url}/chat/completions"
        
        async with self._client.stream("POST", url, headers=headers, json=data) as response:
            if response.status_code >= 400:
                error_text = await response.aread()
                raise ProviderError(f"HTTP {response.status_code}: {error_text}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk_data = eval(data_str)  # Parse JSON
                        yield self._parse_chat_response(chunk_data, is_stream=True)
                    except Exception:
                        continue  # Skip malformed chunks
    
    def _parse_chat_response(self, data: Dict[str, Any], is_stream: bool = False) -> ChatCompletionResponse:
        """Parse chat completion response."""
        from ..models import ChatCompletionChoice, ChatMessage, ChatCompletionUsage
        
        choices = []
        for choice_data in data.get("choices", []):
            if is_stream:
                delta_data = choice_data.get("delta", {})
                delta = ChatMessage(
                    role=delta_data.get("role", "assistant"),
                    content=delta_data.get("content", "")
                ) if delta_data else None
                
                choice = ChatCompletionChoice(
                    index=choice_data.get("index", 0),
                    delta=delta,
                    finish_reason=choice_data.get("finish_reason")
                )
            else:
                message_data = choice_data.get("message", {})
                message = ChatMessage(
                    role=message_data.get("role", "assistant"),
                    content=message_data.get("content", "")
                )
                
                choice = ChatCompletionChoice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason")
                )
            
            choices.append(choice)
        
        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = ChatCompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return ChatCompletionResponse(
            id=data.get("id", ""),
            created=data.get("created", int(time.time())),
            model=data.get("model", ""),
            provider=self.name,
            choices=choices,
            usage=usage
        )