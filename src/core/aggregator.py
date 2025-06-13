"""
Main LLM Aggregator class that orchestrates provider selection and routing.
"""

import asyncio
import logging
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ProviderConfig,
    AccountCredentials,
    RoutingRule,
    ModelCapability,
)
from ..providers.base import BaseProvider, ProviderError, RateLimitError, AuthenticationError
from .account_manager import AccountManager
from .router import ProviderRouter
from .rate_limiter import RateLimiter
from .meta_controller import MetaModelController, ModelCapabilityProfile # Removed TaskComplexityAnalyzer, not directly used here
from .ensemble_system import EnsembleSystem
from .auto_updater import AutoUpdater, integrate_auto_updater
from src.config import settings # Centralized settings


logger = logging.getLogger(__name__)


class LLMAggregator:
    """Main LLM API Aggregator class."""
    
    def __init__(
        self,
        providers: List[BaseProvider],
        account_manager: AccountManager,
        router: ProviderRouter,
        rate_limiter: RateLimiter,
        max_retries: Optional[int] = None, # Will be sourced from settings
        retry_delay: Optional[float] = None, # Will be sourced from settings
        enable_meta_controller: bool = True,
        enable_ensemble: bool = False,
        enable_auto_updater: bool = True,
        auto_update_interval: Optional[int] = None # Will be sourced from settings
    ):
        self.providers = {provider.name: provider for provider in providers}
        self.account_manager = account_manager
        self.router = router
        self.rate_limiter = rate_limiter
        
        # Enhanced features
        self.enable_meta_controller = enable_meta_controller
        self.enable_ensemble = enable_ensemble
        self.enable_auto_updater = enable_auto_updater
        
        # Configuration from settings or defaults
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
        self.retry_delay = retry_delay if retry_delay is not None else settings.RETRY_DELAY
        self.auto_update_interval = auto_update_interval if auto_update_interval is not None else settings.AUTO_UPDATE_INTERVAL_MINUTES

        # Initialize meta-controller if enabled
        if self.enable_meta_controller:
            self.meta_controller = self._initialize_meta_controller()
        else:
            self.meta_controller = None
            
        # Initialize ensemble system if enabled
        if self.enable_ensemble:
            self.ensemble_system = EnsembleSystem()
        else:
            self.ensemble_system = None
            
        # Initialize auto-updater if enabled
        if self.enable_auto_updater:
            self.auto_updater = AutoUpdater(account_manager=self.account_manager)
            # Start auto-update task
            asyncio.create_task(self._start_auto_updater())
        else:
            self.auto_updater = None
        
        logger.info(f"Initialized LLM Aggregator with {len(self.providers)} providers")
        logger.info(f"Meta-controller enabled: {self.enable_meta_controller}")
        logger.info(f"Ensemble system enabled: {self.enable_ensemble}")
    
    def _initialize_meta_controller(self) -> MetaModelController:
        """Initialize the meta-controller with model capability profiles."""
        
        model_profiles = {}
        
        for provider_name, provider in self.providers.items():
            models = provider.list_models()
            
            for model in models:
                # Create capability profile for each model
                profile = ModelCapabilityProfile(
                    model_name=model.name,
                    provider=provider_name,
                    size_category=self._categorize_model_size(model),
                    
                    # Capability scores (estimated based on model characteristics)
                    reasoning_ability=self._estimate_reasoning_ability(model),
                    code_generation=self._estimate_code_generation(model),
                    mathematical_reasoning=self._estimate_math_reasoning(model),
                    creative_writing=self._estimate_creative_writing(model),
                    factual_knowledge=self._estimate_factual_knowledge(model),
                    instruction_following=self._estimate_instruction_following(model),
                    context_handling=self._estimate_context_handling(model),
                    
                    # Performance metrics (initial estimates)
                    avg_response_time=2.0,  # Will be updated with real data
                    reliability_score=0.8,  # Will be updated with real data
                    cost_per_token=0.0 if model.is_free else 0.001,
                    max_context_length=model.context_length or 4096,
                    
                    # Specializations
                    domain_expertise=self._identify_domain_expertise(model),
                    preferred_task_types=self._identify_preferred_tasks(model)
                )
                
                model_profiles[model.name] = profile
        
        return MetaModelController(model_profiles)
    
    def _categorize_model_size(self, model: ModelInfo) -> str:
        """Categorize model size based on name and characteristics."""
        
        model_name_lower = model.name.lower()
        
        # Check for size indicators in model name
        if any(indicator in model_name_lower for indicator in ['7b', '8b', 'small', 'mini']):
            return "small"
        elif any(indicator in model_name_lower for indicator in ['13b', '14b', '20b', '27b', 'medium']):
            return "medium"
        elif any(indicator in model_name_lower for indicator in ['70b', '72b', '405b', 'large', 'xl']):
            return "large"
        else:
            # Default categorization based on context length
            if model.context_length and model.context_length > 100000:
                return "large"
            elif model.context_length and model.context_length > 32000:
                return "medium"
            else:
                return "small"
    
    def _estimate_reasoning_ability(self, model: ModelInfo) -> float:
        """Estimate reasoning ability based on model characteristics."""
        
        model_name_lower = model.name.lower()
        
        # Models known for reasoning
        if any(keyword in model_name_lower for keyword in ['r1', 'reasoning', 'think', 'o1']):
            return 0.9
        elif any(keyword in model_name_lower for keyword in ['llama', 'qwen', 'deepseek']):
            return 0.8
        elif any(keyword in model_name_lower for keyword in ['gemma', 'mistral']):
            return 0.7
        else:
            return 0.6
    
    def _estimate_code_generation(self, model: ModelInfo) -> float:
        """Estimate code generation ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['code', 'coder', 'codestral']):
            return 0.9
        elif any(keyword in model_name_lower for keyword in ['deepseek', 'qwen']):
            return 0.8
        elif any(keyword in model_name_lower for keyword in ['llama', 'mistral']):
            return 0.7
        else:
            return 0.5
    
    def _estimate_math_reasoning(self, model: ModelInfo) -> float:
        """Estimate mathematical reasoning ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['math', 'deepseek', 'qwen']):
            return 0.8
        elif any(keyword in model_name_lower for keyword in ['llama', 'r1']):
            return 0.7
        else:
            return 0.6
    
    def _estimate_creative_writing(self, model: ModelInfo) -> float:
        """Estimate creative writing ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['creative', 'story', 'writer']):
            return 0.9
        elif any(keyword in model_name_lower for keyword in ['llama', 'mistral', 'gemma']):
            return 0.7
        else:
            return 0.6
    
    def _estimate_factual_knowledge(self, model: ModelInfo) -> float:
        """Estimate factual knowledge capability."""
        
        # Larger models generally have more factual knowledge
        if model.context_length and model.context_length > 100000:
            return 0.8
        elif model.context_length and model.context_length > 32000:
            return 0.7
        else:
            return 0.6
    
    def _estimate_instruction_following(self, model: ModelInfo) -> float:
        """Estimate instruction following ability."""
        
        model_name_lower = model.name.lower()
        
        if any(keyword in model_name_lower for keyword in ['instruct', 'chat', 'assistant']):
            return 0.8
        else:
            return 0.7
    
    def _estimate_context_handling(self, model: ModelInfo) -> float:
        """Estimate context handling ability."""
        
        if model.context_length:
            if model.context_length >= 128000:
                return 0.9
            elif model.context_length >= 32000:
                return 0.8
            elif model.context_length >= 8000:
                return 0.7
            else:
                return 0.6
        else:
            return 0.6
    
    def _identify_domain_expertise(self, model: ModelInfo) -> List[str]:
        """Identify domain expertise based on model characteristics."""
        
        model_name_lower = model.name.lower()
        expertise = []
        
        if any(keyword in model_name_lower for keyword in ['code', 'coder']):
            expertise.append('programming')
        if any(keyword in model_name_lower for keyword in ['math']):
            expertise.append('mathematics')
        if any(keyword in model_name_lower for keyword in ['reasoning', 'r1']):
            expertise.append('reasoning')
        if any(keyword in model_name_lower for keyword in ['creative', 'story']):
            expertise.append('creative_writing')
        
        return expertise if expertise else ['general']
    
    def _identify_preferred_tasks(self, model: ModelInfo) -> List[str]:
        """Identify preferred task types for the model."""
        
        model_name_lower = model.name.lower()
        tasks = []
        
        if any(keyword in model_name_lower for keyword in ['code', 'coder']):
            tasks.extend(['code_generation', 'debugging', 'code_review'])
        if any(keyword in model_name_lower for keyword in ['math']):
            tasks.extend(['mathematical_reasoning', 'problem_solving'])
        if any(keyword in model_name_lower for keyword in ['reasoning', 'r1']):
            tasks.extend(['logical_reasoning', 'analysis', 'problem_solving'])
        if any(keyword in model_name_lower for keyword in ['chat', 'assistant']):
            tasks.extend(['conversation', 'question_answering'])
        
        return tasks if tasks else ['general_text_generation']
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Perform chat completion with intelligent provider selection and optional ensemble."""
        
        if request.stream:
            raise ValueError("Use chat_completion_stream for streaming requests")
        
        # Apply rate limiting
        await self.rate_limiter.acquire(user_id)
        
        try:
            # Use meta-controller for intelligent model selection if enabled
            if self.enable_meta_controller and self.meta_controller:
                return await self._chat_completion_with_meta_controller(request, user_id)
            
            # Use ensemble system if enabled
            elif self.enable_ensemble and self.ensemble_system:
                return await self._chat_completion_with_ensemble(request, user_id)
            
            # Fallback to traditional routing
            else:
                return await self._chat_completion_traditional(request, user_id)
            
        finally:
            self.rate_limiter.release(user_id)
    
    async def _chat_completion_with_meta_controller(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Chat completion using meta-controller for intelligent model selection."""
        
        start_time = time.time()
        
        # Get optimal model from meta-controller
        optimal_model, confidence = await self.meta_controller.select_optimal_model(request, user_id)
        
        logger.info(f"Meta-controller selected model: {optimal_model} (confidence: {confidence:.2f})")
        
        # If confidence is low, get cascade chain
        if confidence < 0.7:
            cascade_chain = await self.meta_controller.get_cascade_chain(request)
            logger.info(f"Low confidence, using cascade: {cascade_chain}")
        else:
            cascade_chain = [optimal_model]
        
        # Try models in cascade order
        last_error = None
        for model_name in cascade_chain:
            # Find provider for this model
            provider_name = None
            for prov_name, provider in self.providers.items():
                if any(model.name == model_name for model in provider.list_models()):
                    provider_name = prov_name
                    break
            
            if not provider_name:
                continue
            
            try:
                # Create request with specific model
                model_request = request.model_copy()
                model_request.model = model_name
                
                response = await self._try_provider(provider_name, model_request)
                if response:
                    # Update performance feedback
                    response_time = time.time() - start_time
                    await self.meta_controller.update_performance_feedback(
                        model_name, request, True, response_time, None
                    )
                    
                    logger.info(f"Successfully completed request using {model_name} via {provider_name}")
                    return response
                    
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                # Update performance feedback for failure
                response_time = time.time() - start_time
                await self.meta_controller.update_performance_feedback(
                    model_name, request, False, response_time, None
                )
                if provider_name: # Ensure provider_name was found
                    self.router.update_provider_score(provider_name, success=False, model_name=model_name)
                last_error = e
                continue
        
        # If all models in cascade failed, fallback to traditional routing
        logger.warning(f"Meta-controller cascade failed. Last error: {last_error}. Falling back to traditional routing.")
        return await self._chat_completion_traditional(request, user_id)
    
    async def _chat_completion_with_ensemble(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Chat completion using ensemble system for improved accuracy."""
        
        # Get top 3 models for ensemble
        if self.meta_controller:
            cascade_chain = await self.meta_controller.get_cascade_chain(request)
            ensemble_models = cascade_chain[:3]  # Top 3 models
        else:
            # Fallback to provider chain
            provider_chain = await self._get_provider_chain(request)
            ensemble_models = []
            for provider_name in provider_chain[:3]:
                provider = self.providers[provider_name]
                models = provider.list_models()
                if models:
                    ensemble_models.append(models[0].name)
        
        # Generate responses from multiple models
        model_responses = {}
        model_metadata = {}
        
        tasks = []
        for model_name in ensemble_models:
            # Find provider for this model
            provider_name = None
            for prov_name, provider in self.providers.items():
                if any(model.name == model_name for model in provider.list_models()):
                    provider_name = prov_name
                    break
            
            if provider_name:
                task = self._get_model_response(provider_name, model_name, request)
                tasks.append((model_name, provider_name, task))
        
        # Execute all requests concurrently
        for model_name, provider_name, task in tasks:
            try:
                start_time = time.time()
                response = await task
                response_time = time.time() - start_time
                
                if response:
                    model_responses[model_name] = response
                    model_metadata[model_name] = {
                        'provider': provider_name,
                        'response_time': response_time,
                        'confidence': 0.8,  # Default confidence
                        'cost_estimate': 0.0
                    }
            except Exception as e:
                logger.warning(f"Ensemble model {model_name} failed: {e}")
                continue
        
        # If we have multiple responses, use ensemble system
        if len(model_responses) > 1:
            logger.info(f"Generating ensemble response from {len(model_responses)} models")
            return await self.ensemble_system.generate_ensemble_response(
                request, model_responses, model_metadata
            )
        
        # If only one response, return it
        elif model_responses:
            return list(model_responses.values())[0]
        
        # If no responses, fallback to traditional routing
        else:
            logger.warning("Ensemble failed, falling back to traditional routing")
            return await self._chat_completion_traditional(request, user_id)
    
    async def _get_model_response(self, provider_name: str, model_name: str, 
                                request: ChatCompletionRequest) -> Optional[ChatCompletionResponse]:
        """Get response from a specific model."""
        
        model_request = request.model_copy()
        model_request.model = model_name
        
        return await self._try_provider(provider_name, model_request)
    
    async def _chat_completion_traditional(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Traditional chat completion with provider chain fallback."""
        
        # Get provider selection strategy
        provider_chain = await self._get_provider_chain(request)
        
        # Try providers in order
        last_error = None
        for provider_name in provider_chain:
            try:
                response = await self._try_provider(provider_name, request)
                if response:
                    logger.info(f"Successfully completed request using {provider_name}")
                    return response
                    
            except RateLimitError as e:
                logger.warning(f"Rate limit hit for {provider_name}: {e}")
                last_error = e
                continue
                
            except AuthenticationError as e:
                logger.error(f"Authentication failed for {provider_name}: {e}")
                # Mark credentials as invalid
                await self.account_manager.mark_credentials_invalid(provider_name)
                last_error = e
                continue
                
            except ProviderError as e:
                logger.warning(f"Provider error for {provider_name}: {e}")
                last_error = e
                continue
        
        # If we get here, all providers failed
        raise ProviderError(f"All providers failed. Last error: {last_error}")
    
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Perform streaming chat completion."""
        
        # Apply rate limiting
        await self.rate_limiter.acquire(user_id)
        
        try:
            # Get provider selection strategy
            provider_chain = await self._get_provider_chain(request)
            
            # Try providers in order
            for provider_name in provider_chain:
                try:
                    async for chunk in self._try_provider_stream(provider_name, request):
                        yield chunk
                    return  # Successfully completed
                    
                except RateLimitError as e:
                    logger.warning(f"Rate limit hit for {provider_name}: {e}")
                    continue
                    
                except AuthenticationError as e:
                    logger.error(f"Authentication failed for {provider_name}: {e}")
                    await self.account_manager.mark_credentials_invalid(provider_name)
                    continue
                    
                except ProviderError as e:
                    logger.warning(f"Provider error for {provider_name}: {e}")
                    continue
            
            # If we get here, all providers failed
            raise ProviderError("All providers failed for streaming request")
            
        finally:
            self.rate_limiter.release(user_id)
    
    async def _get_provider_chain(self, request: ChatCompletionRequest) -> List[str]:
        """Get ordered list of providers to try for this request."""
        
        # If specific provider requested, use it
        if request.provider and request.provider in self.providers:
            return [request.provider]
        
        # Use router to determine provider chain
        return await self.router.get_provider_chain(request)
    
    async def _try_provider(
        self,
        provider_name: str,
        request: ChatCompletionRequest
    ) -> Optional[ChatCompletionResponse]:
        """Try to complete request with specific provider."""
        
        provider = self.providers.get(provider_name)
        if not provider or not provider.is_available:
            return None
        
        # Get credentials for this provider
        credentials = await self.account_manager.get_credentials(provider_name)
        if not credentials:
            logger.warning(f"No valid credentials for {provider_name}")
            return None
        
        # Resolve model name if "auto" is specified
        model_name = request.model
        if model_name == "auto":
            model_name = await self._select_model(provider, request)
            if not model_name:
                logger.warning(f"No suitable model found for {provider_name}")
                return None
        
        # Create request with resolved model
        resolved_request = request.copy()
        resolved_request.model = model_name
        
        # Attempt request with retries
        for attempt in range(self.max_retries):
            try:
                response = await provider.chat_completion(resolved_request, credentials)
                
                # Update credentials usage
                await self.account_manager.update_usage(credentials)
                
                return response
                
            except RateLimitError:
                # Don't retry rate limit errors
                logger.warning(f"RateLimitError with {provider_name} for model {model_name}. Not retrying.")
                raise
                
            except ProviderError as e:
                logger.warning(f"ProviderError with {provider_name} for model {model_name} (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                logger.info(f"Retrying {provider_name} (attempt {attempt + 2}/{self.max_retries}) for model {model_name}")
            except Exception as e:
                logger.error(f"Unexpected error with {provider_name} for model {model_name} (attempt {attempt + 1}/{self.max_retries}): {e}", exc_info=True)
                if attempt == self.max_retries - 1:
                    # Raise as ProviderError to ensure consistent error type from this function
                    raise ProviderError(f"Unexpected error with {provider_name} after {self.max_retries} attempts: {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                logger.info(f"Retrying {provider_name} (attempt {attempt + 2}/{self.max_retries}) for model {model_name} after unexpected error.")
        
        return None
    
    async def _try_provider_stream(
        self,
        provider_name: str,
        request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Try streaming request with specific provider."""
        
        provider = self.providers.get(provider_name)
        if not provider or not provider.is_available:
            return
        
        # Get credentials for this provider
        credentials = await self.account_manager.get_credentials(provider_name)
        if not credentials:
            logger.warning(f"No valid credentials for {provider_name}")
            return
        
        # Resolve model name if "auto" is specified
        model_name = request.model
        if model_name == "auto":
            model_name = await self._select_model(provider, request)
            if not model_name:
                logger.warning(f"No suitable model found for {provider_name}")
                return
        
        # Create request with resolved model
        resolved_request = request.copy()
        resolved_request.model = model_name
        
        # Stream response
        try:
            async for chunk in provider.chat_completion_stream(resolved_request, credentials):
                yield chunk

            # Update credentials usage
            await self.account_manager.update_usage(credentials)
        except RateLimitError as e:
            logger.warning(f"RateLimitError during stream with {provider_name} for model {model_name}: {e}")
            raise  # Propagate to be handled by the caller
        except ProviderError as e:
            logger.warning(f"ProviderError during stream with {provider_name} for model {model_name}: {e}")
            raise # Propagate to be handled by the caller
        except Exception as e:
            logger.error(f"Unexpected error during stream with {provider_name} for model {model_name}: {e}", exc_info=True)
            # Wrap unexpected errors in ProviderError for consistent error handling
            raise ProviderError(f"Unexpected error during stream with {provider_name} for model {model_name}: {e}")
    
    async def _select_model(
        self,
        provider: BaseProvider,
        request: ChatCompletionRequest
    ) -> Optional[str]:
        """Select best model for request from provider's available models."""
        
        available_models = provider.list_models()
        if not available_models:
            return None
        
        # Simple model selection logic - can be enhanced
        # Prefer free models, then by capability match
        
        # Determine required capabilities from request
        required_capabilities = self._infer_capabilities(request)
        
        # Score models based on capability match and other factors
        scored_models = []
        for model in available_models:
            score = 0
            
            # Prefer free models
            if model.is_free:
                score += 100
            
            # Score based on capability match
            matching_caps = set(model.capabilities) & set(required_capabilities)
            score += len(matching_caps) * 10
            
            # Prefer larger context windows
            if model.context_length:
                score += min(model.context_length / 1000, 50)  # Cap at 50 points
            
            scored_models.append((score, model))
        
        # Sort by score and return best model
        scored_models.sort(key=lambda x: x[0], reverse=True)
        return scored_models[0][1].name if scored_models else None
    
    def _infer_capabilities(self, request: ChatCompletionRequest) -> List[ModelCapability]:
        """Infer required capabilities from request."""
        capabilities = [ModelCapability.TEXT_GENERATION]
        
        # Analyze message content for capability hints
        content = " ".join(msg.content.lower() for msg in request.messages)
        
        if any(keyword in content for keyword in ["code", "python", "javascript", "programming"]):
            capabilities.append(ModelCapability.CODE_GENERATION)
        
        if any(keyword in content for keyword in ["think", "reason", "solve", "analyze"]):
            capabilities.append(ModelCapability.REASONING)
        
        return capabilities
    
    async def list_available_models(self) -> Dict[str, List[ModelInfo]]:
        """List all available models across providers."""
        models_by_provider = {}
        
        for provider_name, provider in self.providers.items():
            if provider.is_available:
                models_by_provider[provider_name] = provider.list_models()
        
        return models_by_provider
    
    async def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        
        for provider_name, provider in self.providers.items():
            status[provider_name] = {
                "available": provider.is_available,
                "status": provider.config.status.value,
                "metrics": provider.metrics.dict(),
                "models_count": len(provider.config.models),
                "credentials_count": len(provider.credentials)
            }
        
        return status
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all providers."""
        results = {}
        
        tasks = []
        for provider_name, provider in self.providers.items():
            if provider.is_available:
                tasks.append(self._provider_health_check(provider_name, provider))
        
        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(health_results):
                provider_name = list(self.providers.keys())[i]
                results[provider_name] = isinstance(result, bool) and result
        
        return results
    
    async def _provider_health_check(self, provider_name: str, provider: BaseProvider) -> bool:
        """Perform health check on single provider."""
        try:
            return await provider.health_check()
        except Exception as e:
            logger.error(f"Health check failed for {provider_name}: {e}")
            return False
    
    async def get_meta_controller_insights(self) -> Optional[Dict[str, Any]]:
        """Get insights from the meta-controller about model performance and selection."""
        
        if not self.meta_controller:
            return None
        
        return self.meta_controller.get_model_insights()
    
    async def get_ensemble_insights(self, request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
        """Get insights about ensemble decision process for a given request."""
        
        if not self.ensemble_system or not self.meta_controller:
            return None
        
        # Get candidate models
        cascade_chain = await self.meta_controller.get_cascade_chain(request)
        
        # Create mock candidates for analysis
        from .ensemble_system import ResponseCandidate
        candidates = []
        
        for model_name in cascade_chain[:3]:
            # Find provider for this model
            provider_name = None
            for prov_name, provider in self.providers.items():
                if any(model.name == model_name for model in provider.list_models()):
                    provider_name = prov_name
                    break
            
            if provider_name and model_name in self.meta_controller.model_profiles:
                profile = self.meta_controller.model_profiles[model_name]
                
                # Create mock candidate
                candidate = ResponseCandidate(
                    model_name=model_name,
                    provider=provider_name,
                    response=None,  # Mock response
                    confidence_score=profile.reliability_score,
                    response_time=profile.avg_response_time,
                    cost_estimate=profile.cost_per_token,
                    coherence_score=profile.instruction_following,
                    relevance_score=profile.factual_knowledge,
                    factual_accuracy_score=profile.factual_knowledge,
                    creativity_score=profile.creative_writing,
                    safety_score=0.8  # Default safety score
                )
                candidates.append(candidate)
        
        return self.ensemble_system.get_ensemble_insights(candidates)
    
    async def analyze_task_complexity(self, request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
        """Analyze the complexity of a task using the meta-controller."""
        
        if not self.meta_controller:
            return None
        
        complexity = self.meta_controller.complexity_analyzer.analyze_task_complexity(request)
        
        return {
            'reasoning_depth': complexity.reasoning_depth,
            'domain_specificity': complexity.domain_specificity,
            'context_length': complexity.context_length,
            'computational_intensity': complexity.computational_intensity,
            'creativity_required': complexity.creativity_required,
            'factual_accuracy_importance': complexity.factual_accuracy_importance,
            'overall_complexity': complexity.overall_complexity
        }
    
    async def get_model_recommendations(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Get model recommendations for a specific request."""
        
        recommendations = {
            'traditional_routing': await self._get_provider_chain(request),
            'meta_controller_insights': None,
            'ensemble_insights': None,
            'task_complexity': None
        }
        
        if self.meta_controller:
            # Get meta-controller recommendation
            optimal_model, confidence = await self.meta_controller.select_optimal_model(request)
            cascade_chain = await self.meta_controller.get_cascade_chain(request)
            
            recommendations['meta_controller_insights'] = {
                'optimal_model': optimal_model,
                'confidence': confidence,
                'cascade_chain': cascade_chain,
                'model_insights': await self.get_meta_controller_insights()
            }
            
            # Get task complexity analysis
            recommendations['task_complexity'] = await self.analyze_task_complexity(request)
        
        if self.ensemble_system:
            # Get ensemble insights
            recommendations['ensemble_insights'] = await self.get_ensemble_insights(request)
        
        return recommendations
    
    async def _start_auto_updater(self):
        """Start the auto-updater background task."""
        if not self.auto_updater:
            return
        
        try:
            # Integrate auto-updater with this aggregator
            await integrate_auto_updater(self, self.auto_updater)
            
            # Start the auto-update loop
            await self.auto_updater.start_auto_update(self.auto_update_interval)
            
        except Exception as e:
            logger.error(f"Error in auto-updater: {e}")
    
    async def force_update_providers(self) -> Dict[str, Any]:
        """Force an immediate update of all provider information."""
        if not self.auto_updater:
            return {"error": "Auto-updater not enabled"}
        
        try:
            updates = await self.auto_updater.force_update_all()
            
            return {
                "status": "success",
                "updates_found": len(updates),
                "updates": [
                    {
                        "provider": update.provider_name,
                        "models_added": len(update.models_added),
                        "models_removed": len(update.models_removed),
                        "models_updated": len(update.models_updated),
                        "rate_limits_updated": bool(update.rate_limits_updated),
                        "timestamp": update.timestamp.isoformat() if update.timestamp else None
                    }
                    for update in updates
                ]
            }
            
        except Exception as e:
            logger.error(f"Error forcing provider updates: {e}")
            return {"error": str(e)}
    
    async def get_auto_update_status(self) -> Dict[str, Any]:
        """Get the current status of the auto-updater."""
        if not self.auto_updater:
            return {"enabled": False}
        
        try:
            status = await self.auto_updater.get_update_status()
            status["enabled"] = True
            return status
            
        except Exception as e:
            logger.error(f"Error getting auto-update status: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def configure_auto_updater(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure auto-updater settings."""
        if not self.auto_updater:
            return {"error": "Auto-updater not enabled"}
        
        try:
            # Update interval
            if "update_interval" in config:
                self.auto_update_interval = config["update_interval"]
            
            # Enable/disable specific sources
            if "sources" in config:
                for source_config in config["sources"]:
                    source_name = source_config.get("name")
                    if source_name:
                        # Find and update source
                        for source in self.auto_updater.sources:
                            if source.name == source_name:
                                for key, value in source_config.items():
                                    if hasattr(source, key):
                                        setattr(source, key, value)
                                break
            
            # Save updated configuration
            self.auto_updater._save_update_sources(self.auto_updater.sources)
            
            return {"status": "success", "message": "Auto-updater configuration updated"}
            
        except Exception as e:
            logger.error(f"Error configuring auto-updater: {e}")
            return {"error": str(e)}
    
    async def get_provider_updates_history(self, provider_name: str = None) -> Dict[str, Any]:
        """Get history of provider updates."""
        if not self.auto_updater:
            return {"error": "Auto-updater not enabled"}
        
        try:
            # This would require storing update history
            # For now, return cached data
            if provider_name:
                cached_data = self.auto_updater.cache.get(f"api_{provider_name}_models")
                if cached_data:
                    return {
                        "provider": provider_name,
                        "cached_models": len(cached_data),
                        "last_update": "Available in cache"
                    }
                else:
                    return {"provider": provider_name, "error": "No cached data"}
            else:
                # Return summary for all providers
                summary = {}
                for key, value in self.auto_updater.cache.items():
                    if key.startswith("api_") and key.endswith("_models"):
                        provider = key.replace("api_", "").replace("_models", "")
                        summary[provider] = {
                            "cached_models": len(value) if isinstance(value, list) else 0
                        }
                
                return {"providers": summary}
                
        except Exception as e:
            logger.error(f"Error getting provider updates history: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close all providers and cleanup resources."""
        tasks = []
        for provider in self.providers.values():
            tasks.append(provider.close())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close auto-updater
        if self.auto_updater:
            await self.auto_updater.close()
        
        logger.info("LLM Aggregator closed")