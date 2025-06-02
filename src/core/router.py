"""
Provider routing and selection logic.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models import (
    ChatCompletionRequest,
    RoutingRule,
    ModelCapability,
    ProviderConfig
)


logger = logging.getLogger(__name__)


class ProviderRouter:
    """Intelligent provider routing and selection."""
    
    def __init__(self, providers: Dict[str, ProviderConfig], routing_rules: Optional[List[RoutingRule]] = None):
        self.providers = providers
        self.routing_rules = routing_rules or []
        self.provider_scores: Dict[str, float] = {}
        
        # Initialize default routing rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default routing rules."""
        
        # Rule for code generation tasks
        code_rule = RoutingRule(
            name="code_generation",
            conditions={
                "content_keywords": ["code", "python", "javascript", "programming", "function", "class"],
                "capabilities": [ModelCapability.CODE_GENERATION]
            },
            provider_preferences=["openrouter", "groq", "cerebras"],
            fallback_chain=["openrouter", "groq", "cerebras", "together"],
            is_active=True
        )
        
        # Rule for reasoning tasks
        reasoning_rule = RoutingRule(
            name="reasoning",
            conditions={
                "content_keywords": ["think", "reason", "solve", "analyze", "logic", "problem"],
                "capabilities": [ModelCapability.REASONING]
            },
            provider_preferences=["openrouter", "groq"],
            fallback_chain=["openrouter", "groq", "cerebras"],
            is_active=True
        )
        
        # Rule for general text generation
        general_rule = RoutingRule(
            name="general_text",
            conditions={
                "capabilities": [ModelCapability.TEXT_GENERATION]
            },
            provider_preferences=["openrouter", "groq", "cerebras", "together"],
            fallback_chain=["openrouter", "groq", "cerebras", "together", "cohere"],
            is_active=True
        )
        
        # Rule for fast responses (prioritize Groq and Cerebras)
        fast_rule = RoutingRule(
            name="fast_response",
            conditions={
                "max_tokens": 100,  # Short responses
                "temperature": 0.0  # Deterministic responses
            },
            provider_preferences=["groq", "cerebras", "openrouter"],
            fallback_chain=["groq", "cerebras", "openrouter"],
            is_active=True
        )
        
        self.routing_rules.extend([code_rule, reasoning_rule, general_rule, fast_rule])
    
    async def get_provider_chain(self, request: ChatCompletionRequest) -> List[str]:
        """Get ordered list of providers to try for this request."""
        
        # Find matching routing rules
        matching_rules = self._find_matching_rules(request)
        
        if matching_rules:
            # Use the first matching rule (rules should be ordered by priority)
            rule = matching_rules[0]
            logger.debug(f"Using routing rule: {rule.name}")
            
            # Start with preferred providers
            provider_chain = rule.provider_preferences.copy()
            
            # Add fallback providers that aren't already in the chain
            for provider in rule.fallback_chain:
                if provider not in provider_chain:
                    provider_chain.append(provider)
        else:
            # No specific rule matched, use default ordering
            provider_chain = self._get_default_provider_order()
        
        # Filter to only available providers
        available_providers = [
            provider for provider in provider_chain
            if provider in self.providers and self._is_provider_available(provider)
        ]
        
        # Apply dynamic scoring to reorder providers
        scored_providers = await self._score_providers(available_providers, request)
        
        return scored_providers
    
    def _find_matching_rules(self, request: ChatCompletionRequest) -> List[RoutingRule]:
        """Find routing rules that match the request."""
        
        matching_rules = []
        
        for rule in self.routing_rules:
            if not rule.is_active:
                continue
            
            if self._rule_matches_request(rule, request):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _rule_matches_request(self, rule: RoutingRule, request: ChatCompletionRequest) -> bool:
        """Check if a routing rule matches the request."""
        
        conditions = rule.conditions
        
        # Check content keywords
        if "content_keywords" in conditions:
            content = " ".join(msg.content.lower() for msg in request.messages)
            keywords = conditions["content_keywords"]
            if not any(keyword in content for keyword in keywords):
                return False
        
        # Check request parameters
        if "max_tokens" in conditions:
            if not request.max_tokens or request.max_tokens > conditions["max_tokens"]:
                return False
        
        if "temperature" in conditions:
            if request.temperature != conditions["temperature"]:
                return False
        
        # Check model capabilities (this would require model analysis)
        if "capabilities" in conditions:
            required_caps = conditions["capabilities"]
            inferred_caps = self._infer_request_capabilities(request)
            if not any(cap in inferred_caps for cap in required_caps):
                return False
        
        return True
    
    def _infer_request_capabilities(self, request: ChatCompletionRequest) -> List[ModelCapability]:
        """Infer required capabilities from request."""
        capabilities = [ModelCapability.TEXT_GENERATION]
        
        # Analyze message content
        content = " ".join(msg.content.lower() for msg in request.messages)
        
        if any(keyword in content for keyword in ["code", "python", "javascript", "programming", "function", "class"]):
            capabilities.append(ModelCapability.CODE_GENERATION)
        
        if any(keyword in content for keyword in ["think", "reason", "solve", "analyze", "logic"]):
            capabilities.append(ModelCapability.REASONING)
        
        if any(keyword in content for keyword in ["image", "picture", "photo", "visual"]):
            capabilities.append(ModelCapability.VISION)
        
        return capabilities
    
    def _get_default_provider_order(self) -> List[str]:
        """Get default provider ordering based on priority and performance."""
        
        # Sort providers by priority (lower number = higher priority)
        sorted_providers = sorted(
            self.providers.items(),
            key=lambda x: (x[1].priority, x[0])  # Secondary sort by name for consistency
        )
        
        return [provider_name for provider_name, _ in sorted_providers]
    
    def _is_provider_available(self, provider_name: str) -> bool:
        """Check if provider is currently available."""
        
        if provider_name not in self.providers:
            return False
        
        provider_config = self.providers[provider_name]
        return provider_config.status.value == "active"
    
    async def _score_providers(self, providers: List[str], request: ChatCompletionRequest) -> List[str]:
        """Score and reorder providers based on current performance and suitability."""
        
        scored_providers = []
        
        for provider_name in providers:
            score = await self._calculate_provider_score(provider_name, request)
            scored_providers.append((score, provider_name))
        
        # Sort by score (higher is better)
        scored_providers.sort(key=lambda x: x[0], reverse=True)
        
        return [provider_name for _, provider_name in scored_providers]
    
    async def _calculate_provider_score(self, provider_name: str, request: ChatCompletionRequest) -> float:
        """Calculate score for a provider based on various factors."""
        
        if provider_name not in self.providers:
            return 0.0
        
        provider_config = self.providers[provider_name]
        score = 0.0
        
        # Base score from priority (invert so lower priority number = higher score)
        score += (10 - provider_config.priority) * 10
        
        # Bonus for free providers
        if provider_config.provider_type.value == "free":
            score += 50
        elif provider_config.provider_type.value == "trial_credit":
            score += 30
        
        # Bonus for streaming support if requested
        if request.stream and provider_config.supports_streaming:
            score += 20
        
        # Model availability bonus
        suitable_models = self._count_suitable_models(provider_config, request)
        score += suitable_models * 5
        
        # Performance bonus (if we have metrics)
        if provider_name in self.provider_scores:
            score += self.provider_scores[provider_name]
        
        return score
    
    def _count_suitable_models(self, provider_config: ProviderConfig, request: ChatCompletionRequest) -> int:
        """Count how many suitable models the provider has for this request."""
        
        required_capabilities = self._infer_request_capabilities(request)
        suitable_count = 0
        
        for model in provider_config.models:
            # Check if model has required capabilities
            if any(cap in model.capabilities for cap in required_capabilities):
                suitable_count += 1
        
        return suitable_count
    
    def update_provider_score(self, provider_name: str, score_delta: float):
        """Update provider performance score based on success/failure."""
        
        if provider_name not in self.provider_scores:
            self.provider_scores[provider_name] = 0.0
        
        # Apply exponential moving average
        alpha = 0.1  # Learning rate
        self.provider_scores[provider_name] = (
            (1 - alpha) * self.provider_scores[provider_name] + alpha * score_delta
        )
        
        # Clamp score to reasonable range
        self.provider_scores[provider_name] = max(-50, min(50, self.provider_scores[provider_name]))
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule."""
        self.routing_rules.append(rule)
        logger.info(f"Added routing rule: {rule.name}")
    
    def remove_routing_rule(self, rule_name: str) -> bool:
        """Remove a routing rule by name."""
        original_count = len(self.routing_rules)
        self.routing_rules = [rule for rule in self.routing_rules if rule.name != rule_name]
        
        removed = len(self.routing_rules) < original_count
        if removed:
            logger.info(f"Removed routing rule: {rule_name}")
        
        return removed
    
    def get_routing_rules(self) -> List[RoutingRule]:
        """Get all routing rules."""
        return self.routing_rules.copy()
    
    def get_provider_scores(self) -> Dict[str, float]:
        """Get current provider performance scores."""
        return self.provider_scores.copy()