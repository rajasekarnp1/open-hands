"""
Browser automation for monitoring LLM provider dashboards and websites.

Uses Playwright for reliable browser automation to:
1. Monitor provider websites for new models and pricing changes
2. Check dashboard rate limits and usage information
3. Detect provider status changes and outages
4. Scrape model documentation and capabilities
"""

import asyncio
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from ..models import ModelInfo, ProviderStatus


logger = logging.getLogger(__name__)


@dataclass
class BrowserConfig:
    """Configuration for browser automation."""
    headless: bool = True
    timeout: int = 30000
    viewport: Dict[str, int] = None
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    def __post_init__(self):
        if self.viewport is None:
            self.viewport = {"width": 1920, "height": 1080}


@dataclass
class ProviderMonitorConfig:
    """Configuration for monitoring a specific provider."""
    name: str
    base_url: str
    models_page: str
    pricing_page: Optional[str] = None
    dashboard_url: Optional[str] = None
    login_url: Optional[str] = None
    
    # Selectors for extracting information
    selectors: Dict[str, str] = None
    
    # Login configuration
    login_method: str = "none"  # "none", "oauth", "credentials"
    login_selectors: Dict[str, str] = None
    
    # Rate limiting
    request_delay: float = 2.0
    
    def __post_init__(self):
        if self.selectors is None:
            self.selectors = {}
        if self.login_selectors is None:
            self.login_selectors = {}


class BrowserMonitor:
    """Browser-based monitoring for LLM providers."""
    
    def __init__(self, config: BrowserConfig = None):
        self.config = config or BrowserConfig()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        # Provider configurations
        self.provider_configs = self._get_default_provider_configs()
        
        # Cache for storing scraped data
        self.cache = {}
        
    def _get_default_provider_configs(self) -> Dict[str, ProviderMonitorConfig]:
        """Get default configurations for known providers."""
        
        return {
            "openrouter": ProviderMonitorConfig(
                name="openrouter",
                base_url="https://openrouter.ai",
                models_page="https://openrouter.ai/models",
                pricing_page="https://openrouter.ai/docs/pricing",
                dashboard_url="https://openrouter.ai/account",
                login_url="https://openrouter.ai/login",
                selectors={
                    "model_cards": ".model-card, [data-testid='model-card']",
                    "model_name": ".model-name, h3, .font-semibold",
                    "model_id": "[data-model-id], .model-id",
                    "pricing": ".pricing, .price, .cost",
                    "free_badge": ".free, [data-free='true'], .badge-free",
                    "context_length": ".context, .max-tokens, .context-length",
                    "description": ".description, .model-description",
                    "capabilities": ".capabilities, .tags, .model-tags",
                    "provider_status": ".status, .health-indicator",
                    "rate_limits": ".rate-limit, .limits",
                    "usage_stats": ".usage, .stats, .account-usage"
                },
                login_selectors={
                    "email": "input[type='email'], input[name='email']",
                    "password": "input[type='password'], input[name='password']",
                    "submit": "button[type='submit'], .login-button",
                    "oauth_github": ".github-login, [data-provider='github']",
                    "oauth_google": ".google-login, [data-provider='google']"
                }
            ),
            
            "groq": ProviderMonitorConfig(
                name="groq",
                base_url="https://groq.com",
                models_page="https://groq.com/models/",
                dashboard_url="https://console.groq.com",
                selectors={
                    "model_cards": ".model-card, .model-item",
                    "model_name": ".model-name, h3",
                    "pricing": ".pricing, .free",
                    "free_badge": ".free, .no-cost",
                    "context_length": ".context-window, .max-tokens",
                    "description": ".description",
                    "performance": ".performance, .speed"
                }
            ),
            
            "cerebras": ProviderMonitorConfig(
                name="cerebras",
                base_url="https://cerebras.ai",
                models_page="https://cerebras.ai/models",
                dashboard_url="https://cloud.cerebras.ai",
                selectors={
                    "model_cards": ".model-card",
                    "model_name": ".model-name, h3",
                    "pricing": ".pricing",
                    "free_badge": ".free",
                    "context_length": ".context-length",
                    "description": ".description"
                }
            ),
            
            "huggingface": ProviderMonitorConfig(
                name="huggingface",
                base_url="https://huggingface.co",
                models_page="https://huggingface.co/models?pipeline_tag=text-generation&sort=trending",
                selectors={
                    "model_cards": ".model-card, article",
                    "model_name": "h4, .text-lg",
                    "model_id": "[data-model-id]",
                    "downloads": ".downloads",
                    "likes": ".likes",
                    "tags": ".tag, .badge",
                    "description": ".description, .text-gray-600"
                }
            ),
            
            "together": ProviderMonitorConfig(
                name="together",
                base_url="https://together.ai",
                models_page="https://together.ai/models",
                dashboard_url="https://api.together.xyz/playground",
                selectors={
                    "model_cards": ".model-card",
                    "model_name": ".model-name",
                    "pricing": ".pricing",
                    "context_length": ".context-length"
                }
            )
        }
    
    async def start(self):
        """Start the browser and create context."""
        playwright = await async_playwright().start()
        
        self.browser = await playwright.chromium.launch(
            headless=self.config.headless,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        
        self.context = await self.browser.new_context(
            viewport=self.config.viewport,
            user_agent=self.config.user_agent
        )
        
        # Set default timeout
        self.context.set_default_timeout(self.config.timeout)
        
        logger.info("Browser monitor started")
    
    async def stop(self):
        """Stop the browser and cleanup."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        
        logger.info("Browser monitor stopped")
    
    async def monitor_provider(self, provider_name: str) -> Dict[str, Any]:
        """Monitor a specific provider for updates."""
        
        if provider_name not in self.provider_configs:
            logger.error(f"Unknown provider: {provider_name}")
            return {}
        
        config = self.provider_configs[provider_name]
        
        try:
            # Create new page for this provider
            page = await self.context.new_page()
            
            # Monitor models page
            models_data = await self._scrape_models_page(page, config)
            
            # Monitor pricing page if available
            pricing_data = {}
            if config.pricing_page:
                pricing_data = await self._scrape_pricing_page(page, config)
            
            # Monitor dashboard if available (requires login)
            dashboard_data = {}
            if config.dashboard_url:
                dashboard_data = await self._scrape_dashboard(page, config)
            
            await page.close()
            
            # Combine all data
            result = {
                "provider": provider_name,
                "timestamp": datetime.now().isoformat(),
                "models": models_data,
                "pricing": pricing_data,
                "dashboard": dashboard_data,
                "status": "success"
            }
            
            # Cache the result
            self.cache[provider_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error monitoring {provider_name}: {e}")
            return {
                "provider": provider_name,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    async def _scrape_models_page(self, page: Page, config: ProviderMonitorConfig) -> List[Dict[str, Any]]:
        """Scrape the models page for a provider."""
        
        logger.info(f"Scraping models page: {config.models_page}")
        
        try:
            await page.goto(config.models_page)
            await page.wait_for_load_state("networkidle")
            
            # Wait for model cards to load
            model_cards_selector = config.selectors.get("model_cards", ".model-card")
            await page.wait_for_selector(model_cards_selector, timeout=10000)
            
            # Extract model information
            models = []
            model_cards = await page.query_selector_all(model_cards_selector)
            
            for card in model_cards:
                try:
                    model_data = await self._extract_model_data(card, config)
                    if model_data:
                        models.append(model_data)
                except Exception as e:
                    logger.warning(f"Error extracting model data: {e}")
                    continue
            
            logger.info(f"Found {len(models)} models for {config.name}")
            return models
            
        except Exception as e:
            logger.error(f"Error scraping models page for {config.name}: {e}")
            return []
    
    async def _extract_model_data(self, card_element, config: ProviderMonitorConfig) -> Optional[Dict[str, Any]]:
        """Extract model data from a model card element."""
        
        model_data = {}
        selectors = config.selectors
        
        try:
            # Model name
            name_selector = selectors.get("model_name", ".model-name")
            name_element = await card_element.query_selector(name_selector)
            if name_element:
                model_data["name"] = (await name_element.text_content()).strip()
            
            # Model ID (if different from name)
            id_selector = selectors.get("model_id")
            if id_selector:
                id_element = await card_element.query_selector(id_selector)
                if id_element:
                    model_data["id"] = await id_element.get_attribute("data-model-id") or await id_element.text_content()
            
            # Pricing information
            pricing_selector = selectors.get("pricing", ".pricing")
            pricing_element = await card_element.query_selector(pricing_selector)
            if pricing_element:
                pricing_text = (await pricing_element.text_content()).strip().lower()
                model_data["pricing_text"] = pricing_text
                model_data["is_free"] = any(keyword in pricing_text for keyword in ["free", "$0", "no cost", "gratis"])
            
            # Free badge
            free_badge_selector = selectors.get("free_badge")
            if free_badge_selector:
                free_badge = await card_element.query_selector(free_badge_selector)
                if free_badge:
                    model_data["is_free"] = True
            
            # Context length
            context_selector = selectors.get("context_length")
            if context_selector:
                context_element = await card_element.query_selector(context_selector)
                if context_element:
                    context_text = await context_element.text_content()
                    context_length = self._extract_number(context_text)
                    if context_length:
                        model_data["context_length"] = context_length
            
            # Description
            desc_selector = selectors.get("description")
            if desc_selector:
                desc_element = await card_element.query_selector(desc_selector)
                if desc_element:
                    model_data["description"] = (await desc_element.text_content()).strip()
            
            # Capabilities/Tags
            caps_selector = selectors.get("capabilities")
            if caps_selector:
                caps_elements = await card_element.query_selector_all(caps_selector)
                capabilities = []
                for cap_element in caps_elements:
                    cap_text = (await cap_element.text_content()).strip()
                    if cap_text:
                        capabilities.append(cap_text)
                if capabilities:
                    model_data["capabilities"] = capabilities
            
            # Only return if we have at least a name
            if "name" in model_data:
                return model_data
            
        except Exception as e:
            logger.warning(f"Error extracting model data: {e}")
        
        return None
    
    async def _scrape_pricing_page(self, page: Page, config: ProviderMonitorConfig) -> Dict[str, Any]:
        """Scrape the pricing page for rate limits and cost information."""
        
        try:
            await page.goto(config.pricing_page)
            await page.wait_for_load_state("networkidle")
            
            # Extract pricing information
            pricing_data = {}
            
            # Look for rate limit information
            rate_limit_patterns = [
                r"(\d+)\s*requests?\s*per\s*(minute|hour|day)",
                r"(\d+)\s*RPM",
                r"(\d+)\s*RPD",
                r"(\d+)\s*tokens?\s*per\s*(minute|hour)"
            ]
            
            page_text = await page.text_content("body")
            
            for pattern in rate_limit_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        count, period = match
                        key = f"rate_limit_{period.lower()}"
                        pricing_data[key] = int(count)
            
            # Look for free tier information
            if "free" in page_text.lower():
                pricing_data["has_free_tier"] = True
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Error scraping pricing page: {e}")
            return {}
    
    async def _scrape_dashboard(self, page: Page, config: ProviderMonitorConfig) -> Dict[str, Any]:
        """Scrape dashboard for usage and account information."""
        
        # This would require authentication, so skip for now
        # In a full implementation, you would:
        # 1. Handle login process
        # 2. Navigate to dashboard
        # 3. Extract usage statistics
        # 4. Extract current rate limits
        
        logger.info(f"Dashboard scraping not implemented for {config.name}")
        return {}
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract a number from text (e.g., context length)."""
        
        # Remove common suffixes and formatting
        text = text.replace(",", "").replace("k", "000").replace("K", "000")
        
        # Find numbers
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        
        return None
    
    async def monitor_all_providers(self) -> Dict[str, Any]:
        """Monitor all configured providers."""
        
        results = {}
        
        for provider_name in self.provider_configs:
            try:
                result = await self.monitor_provider(provider_name)
                results[provider_name] = result
                
                # Add delay between providers to be respectful
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error monitoring {provider_name}: {e}")
                results[provider_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    async def detect_changes(self, provider_name: str) -> Dict[str, Any]:
        """Detect changes since last monitoring."""
        
        current_data = await self.monitor_provider(provider_name)
        
        if provider_name not in self.cache:
            return {"status": "first_run", "changes": []}
        
        previous_data = self.cache[provider_name]
        changes = []
        
        # Compare models
        current_models = {m["name"]: m for m in current_data.get("models", [])}
        previous_models = {m["name"]: m for m in previous_data.get("models", [])}
        
        # New models
        new_models = set(current_models.keys()) - set(previous_models.keys())
        for model_name in new_models:
            changes.append({
                "type": "model_added",
                "model": model_name,
                "data": current_models[model_name]
            })
        
        # Removed models
        removed_models = set(previous_models.keys()) - set(current_models.keys())
        for model_name in removed_models:
            changes.append({
                "type": "model_removed",
                "model": model_name
            })
        
        # Changed models
        for model_name in set(current_models.keys()) & set(previous_models.keys()):
            current_model = current_models[model_name]
            previous_model = previous_models[model_name]
            
            # Check for pricing changes
            if current_model.get("is_free") != previous_model.get("is_free"):
                changes.append({
                    "type": "pricing_changed",
                    "model": model_name,
                    "old_free": previous_model.get("is_free"),
                    "new_free": current_model.get("is_free")
                })
            
            # Check for context length changes
            if current_model.get("context_length") != previous_model.get("context_length"):
                changes.append({
                    "type": "context_length_changed",
                    "model": model_name,
                    "old_length": previous_model.get("context_length"),
                    "new_length": current_model.get("context_length")
                })
        
        return {
            "status": "success",
            "changes": changes,
            "total_changes": len(changes)
        }
    
    def get_cached_data(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get cached data for a provider."""
        return self.cache.get(provider_name)
    
    def clear_cache(self, provider_name: str = None):
        """Clear cache for a specific provider or all providers."""
        if provider_name:
            self.cache.pop(provider_name, None)
        else:
            self.cache.clear()


# Utility functions for integration
async def create_browser_monitor() -> BrowserMonitor:
    """Create and start a browser monitor."""
    monitor = BrowserMonitor()
    await monitor.start()
    return monitor


async def monitor_provider_changes(provider_name: str) -> Dict[str, Any]:
    """Quick function to monitor a single provider for changes."""
    monitor = await create_browser_monitor()
    
    try:
        changes = await monitor.detect_changes(provider_name)
        return changes
    finally:
        await monitor.stop()


async def get_all_provider_models() -> Dict[str, List[Dict[str, Any]]]:
    """Get current models from all providers using browser monitoring."""
    monitor = await create_browser_monitor()
    
    try:
        results = await monitor.monitor_all_providers()
        
        # Extract just the models
        all_models = {}
        for provider_name, data in results.items():
            if data.get("status") == "success":
                all_models[provider_name] = data.get("models", [])
        
        return all_models
    finally:
        await monitor.stop()


# Example usage
if __name__ == "__main__":
    async def main():
        monitor = await create_browser_monitor()
        
        try:
            # Monitor OpenRouter
            result = await monitor.monitor_provider("openrouter")
            print(json.dumps(result, indent=2))
            
            # Check for changes
            changes = await monitor.detect_changes("openrouter")
            print(f"Changes detected: {changes['total_changes']}")
            
        finally:
            await monitor.stop()
    
    asyncio.run(main())