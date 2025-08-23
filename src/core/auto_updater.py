"""
Auto-updater for free LLM API providers, models, and rate limits.

Integrates with existing GitHub projects and uses multiple discovery methods:
1. GitHub API monitoring for community-maintained lists
2. Direct API discovery for new models and limits
3. Web scraping for provider websites
4. Browser automation for dashboard monitoring
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import hashlib

import httpx
import yaml
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from ..models import ProviderConfig, ModelInfo, ProviderStatus
from .account_manager import AccountManager


logger = logging.getLogger(__name__)


@dataclass
class UpdateSource:
    """Configuration for an update source."""
    name: str
    type: str  # 'github', 'api', 'web_scrape', 'browser'
    url: str
    update_interval: int  # hours
    last_updated: Optional[datetime] = None
    enabled: bool = True
    config: Dict[str, Any] = None


@dataclass
class ProviderUpdate:
    """Update information for a provider."""
    provider_name: str
    models_added: List[ModelInfo]
    models_removed: List[str]
    models_updated: List[ModelInfo]
    rate_limits_updated: Dict[str, Any]
    status_changed: Optional[ProviderStatus] = None
    timestamp: datetime = None


class AutoUpdater:
    """Automatic updater for LLM API providers and models."""
    
    def __init__(
        self,
        config_path: str = "config/auto_update.yaml",
        cache_path: str = "cache/provider_cache.json",
        account_manager: Optional[AccountManager] = None
    ):
        self.config_path = Path(config_path)
        self.cache_path = Path(cache_path)
        self.account_manager = account_manager
        
        # Ensure directories exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.sources = self._load_update_sources()
        self.cache = self._load_cache()
        
        # HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "LLM-API-Aggregator/1.0"}
        )
        
        # Update tracking
        self.last_full_update = None
        self.update_callbacks = []
        
        logger.info(f"AutoUpdater initialized with {len(self.sources)} sources")
    
    def _load_update_sources(self) -> List[UpdateSource]:
        """Load update source configurations."""
        
        default_sources = [
            # GitHub community projects
            UpdateSource(
                name="cheahjs/free-llm-api-resources",
                type="github",
                url="https://api.github.com/repos/cheahjs/free-llm-api-resources",
                update_interval=6,  # 6 hours
                config={
                    "content_path": "src/data.py",
                    "readme_path": "README.md",
                    "parser": "python_dict"
                }
            ),
            UpdateSource(
                name="zukixa/cool-ai-stuff",
                type="github", 
                url="https://api.github.com/repos/zukixa/cool-ai-stuff",
                update_interval=12,
                config={
                    "content_path": "README.md",
                    "parser": "markdown"
                }
            ),
            
            # Direct API discovery
            UpdateSource(
                name="openrouter_api",
                type="api",
                url="https://openrouter.ai/api/v1/models",
                update_interval=2,  # 2 hours
                config={
                    "auth_header": "Authorization",
                    "free_filter": lambda m: m.get("pricing", {}).get("prompt", 0) == 0
                }
            ),
            UpdateSource(
                name="groq_api",
                type="api", 
                url="https://api.groq.com/openai/v1/models",
                update_interval=4,
                config={
                    "auth_header": "Authorization",
                    "requires_key": True
                }
            ),
            UpdateSource(
                name="cerebras_api",
                type="api",
                url="https://api.cerebras.ai/v1/models", 
                update_interval=4,
                config={
                    "auth_header": "Authorization",
                    "requires_key": True
                }
            ),
            
            # Web scraping for provider websites
            UpdateSource(
                name="openrouter_website",
                type="web_scrape",
                url="https://openrouter.ai/models",
                update_interval=24,  # Daily
                config={
                    "selectors": {
                        "model_cards": ".model-card",
                        "model_name": ".model-name",
                        "pricing": ".pricing-info",
                        "free_badge": ".free-badge"
                    }
                }
            ),
            
            # Browser automation for dashboards (when needed)
            UpdateSource(
                name="provider_dashboards",
                type="browser",
                url="",  # Multiple URLs
                update_interval=168,  # Weekly
                enabled=False,  # Disabled by default
                config={
                    "providers": [
                        {
                            "name": "openrouter",
                            "login_url": "https://openrouter.ai/login",
                            "dashboard_url": "https://openrouter.ai/account",
                            "selectors": {
                                "usage_info": ".usage-stats",
                                "rate_limits": ".rate-limit-info"
                            }
                        }
                    ]
                }
            )
        ]
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                # Override defaults with config file
                sources_dict = {s.name: s for s in default_sources}
                for source_config in config_data.get('sources', []):
                    name = source_config['name']
                    if name in sources_dict:
                        # Update existing source
                        source = sources_dict[name]
                        for key, value in source_config.items():
                            if hasattr(source, key):
                                setattr(source, key, value)
                    else:
                        # Add new source
                        sources_dict[name] = UpdateSource(**source_config)
                return list(sources_dict.values())
        else:
            # Save default configuration
            self._save_update_sources(default_sources)
            return default_sources
    
    def _save_update_sources(self, sources: List[UpdateSource]):
        """Save update source configurations."""
        config_data = {
            'sources': [asdict(source) for source in sources]
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cached provider data."""
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cached provider data."""
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2, default=str)
    
    def add_update_callback(self, callback):
        """Add callback to be called when updates are found."""
        self.update_callbacks.append(callback)
    
    async def start_auto_update(self, interval_minutes: int = 60):
        """Start automatic update loop."""
        logger.info(f"Starting auto-update loop (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                await self.check_for_updates()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in auto-update loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def check_for_updates(self) -> List[ProviderUpdate]:
        """Check all sources for updates."""
        logger.info("Checking for provider updates...")
        
        updates = []
        current_time = datetime.now()
        
        for source in self.sources:
            if not source.enabled:
                continue
            
            # Check if update is needed
            if source.last_updated:
                time_since_update = current_time - source.last_updated
                if time_since_update.total_seconds() < source.update_interval * 3600:
                    continue
            
            try:
                logger.info(f"Updating from source: {source.name}")
                
                if source.type == "github":
                    source_updates = await self._update_from_github(source)
                elif source.type == "api":
                    source_updates = await self._update_from_api(source)
                elif source.type == "web_scrape":
                    source_updates = await self._update_from_web_scrape(source)
                elif source.type == "browser":
                    source_updates = await self._update_from_browser(source)
                else:
                    logger.warning(f"Unknown source type: {source.type}")
                    continue
                
                updates.extend(source_updates)
                source.last_updated = current_time
                
            except Exception as e:
                logger.error(f"Failed to update from {source.name}: {e}")
        
        # Save updated source configurations
        self._save_update_sources(self.sources)
        
        # Process and notify about updates
        if updates:
            await self._process_updates(updates)
        
        self.last_full_update = current_time
        return updates
    
    async def _update_from_github(self, source: UpdateSource) -> List[ProviderUpdate]:
        """Update from GitHub repository."""
        updates = []
        
        try:
            # Get repository info
            repo_url = source.url
            response = await self.http_client.get(repo_url)
            response.raise_for_status()
            repo_info = response.json()
            
            # Check if repository was updated since last check
            last_commit_date = repo_info.get("updated_at")
            cache_key = f"github_{source.name}_last_commit"
            
            if cache_key in self.cache and self.cache[cache_key] == last_commit_date:
                logger.debug(f"No updates in {source.name}")
                return updates
            
            # Fetch specific content files
            config = source.config or {}
            
            if "content_path" in config:
                content_url = f"{repo_url}/contents/{config['content_path']}"
                content_response = await self.http_client.get(content_url)
                content_response.raise_for_status()
                content_info = content_response.json()
                
                # Download and parse content
                download_url = content_info["download_url"]
                content_response = await self.http_client.get(download_url)
                content_response.raise_for_status()
                content = content_response.text
                
                # Parse based on parser type
                parser = config.get("parser", "markdown")
                if parser == "python_dict":
                    provider_updates = self._parse_python_dict_content(content, source.name)
                elif parser == "markdown":
                    provider_updates = self._parse_markdown_content(content, source.name)
                else:
                    logger.warning(f"Unknown parser: {parser}")
                    provider_updates = []
                
                updates.extend(provider_updates)
            
            # Update cache
            self.cache[cache_key] = last_commit_date
            
        except Exception as e:
            logger.error(f"Error updating from GitHub {source.name}: {e}")
        
        return updates
    
    async def _update_from_api(self, source: UpdateSource) -> List[ProviderUpdate]:
        """Update from direct API calls."""
        updates = []
        
        try:
            config = source.config or {}
            headers = {}
            
            # Add authentication if required
            if config.get("requires_key") and self.account_manager:
                provider_name = source.name.split("_")[0]  # Extract provider name
                credentials = await self.account_manager.get_credentials(provider_name)
                if credentials and config.get("auth_header"):
                    headers[config["auth_header"]] = f"Bearer {credentials[0].api_key}"
            
            # Make API request
            response = await self.http_client.get(source.url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Parse models
            models = data.get("data", []) if "data" in data else data
            provider_name = source.name.split("_")[0]
            
            # Filter for free models if specified
            if "free_filter" in config:
                filter_config = config["free_filter"]
                if callable(filter_config):
                    models = [m for m in models if filter_config(m)]
                elif isinstance(filter_config, str) and filter_config == "pricing.prompt == 0":
                    # Apply the specific logic for OpenRouter's known free_filter string
                    # This assumes 'models' is a list of dictionaries from JSON
                    models = [m for m in models if isinstance(m, dict) and m.get("pricing", {}).get("prompt", 0) == 0]
                elif isinstance(filter_config, str):
                    # Log a warning if it's a string but not the one we know how to handle
                    logger.warning(f"Unsupported string-based free_filter encountered for source {source.name}: {filter_config}")
            
            # Convert to ModelInfo objects
            new_models = []
            for model_data in models:
                model_info = self._convert_api_model_to_model_info(model_data, provider_name)
                if model_info:
                    new_models.append(model_info)
            
            # Compare with cached models
            cache_key = f"api_{source.name}_models"
            cached_models = self.cache.get(cache_key, [])
            
            # Find differences
            added_models, removed_models, updated_models = self._compare_model_lists(
                cached_models, new_models
            )
            
            if added_models or removed_models or updated_models:
                update = ProviderUpdate(
                    provider_name=provider_name,
                    models_added=added_models,
                    models_removed=removed_models,
                    models_updated=updated_models,
                    rate_limits_updated={},
                    timestamp=datetime.now()
                )
                updates.append(update)
            
            # Update cache
            self.cache[cache_key] = [asdict(m) for m in new_models]
            
        except Exception as e:
            logger.error(f"Error updating from API {source.name}: {e}")
        
        return updates
    
    async def _update_from_web_scrape(self, source: UpdateSource) -> List[ProviderUpdate]:
        """Update from web scraping."""
        updates = []
        
        try:
            response = await self.http_client.get(source.url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            config = source.config or {}
            selectors = config.get("selectors", {})
            
            # Extract model information
            models = []
            model_cards = soup.select(selectors.get("model_cards", ".model"))
            
            for card in model_cards:
                try:
                    name_elem = card.select_one(selectors.get("model_name", ".name"))
                    pricing_elem = card.select_one(selectors.get("pricing", ".price"))
                    free_badge = card.select_one(selectors.get("free_badge", ".free"))
                    
                    if name_elem and (free_badge or (pricing_elem and "free" in pricing_elem.text.lower())):
                        model_name = name_elem.text.strip()
                        
                        # Create ModelInfo
                        model_info = ModelInfo(
                            name=model_name,
                            display_name=model_name,
                            provider=provider_name,
                            capabilities=["text_generation"],
                            context_length=4096,  # Default
                            is_free=True
                        )
                        models.append(model_info)
                        
                except Exception as e:
                    logger.warning(f"Error parsing model card: {e}")
                    continue
            
            # Compare with cached data and create updates
            provider_name = source.name.split("_")[0]
            cache_key = f"scrape_{source.name}_models"
            cached_models = self.cache.get(cache_key, [])
            
            added_models, removed_models, updated_models = self._compare_model_lists(
                cached_models, models
            )
            
            if added_models or removed_models or updated_models:
                update = ProviderUpdate(
                    provider_name=provider_name,
                    models_added=added_models,
                    models_removed=removed_models,
                    models_updated=updated_models,
                    rate_limits_updated={},
                    timestamp=datetime.now()
                )
                updates.append(update)
            
            # Update cache
            self.cache[cache_key] = [asdict(m) for m in models]
            
        except Exception as e:
            logger.error(f"Error web scraping {source.name}: {e}")
        
        return updates
    
    async def _update_from_browser(self, source: UpdateSource) -> List[ProviderUpdate]:
        """Update using browser automation."""
        updates = []
        
        if not source.enabled:
            return updates
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                config = source.config or {}
                providers = config.get("providers", [])
                
                for provider_config in providers:
                    try:
                        provider_name = provider_config["name"]
                        
                        # Login if required
                        if "login_url" in provider_config and self.account_manager:
                            credentials = await self.account_manager.get_credentials(provider_name)
                            if credentials:
                                await self._browser_login(page, provider_config, credentials[0])
                        
                        # Navigate to dashboard
                        dashboard_url = provider_config["dashboard_url"]
                        await page.goto(dashboard_url)
                        await page.wait_for_load_state("networkidle")
                        
                        # Extract information
                        selectors = provider_config.get("selectors", {})
                        
                        # Get usage information
                        usage_info = {}
                        if "usage_info" in selectors:
                            usage_elem = await page.query_selector(selectors["usage_info"])
                            if usage_elem:
                                usage_text = await usage_elem.text_content()
                                usage_info = self._parse_usage_info(usage_text)
                        
                        # Get rate limit information
                        rate_limits = {}
                        if "rate_limits" in selectors:
                            rate_limit_elem = await page.query_selector(selectors["rate_limits"])
                            if rate_limit_elem:
                                rate_limit_text = await rate_limit_elem.text_content()
                                rate_limits = self._parse_rate_limits(rate_limit_text)
                        
                        # Create update if changes detected
                        cache_key = f"browser_{provider_name}_info"
                        cached_info = self.cache.get(cache_key, {})
                        
                        current_info = {
                            "usage_info": usage_info,
                            "rate_limits": rate_limits
                        }
                        
                        if current_info != cached_info:
                            update = ProviderUpdate(
                                provider_name=provider_name,
                                models_added=[],
                                models_removed=[],
                                models_updated=[],
                                rate_limits_updated=rate_limits,
                                timestamp=datetime.now()
                            )
                            updates.append(update)
                            
                            # Update cache
                            self.cache[cache_key] = current_info
                        
                    except Exception as e:
                        logger.error(f"Error with browser automation for {provider_config.get('name', 'unknown')}: {e}")
                
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error in browser automation: {e}")
        
        return updates
    
    async def _browser_login(self, page, provider_config, credentials):
        """Handle browser login for a provider."""
        try:
            login_url = provider_config["login_url"]
            await page.goto(login_url)
            
            # This would need to be customized per provider
            # For now, just a basic example
            await page.fill('input[type="email"]', credentials.username or "")
            await page.fill('input[type="password"]', credentials.api_key)  # Assuming API key is password
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("networkidle")
            
        except Exception as e:
            logger.error(f"Browser login failed: {e}")
    
    def _parse_python_dict_content(self, content: str, source_name: str) -> List[ProviderUpdate]:
        """Parse Python dictionary content from GitHub."""
        updates = []
        
        try:
            # Extract MODEL_TO_NAME_MAPPING dictionary
            if "MODEL_TO_NAME_MAPPING" in content:
                # Simple regex to extract the dictionary
                import re
                dict_match = re.search(r'MODEL_TO_NAME_MAPPING\s*=\s*{([^}]+)}', content, re.DOTALL)
                if dict_match:
                    dict_content = "{" + dict_match.group(1) + "}"
                    # This is a simplified parser - in production, use ast.literal_eval
                    logger.info(f"Found {len(dict_content)} characters of model mapping data")
                    
                    # For now, just log that we found updates
                    # In a full implementation, parse the dictionary and compare with cache
                    
        except Exception as e:
            logger.error(f"Error parsing Python dict content: {e}")
        
        return updates
    
    def _parse_markdown_content(self, content: str, source_name: str) -> List[ProviderUpdate]:
        """Parse markdown content from GitHub."""
        updates = []
        
        try:
            # Extract provider sections and model lists
            lines = content.split('\n')
            current_provider = None
            models = []
            
            for line in lines:
                line = line.strip()
                
                # Detect provider headers
                if line.startswith('###') and any(provider in line.lower() for provider in ['openrouter', 'groq', 'cerebras']):
                    if current_provider and models:
                        # Process previous provider
                        # Create update for previous provider
                        pass
                    
                    current_provider = line.replace('###', '').strip()
                    models = []
                
                # Detect model entries (usually bullet points with links)
                elif line.startswith('-') and 'http' in line:
                    # Extract model name from markdown link
                    import re
                    link_match = re.search(r'\[([^\]]+)\]', line)
                    if link_match:
                        model_name = link_match.group(1)
                        models.append(model_name)
            
            # Process last provider
            if current_provider and models:
                # Create update for last provider
                pass
                
        except Exception as e:
            logger.error(f"Error parsing markdown content: {e}")
        
        return updates
    
    def _convert_api_model_to_model_info(self, model_data: Dict, provider_name: str) -> Optional[ModelInfo]:
        """Convert API model data to ModelInfo object."""
        try:
            model_id = model_data.get("id", "")
            
            # Determine if model is free
            is_free = False
            if "pricing" in model_data:
                pricing = model_data["pricing"]
                is_free = pricing.get("prompt", 0) == 0 and pricing.get("completion", 0) == 0
            elif ":free" in model_id:
                is_free = True
            
            return ModelInfo(
                name=model_id,
                display_name=model_data.get("name", model_id),
                provider=provider_name,
                capabilities=["text_generation"],  # Default capability
                context_length=model_data.get("context_length", 4096),
                is_free=is_free
            )
            
        except Exception as e:
            logger.error(f"Error converting API model data: {e}")
            return None
    
    def _compare_model_lists(self, cached_models: List[Dict], new_models: List[ModelInfo]) -> tuple:
        """Compare model lists and return added, removed, updated models."""
        
        # Convert cached models to ModelInfo objects
        cached_model_infos = []
        for cached_model in cached_models:
            if isinstance(cached_model, dict):
                cached_model_infos.append(ModelInfo(**cached_model))
            else:
                cached_model_infos.append(cached_model)
        
        cached_names = {m.name for m in cached_model_infos}
        new_names = {m.name for m in new_models}
        
        # Find differences
        added_names = new_names - cached_names
        removed_names = cached_names - new_names
        common_names = cached_names & new_names
        
        added_models = [m for m in new_models if m.name in added_names]
        removed_models = list(removed_names)
        
        # Check for updates in common models
        updated_models = []
        cached_dict = {m.name: m for m in cached_model_infos}
        new_dict = {m.name: m for m in new_models}
        
        for name in common_names:
            cached_model = cached_dict[name]
            new_model = new_dict[name]
            
            # Compare relevant fields
            if (cached_model.display_name != new_model.display_name or
                cached_model.context_length != new_model.context_length or
                cached_model.is_free != new_model.is_free):
                updated_models.append(new_model)
        
        return added_models, removed_models, updated_models
    
    def _parse_usage_info(self, usage_text: str) -> Dict[str, Any]:
        """Parse usage information from text."""
        usage_info = {}
        
        # Extract common usage patterns
        import re
        
        # Requests per day/minute
        requests_match = re.search(r'(\d+)\s*requests?\s*per\s*(day|minute|hour)', usage_text, re.IGNORECASE)
        if requests_match:
            count = int(requests_match.group(1))
            period = requests_match.group(2).lower()
            usage_info[f"requests_per_{period}"] = count
        
        # Tokens per minute
        tokens_match = re.search(r'(\d+)\s*tokens?\s*per\s*(minute|hour)', usage_text, re.IGNORECASE)
        if tokens_match:
            count = int(tokens_match.group(1))
            period = tokens_match.group(2).lower()
            usage_info[f"tokens_per_{period}"] = count
        
        return usage_info
    
    def _parse_rate_limits(self, rate_limit_text: str) -> Dict[str, Any]:
        """Parse rate limit information from text."""
        return self._parse_usage_info(rate_limit_text)  # Same parsing logic
    
    async def _process_updates(self, updates: List[ProviderUpdate]):
        """Process and notify about updates."""
        logger.info(f"Processing {len(updates)} provider updates")
        
        for update in updates:
            logger.info(f"Provider {update.provider_name} updates:")
            if update.models_added:
                logger.info(f"  Added models: {[m.name for m in update.models_added]}")
            if update.models_removed:
                logger.info(f"  Removed models: {update.models_removed}")
            if update.models_updated:
                logger.info(f"  Updated models: {[m.name for m in update.models_updated]}")
            if update.rate_limits_updated:
                logger.info(f"  Rate limits updated: {update.rate_limits_updated}")
        
        # Call registered callbacks
        for callback in self.update_callbacks:
            try:
                await callback(updates)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
        
        # Save cache
        self._save_cache()
    
    async def force_update_all(self) -> List[ProviderUpdate]:
        """Force update from all sources regardless of intervals."""
        logger.info("Forcing update from all sources")
        
        # Reset last_updated for all sources
        for source in self.sources:
            source.last_updated = None
        
        return await self.check_for_updates()
    
    async def get_update_status(self) -> Dict[str, Any]:
        """Get current update status."""
        return {
            "last_full_update": self.last_full_update,
            "sources": [
                {
                    "name": source.name,
                    "type": source.type,
                    "enabled": source.enabled,
                    "last_updated": source.last_updated,
                    "next_update": (
                        source.last_updated + timedelta(hours=source.update_interval)
                        if source.last_updated else None
                    )
                }
                for source in self.sources
            ],
            "cache_size": len(self.cache)
        }
    
    async def close(self):
        """Close the auto-updater and cleanup resources."""
        await self.http_client.aclose()
        logger.info("AutoUpdater closed")


# Integration function for the main aggregator
async def integrate_auto_updater(aggregator, auto_updater: AutoUpdater):
    """Integrate auto-updater with the main aggregator."""
    
    async def update_callback(updates: List[ProviderUpdate]):
        """Callback to update aggregator when new data is available."""
        for update in updates:
            provider_name = update.provider_name
            
            if provider_name in aggregator.providers:
                provider = aggregator.providers[provider_name]
                
                # Update models
                if update.models_added:
                    for model in update.models_added:
                        provider.config.models.append(model)
                        logger.info(f"Added model {model.name} to {provider_name}")
                
                if update.models_removed:
                    provider.config.models = [
                        m for m in provider.config.models 
                        if m.name not in update.models_removed
                    ]
                    logger.info(f"Removed models {update.models_removed} from {provider_name}")
                
                if update.models_updated:
                    model_dict = {m.name: m for m in provider.config.models}
                    for updated_model in update.models_updated:
                        if updated_model.name in model_dict:
                            # Replace with updated model
                            idx = provider.config.models.index(model_dict[updated_model.name])
                            provider.config.models[idx] = updated_model
                            logger.info(f"Updated model {updated_model.name} in {provider_name}")
                
                # Update rate limits
                if update.rate_limits_updated:
                    # Update provider rate limits
                    provider.config.rate_limits.update(update.rate_limits_updated)
                    logger.info(f"Updated rate limits for {provider_name}")
                
                # Update meta-controller if available
                if hasattr(aggregator, 'meta_controller') and aggregator.meta_controller:
                    # Update model profiles for new/updated models
                    for model in update.models_added + update.models_updated:
                        profile = aggregator._create_model_capability_profile(model, provider_name)
                        aggregator.meta_controller.model_profiles[model.name] = profile
    
    # Register the callback
    auto_updater.add_update_callback(update_callback)
    
    return auto_updater