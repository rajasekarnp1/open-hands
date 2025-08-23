#!/usr/bin/env python3
"""
Extensible Plugin System

Allows dynamic loading and management of OpenHands plugins.
"""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    entry_point: str

class PluginBase(ABC):
    """Base class for all plugins."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Get plugin information."""
        pass

class PluginManager:
    """Manage OpenHands plugins."""

    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugin_dir = plugin_dir or Path("plugins")
        self.loaded_plugins: Dict[str, PluginBase] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.hooks: Dict[str, List[Callable]] = {}

    async def discover_plugins(self) -> List[PluginInfo]:
        """Discover available plugins."""

        plugins = []

        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return plugins

        for plugin_path in self.plugin_dir.iterdir():
            if plugin_path.is_dir() and (plugin_path / "__init__.py").exists():
                try:
                    plugin_info = await self._load_plugin_info(plugin_path)
                    if plugin_info:
                        plugins.append(plugin_info)
                except Exception as e:
                    logger.error(f"Error discovering plugin {plugin_path.name}: {e}")

        logger.info(f"Discovered {len(plugins)} plugins")
        return plugins

    async def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load and initialize a plugin."""

        if plugin_name in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} is already loaded")
            return True

        try:
            # Import plugin module
            plugin_module = importlib.import_module(f"plugins.{plugin_name}")

            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(plugin_module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginBase) and
                    obj != PluginBase):
                    plugin_class = obj
                    break

            if not plugin_class:
                logger.error(f"No plugin class found in {plugin_name}")
                return False

            # Create and initialize plugin instance
            plugin_instance = plugin_class()
            plugin_config = config or self.plugin_configs.get(plugin_name, {})

            if await plugin_instance.initialize(plugin_config):
                self.loaded_plugins[plugin_name] = plugin_instance
                logger.info(f"Successfully loaded plugin: {plugin_name}")
                return True
            else:
                logger.error(f"Failed to initialize plugin: {plugin_name}")
                return False

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""

        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} is not loaded")
            return True

        try:
            plugin = self.loaded_plugins[plugin_name]
            await plugin.cleanup()
            del self.loaded_plugins[plugin_name]

            logger.info(f"Successfully unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin."""

        config = self.plugin_configs.get(plugin_name, {})

        if plugin_name in self.loaded_plugins:
            await self.unload_plugin(plugin_name)

        return await self.load_plugin(plugin_name, config)

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback."""

        if hook_name not in self.hooks:
            self.hooks[hook_name] = []

        self.hooks[hook_name].append(callback)
        logger.debug(f"Registered hook: {hook_name}")

    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Trigger all callbacks for a hook."""

        if hook_name not in self.hooks:
            return []

        results = []

        for callback in self.hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {hook_name} callback: {e}")

        return results

    async def _load_plugin_info(self, plugin_path: Path) -> Optional[PluginInfo]:
        """Load plugin information from plugin.json."""

        info_file = plugin_path / "plugin.json"
        if not info_file.exists():
            return None

        try:
            import json
            with open(info_file, 'r') as f:
                info_data = json.load(f)

            return PluginInfo(
                name=info_data['name'],
                version=info_data['version'],
                description=info_data['description'],
                author=info_data['author'],
                dependencies=info_data.get('dependencies', []),
                entry_point=info_data['entry_point']
            )
        except Exception as e:
            logger.error(f"Error loading plugin info from {info_file}: {e}")
            return None

    def get_loaded_plugins(self) -> Dict[str, PluginInfo]:
        """Get information about loaded plugins."""

        return {
            name: plugin.get_info()
            for name, plugin in self.loaded_plugins.items()
        }

    async def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin."""

        self.plugin_configs[plugin_name] = config

        # If plugin is loaded, reconfigure it
        if plugin_name in self.loaded_plugins:
            return await self.reload_plugin(plugin_name)

        return True

# Global plugin manager
global_plugin_manager = PluginManager()

# Example plugin hooks
HOOK_BEFORE_TASK_EXECUTION = "before_task_execution"
HOOK_AFTER_TASK_EXECUTION = "after_task_execution"
HOOK_ON_ERROR = "on_error"
HOOK_ON_STARTUP = "on_startup"
HOOK_ON_SHUTDOWN = "on_shutdown"
