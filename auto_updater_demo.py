#!/usr/bin/env python3
"""
Auto-Updater Demo for LLM API Aggregator

Demonstrates the automatic discovery and updating of:
1. Free LLM providers from GitHub community projects
2. New models and rate limits via API discovery
3. Provider website monitoring via web scraping
4. Dashboard monitoring via browser automation (optional)

This system integrates with existing GitHub projects like:
- cheahjs/free-llm-api-resources
- zukixa/cool-ai-stuff
- wdhdev/free-for-life
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from src.core.auto_updater import AutoUpdater, UpdateSource
from src.core.browser_monitor import BrowserMonitor
from src.core.account_manager import AccountManager


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


async def demo_github_integration():
    """Demonstrate GitHub integration for community-maintained lists."""
    
    console.print("\n[bold blue]🐙 GitHub Integration Demo[/bold blue]")
    console.print("Monitoring community-maintained free LLM API lists...")
    
    # Create auto-updater
    auto_updater = AutoUpdater()
    
    # Show configured GitHub sources
    github_sources = [s for s in auto_updater.sources if s.type == "github"]
    
    table = Table(title="GitHub Sources")
    table.add_column("Repository", style="cyan")
    table.add_column("Update Interval", style="green")
    table.add_column("Content Path", style="yellow")
    table.add_column("Parser", style="magenta")
    
    for source in github_sources:
        config = source.config or {}
        table.add_row(
            source.name,
            f"{source.update_interval}h",
            config.get("content_path", "N/A"),
            config.get("parser", "N/A")
        )
    
    console.print(table)
    
    # Demonstrate checking for updates
    console.print("\n[yellow]Checking for updates from GitHub sources...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for source in github_sources[:2]:  # Check first 2 sources
            task = progress.add_task(f"Checking {source.name}...", total=None)
            
            try:
                updates = await auto_updater._update_from_github(source)
                progress.update(task, description=f"✅ {source.name} - {len(updates)} updates")
                
                if updates:
                    console.print(f"   Found {len(updates)} updates from {source.name}")
                
            except Exception as e:
                progress.update(task, description=f"❌ {source.name} - Error: {str(e)[:50]}")
    
    await auto_updater.close()


async def demo_api_discovery():
    """Demonstrate direct API discovery for new models."""
    
    console.print("\n[bold blue]🔍 API Discovery Demo[/bold blue]")
    console.print("Discovering models directly from provider APIs...")
    
    auto_updater = AutoUpdater()
    
    # Show API sources
    api_sources = [s for s in auto_updater.sources if s.type == "api"]
    
    table = Table(title="API Discovery Sources")
    table.add_column("Provider", style="cyan")
    table.add_column("Endpoint", style="green")
    table.add_column("Auth Required", style="yellow")
    table.add_column("Update Interval", style="magenta")
    
    for source in api_sources:
        config = source.config or {}
        table.add_row(
            source.name,
            source.url,
            "Yes" if config.get("requires_key") else "No",
            f"{source.update_interval}h"
        )
    
    console.print(table)
    
    # Demonstrate API discovery
    console.print("\n[yellow]Discovering models from APIs...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Try OpenRouter API (doesn't require auth)
        openrouter_source = next((s for s in api_sources if "openrouter" in s.name), None)
        
        if openrouter_source:
            task = progress.add_task("Discovering OpenRouter models...", total=None)
            
            try:
                updates = await auto_updater._update_from_api(openrouter_source)
                progress.update(task, description=f"✅ OpenRouter - {len(updates)} updates")
                
                if updates:
                    for update in updates:
                        console.print(f"   Provider: {update.provider_name}")
                        console.print(f"   Models added: {len(update.models_added)}")
                        console.print(f"   Models updated: {len(update.models_updated)}")
                
            except Exception as e:
                progress.update(task, description=f"❌ OpenRouter - Error: {str(e)[:50]}")
                console.print(f"   [red]Error: {e}[/red]")
    
    await auto_updater.close()


async def demo_web_scraping():
    """Demonstrate web scraping for provider websites."""
    
    console.print("\n[bold blue]🕷️ Web Scraping Demo[/bold blue]")
    console.print("Scraping provider websites for model information...")
    
    auto_updater = AutoUpdater()
    
    # Show web scraping sources
    scrape_sources = [s for s in auto_updater.sources if s.type == "web_scrape"]
    
    table = Table(title="Web Scraping Sources")
    table.add_column("Provider", style="cyan")
    table.add_column("URL", style="green")
    table.add_column("Update Interval", style="yellow")
    table.add_column("Selectors", style="magenta")
    
    for source in scrape_sources:
        config = source.config or {}
        selectors = config.get("selectors", {})
        table.add_row(
            source.name,
            source.url,
            f"{source.update_interval}h",
            f"{len(selectors)} configured"
        )
    
    console.print(table)
    
    # Demonstrate web scraping
    console.print("\n[yellow]Scraping provider websites...[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for source in scrape_sources[:1]:  # Try first source
            task = progress.add_task(f"Scraping {source.name}...", total=None)
            
            try:
                updates = await auto_updater._update_from_web_scrape(source)
                progress.update(task, description=f"✅ {source.name} - {len(updates)} updates")
                
                if updates:
                    for update in updates:
                        console.print(f"   Provider: {update.provider_name}")
                        console.print(f"   Models found: {len(update.models_added)}")
                
            except Exception as e:
                progress.update(task, description=f"❌ {source.name} - Error: {str(e)[:50]}")
                console.print(f"   [red]Note: Web scraping may fail due to website changes[/red]")
    
    await auto_updater.close()


async def demo_browser_monitoring():
    """Demonstrate browser automation for advanced monitoring."""
    
    console.print("\n[bold blue]🌐 Browser Monitoring Demo[/bold blue]")
    console.print("Using browser automation to monitor provider dashboards...")
    
    try:
        # Create browser monitor
        browser_monitor = BrowserMonitor()
        await browser_monitor.start()
        
        # Show provider configurations
        table = Table(title="Browser Monitoring Targets")
        table.add_column("Provider", style="cyan")
        table.add_column("Models Page", style="green")
        table.add_column("Dashboard", style="yellow")
        table.add_column("Login Required", style="magenta")
        
        for provider_name, config in browser_monitor.provider_configs.items():
            table.add_row(
                provider_name,
                config.models_page,
                config.dashboard_url or "N/A",
                "Yes" if config.login_url else "No"
            )
        
        console.print(table)
        
        # Demonstrate browser monitoring
        console.print("\n[yellow]Monitoring providers with browser automation...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Monitor OpenRouter (no login required)
            task = progress.add_task("Monitoring OpenRouter...", total=None)
            
            try:
                result = await browser_monitor.monitor_provider("openrouter")
                progress.update(task, description="✅ OpenRouter - Monitoring complete")
                
                if result.get("status") == "success":
                    models = result.get("models", [])
                    console.print(f"   Found {len(models)} models via browser monitoring")
                    
                    # Show sample models
                    if models:
                        sample_table = Table(title="Sample Models Found")
                        sample_table.add_column("Model Name", style="cyan")
                        sample_table.add_column("Free", style="green")
                        sample_table.add_column("Context Length", style="yellow")
                        
                        for model in models[:5]:  # Show first 5
                            sample_table.add_row(
                                model.get("name", "Unknown"),
                                "Yes" if model.get("is_free") else "No",
                                str(model.get("context_length", "Unknown"))
                            )
                        
                        console.print(sample_table)
                
            except Exception as e:
                progress.update(task, description=f"❌ OpenRouter - Error: {str(e)[:50]}")
                console.print(f"   [red]Browser monitoring error: {e}[/red]")
        
        await browser_monitor.stop()
        
    except Exception as e:
        console.print(f"[red]Browser monitoring not available: {e}[/red]")
        console.print("[yellow]Install playwright with: pip install playwright && playwright install[/yellow]")


async def demo_update_integration():
    """Demonstrate how updates are integrated into the main system."""
    
    console.print("\n[bold blue]🔄 Update Integration Demo[/bold blue]")
    console.print("Showing how discovered updates are integrated into the aggregator...")
    
    # Create auto-updater
    auto_updater = AutoUpdater()
    
    # Simulate some updates
    from src.core.auto_updater import ProviderUpdate
    from src.models import ModelInfo
    
    # Create mock updates
    mock_updates = [
        ProviderUpdate(
            provider_name="openrouter",
            models_added=[
                ModelInfo(
                    name="new-model-1",
                    display_name="New Model 1",
                    provider="openrouter",
                    capabilities=["text_generation"],
                    context_length=8192,
                    is_free=True
                )
            ],
            models_removed=["old-model-1"],
            models_updated=[],
            rate_limits_updated={"requests_per_day": 100},
            timestamp=datetime.now()
        ),
        ProviderUpdate(
            provider_name="groq",
            models_added=[],
            models_removed=[],
            models_updated=[
                ModelInfo(
                    name="existing-model",
                    display_name="Updated Model",
                    provider="groq",
                    capabilities=["text_generation", "code_generation"],
                    context_length=16384,  # Increased
                    is_free=True
                )
            ],
            rate_limits_updated={},
            timestamp=datetime.now()
        )
    ]
    
    # Show update summary
    table = Table(title="Discovered Updates")
    table.add_column("Provider", style="cyan")
    table.add_column("Models Added", style="green")
    table.add_column("Models Removed", style="red")
    table.add_column("Models Updated", style="yellow")
    table.add_column("Rate Limits", style="magenta")
    
    for update in mock_updates:
        table.add_row(
            update.provider_name,
            str(len(update.models_added)),
            str(len(update.models_removed)),
            str(len(update.models_updated)),
            "Yes" if update.rate_limits_updated else "No"
        )
    
    console.print(table)
    
    # Show integration process
    console.print("\n[yellow]Integration Process:[/yellow]")
    console.print("1. 🔍 Auto-updater discovers changes from multiple sources")
    console.print("2. 📊 Changes are validated and deduplicated")
    console.print("3. 🔄 Provider configurations are updated automatically")
    console.print("4. 🧠 Meta-controller model profiles are refreshed")
    console.print("5. 📝 Changes are logged and cached for future reference")
    console.print("6. 🔔 Notifications are sent (if configured)")
    
    # Show configuration update
    console.print("\n[green]✅ Updates would be automatically applied to:[/green]")
    console.print("   • Provider model lists")
    console.print("   • Rate limit configurations")
    console.print("   • Meta-controller capability profiles")
    console.print("   • Ensemble system model rankings")
    
    await auto_updater.close()


async def demo_monitoring_dashboard():
    """Create a live monitoring dashboard."""
    
    console.print("\n[bold blue]📊 Live Monitoring Dashboard[/bold blue]")
    console.print("Real-time monitoring of auto-updater status...")
    
    auto_updater = AutoUpdater()
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    layout["body"].split_row(
        Layout(name="sources"),
        Layout(name="status")
    )
    
    def create_header():
        return Panel(
            Text("🤖 LLM API Aggregator - Auto-Updater Dashboard", justify="center"),
            style="bold blue"
        )
    
    def create_sources_panel():
        table = Table(title="Update Sources")
        table.add_column("Source", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Last Update", style="magenta")
        
        for source in auto_updater.sources:
            status = "🟢 Enabled" if source.enabled else "🔴 Disabled"
            last_update = source.last_updated.strftime("%H:%M:%S") if source.last_updated else "Never"
            
            table.add_row(
                source.name,
                source.type,
                status,
                last_update
            )
        
        return Panel(table, title="Sources", border_style="green")
    
    def create_status_panel():
        status_text = Text()
        status_text.append("🔄 Auto-updater Status\n\n", style="bold")
        status_text.append(f"Sources configured: {len(auto_updater.sources)}\n")
        status_text.append(f"Enabled sources: {sum(1 for s in auto_updater.sources if s.enabled)}\n")
        status_text.append(f"Cache entries: {len(auto_updater.cache)}\n")
        status_text.append(f"Last full update: {auto_updater.last_full_update or 'Never'}\n")
        
        return Panel(status_text, title="Status", border_style="blue")
    
    def create_footer():
        return Panel(
            Text("Press Ctrl+C to exit", justify="center"),
            style="dim"
        )
    
    # Update layout
    layout["header"].update(create_header())
    layout["sources"].update(create_sources_panel())
    layout["status"].update(create_status_panel())
    layout["footer"].update(create_footer())
    
    # Show dashboard for a few seconds
    with Live(layout, console=console, refresh_per_second=1):
        await asyncio.sleep(5)
    
    await auto_updater.close()


async def main():
    """Run the complete auto-updater demo."""
    
    console.print(Panel.fit(
        "[bold green]🤖 LLM API Aggregator - Auto-Updater Demo[/bold green]\n\n"
        "This demo showcases automatic discovery and updating of:\n"
        "• Free LLM providers from GitHub community projects\n"
        "• New models and rate limits via API discovery\n"
        "• Provider websites via web scraping\n"
        "• Dashboard monitoring via browser automation\n\n"
        "[yellow]Integrates with existing projects:[/yellow]\n"
        "• cheahjs/free-llm-api-resources\n"
        "• zukixa/cool-ai-stuff\n"
        "• wdhdev/free-for-life",
        title="Auto-Updater Demo",
        border_style="blue"
    ))
    
    try:
        # Run all demos
        await demo_github_integration()
        await demo_api_discovery()
        await demo_web_scraping()
        await demo_browser_monitoring()
        await demo_update_integration()
        await demo_monitoring_dashboard()
        
        # Final summary
        console.print("\n" + "="*80)
        console.print(Panel.fit(
            "[bold green]✅ Auto-Updater Demo Complete![/bold green]\n\n"
            "[yellow]Key Features Demonstrated:[/yellow]\n"
            "🐙 GitHub integration with community projects\n"
            "🔍 Direct API discovery for real-time updates\n"
            "🕷️ Web scraping for provider websites\n"
            "🌐 Browser automation for advanced monitoring\n"
            "🔄 Automatic integration with main aggregator\n"
            "📊 Live monitoring and status dashboard\n\n"
            "[cyan]The system automatically:[/cyan]\n"
            "• Discovers new free models and providers\n"
            "• Updates rate limits and pricing information\n"
            "• Integrates changes into the aggregator\n"
            "• Maintains compatibility with existing APIs\n"
            "• Provides real-time monitoring and alerts\n\n"
            "[green]Ready for production deployment! 🚀[/green]",
            title="Demo Summary",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        logger.exception("Demo failed")


if __name__ == "__main__":
    asyncio.run(main())