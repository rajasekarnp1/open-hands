#!/usr/bin/env python3
"""
Demonstration script for the LLM API Aggregator.
Shows key features and capabilities.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import ChatMessage, ChatCompletionRequest
from src.core.aggregator import LLMAggregator
from src.core.account_manager import AccountManager
from src.core.router import ProviderRouter
from src.core.rate_limiter import RateLimiter
from src.providers.openrouter import create_openrouter_provider
from src.providers.groq import create_groq_provider
from src.providers.cerebras import create_cerebras_provider

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


console = Console()


async def demo_basic_functionality():
    """Demonstrate basic LLM aggregator functionality."""
    
    console.print(Panel.fit(
        "🚀 LLM API Aggregator Demo\n"
        "This demo shows the key features of the system",
        title="Demo Start"
    ))
    
    # Initialize components
    console.print("\n[bold blue]1. Initializing Components[/bold blue]")
    
    account_manager = AccountManager()
    rate_limiter = RateLimiter()
    
    # Create providers (with empty credentials for demo)
    providers = []
    
    console.print("   📡 Creating OpenRouter provider...")
    openrouter = create_openrouter_provider([])
    providers.append(openrouter)
    
    console.print("   ⚡ Creating Groq provider...")
    groq = create_groq_provider([])
    providers.append(groq)
    
    console.print("   🧠 Creating Cerebras provider...")
    cerebras = create_cerebras_provider([])
    providers.append(cerebras)
    
    # Create provider configs
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
    
    console.print("   ✅ All components initialized successfully!")
    
    # Show available models
    console.print("\n[bold blue]2. Available Models[/bold blue]")
    
    models_by_provider = await aggregator.list_available_models()
    
    for provider_name, models in models_by_provider.items():
        table = Table(title=f"{provider_name.title()} Models")
        table.add_column("Model", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Context", style="yellow")
        table.add_column("Free", style="magenta")
        
        for model in models[:3]:  # Show first 3 models
            context = f"{model.context_length:,}" if model.context_length else "N/A"
            is_free = "✅" if model.is_free else "❌"
            
            table.add_row(
                model.name,
                model.display_name,
                context,
                is_free
            )
        
        console.print(table)
        console.print()
    
    # Show provider status
    console.print("[bold blue]3. Provider Status[/bold blue]")
    
    status = await aggregator.get_provider_status()
    
    status_table = Table(title="Provider Status")
    status_table.add_column("Provider", style="cyan")
    status_table.add_column("Available", style="green")
    status_table.add_column("Models", style="yellow")
    status_table.add_column("Status", style="blue")
    
    for provider_name, provider_status in status.items():
        available = "✅" if provider_status["available"] else "❌"
        
        status_table.add_row(
            provider_name,
            available,
            str(provider_status["models_count"]),
            provider_status["status"]
        )
    
    console.print(status_table)
    
    # Demonstrate routing logic
    console.print("\n[bold blue]4. Intelligent Routing Demo[/bold blue]")
    
    test_requests = [
        {
            "content": "Write a Python function to sort a list",
            "expected_capability": "Code Generation"
        },
        {
            "content": "Think step by step and solve this math problem: 2x + 5 = 15",
            "expected_capability": "Reasoning"
        },
        {
            "content": "Tell me a short story about a robot",
            "expected_capability": "Text Generation"
        }
    ]
    
    for i, test_req in enumerate(test_requests, 1):
        console.print(f"\n   [bold]Test {i}:[/bold] {test_req['content']}")
        console.print(f"   [dim]Expected capability: {test_req['expected_capability']}[/dim]")
        
        request = ChatCompletionRequest(
            model="auto",
            messages=[ChatMessage(role="user", content=test_req["content"])]
        )
        
        # Get provider chain
        provider_chain = await router.get_provider_chain(request)
        console.print(f"   [green]Provider chain: {' → '.join(provider_chain)}[/green]")
    
    # Show rate limiting
    console.print("\n[bold blue]5. Rate Limiting Demo[/bold blue]")
    
    rate_status = rate_limiter.get_rate_limit_status()
    
    console.print(f"   Global requests per minute: {rate_status['global']['requests_per_minute']['remaining']}/{rate_status['global']['requests_per_minute']['limit']}")
    console.print(f"   Concurrent requests: {rate_status['global']['concurrent_requests']['remaining']}/{rate_status['global']['concurrent_requests']['limit']}")
    
    # Demonstrate account management
    console.print("\n[bold blue]6. Account Management Demo[/bold blue]")
    
    console.print("   📝 Adding demo credentials...")
    
    # Add some demo credentials (these won't work but show the structure)
    await account_manager.add_credentials(
        provider="openrouter",
        account_id="demo_account_1",
        api_key="sk-or-demo-key-1",
        additional_headers={"HTTP-Referer": "https://demo-app.com"}
    )
    
    await account_manager.add_credentials(
        provider="groq",
        account_id="demo_account_1",
        api_key="gsk_demo_key_1"
    )
    
    credentials = await account_manager.list_credentials()
    
    cred_table = Table(title="Stored Credentials")
    cred_table.add_column("Provider", style="cyan")
    cred_table.add_column("Account ID", style="green")
    cred_table.add_column("Status", style="yellow")
    cred_table.add_column("Usage", style="blue")
    
    for provider, accounts in credentials.items():
        for account in accounts:
            status = "✅ Active" if account["is_active"] else "❌ Inactive"
            
            cred_table.add_row(
                provider,
                account["account_id"],
                status,
                str(account["usage_count"])
            )
    
    console.print(cred_table)
    
    # Show usage statistics
    console.print("\n[bold blue]7. Usage Statistics[/bold blue]")
    
    usage_stats = await account_manager.get_usage_stats()
    
    for provider, stats in usage_stats.items():
        console.print(f"   [cyan]{provider}:[/cyan]")
        console.print(f"     Total usage: {stats['total_usage']}")
        console.print(f"     Active accounts: {stats['active_accounts']}")
    
    # Cleanup
    await aggregator.close()
    
    console.print(Panel.fit(
        "✅ Demo completed successfully!\n\n"
        "Key features demonstrated:\n"
        "• Multi-provider support with 20+ models\n"
        "• Intelligent routing based on content analysis\n"
        "• Secure credential management with encryption\n"
        "• Rate limiting and quota management\n"
        "• Real-time monitoring and analytics\n\n"
        "To get started:\n"
        "1. Run 'python setup.py configure' to add your API keys\n"
        "2. Run 'python main.py' to start the server\n"
        "3. Use the CLI, web UI, or API endpoints",
        title="Demo Complete"
    ))


async def demo_api_compatibility():
    """Demonstrate OpenAI API compatibility."""
    
    console.print(Panel.fit(
        "🔌 OpenAI API Compatibility Demo\n"
        "Shows how the aggregator works as a drop-in replacement",
        title="API Compatibility"
    ))
    
    # Example requests that would work with OpenAI API
    example_requests = [
        {
            "name": "Basic Chat Completion",
            "request": {
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            }
        },
        {
            "name": "Chat with System Message",
            "request": {
                "model": "auto",
                "messages": [
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        },
        {
            "name": "Streaming Response",
            "request": {
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "Tell me a story"}
                ],
                "stream": True
            }
        },
        {
            "name": "Provider-Specific Request",
            "request": {
                "model": "llama-3.3-70b-versatile",
                "provider": "groq",
                "messages": [
                    {"role": "user", "content": "Explain quantum computing"}
                ],
                "temperature": 0.3
            }
        }
    ]
    
    for example in example_requests:
        console.print(f"\n[bold green]{example['name']}[/bold green]")
        console.print("[dim]Request:[/dim]")
        console.print(json.dumps(example["request"], indent=2))
        console.print()


def main():
    """Main demo function."""
    
    console.print("[bold]🤖 LLM API Aggregator - Complete Demo[/bold]")
    console.print()
    
    # Run basic functionality demo
    asyncio.run(demo_basic_functionality())
    
    console.print("\n" + "="*60 + "\n")
    
    # Show API compatibility
    asyncio.run(demo_api_compatibility())
    
    console.print(Panel.fit(
        "🎉 Thank you for trying the LLM API Aggregator!\n\n"
        "Next steps:\n"
        "• Check out USAGE.md for detailed instructions\n"
        "• Run the web interface: streamlit run web_ui.py\n"
        "• Try the CLI: python cli.py chat\n"
        "• Deploy with Docker: docker-compose up\n\n"
        "Happy coding! 🚀",
        title="Demo End"
    ))


if __name__ == "__main__":
    main()