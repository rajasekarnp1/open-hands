#!/usr/bin/env python3
"""
Command-line interface for the LLM API Aggregator.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import ChatCompletionRequest, ChatMessage


console = Console()


class LLMClient:
    """Client for interacting with the LLM API Aggregator."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = "auto",
        provider: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ):
        """Send chat completion request."""
        
        request_data = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "model": model,
            "stream": stream,
            **kwargs
        }
        
        if provider:
            request_data["provider"] = provider
        
        url = f"{self.base_url}/v1/chat/completions"
        
        if stream:
            async with self.client.stream("POST", url, json=request_data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
        else:
            response = await self.client.post(url, json=request_data)
            response.raise_for_status()
            return response.json()
    
    async def list_models(self):
        """List available models."""
        response = await self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()
    
    async def get_provider_status(self):
        """Get provider status."""
        response = await self.client.get(f"{self.base_url}/admin/providers")
        response.raise_for_status()
        return response.json()
    
    async def get_usage_stats(self):
        """Get usage statistics."""
        response = await self.client.get(f"{self.base_url}/admin/usage-stats")
        response.raise_for_status()
        return response.json()
    
    async def health_check(self):
        """Perform health check."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


async def chat_interactive(client: LLMClient, model: str, provider: Optional[str]):
    """Interactive chat session."""
    
    console.print(Panel.fit(
        f"🤖 Interactive Chat Session\n"
        f"Model: {model}\n"
        f"Provider: {provider or 'auto'}\n"
        f"Type 'exit' to quit, 'clear' to clear history",
        title="LLM Chat"
    ))
    
    messages = []
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.lower() == 'clear':
                messages = []
                console.print("[dim]Chat history cleared[/dim]")
                continue
            elif not user_input.strip():
                continue
            
            # Add user message
            messages.append(ChatMessage(role="user", content=user_input))
            
            # Show thinking indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                
                try:
                    # Get response
                    response = await client.chat_completion(
                        messages=messages,
                        model=model,
                        provider=provider
                    )
                    
                    progress.remove_task(task)
                    
                    # Extract assistant response
                    assistant_message = response["choices"][0]["message"]["content"]
                    
                    # Display response
                    console.print(f"\n[bold green]Assistant[/bold green] ([dim]{response['provider']}[/dim]):")
                    console.print(assistant_message)
                    
                    # Add to message history
                    messages.append(ChatMessage(role="assistant", content=assistant_message))
                    
                except Exception as e:
                    progress.remove_task(task)
                    console.print(f"[red]Error: {e}[/red]")
        
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    console.print("\n[dim]Chat session ended[/dim]")


async def chat_single(client: LLMClient, message: str, model: str, provider: Optional[str], stream: bool):
    """Single chat completion."""
    
    messages = [ChatMessage(role="user", content=message)]
    
    if stream:
        console.print(f"[bold green]Assistant[/bold green]:")
        
        async for chunk in client.chat_completion(
            messages=messages,
            model=model,
            provider=provider,
            stream=True
        ):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    console.print(content, end="")
        
        console.print()  # New line at end
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            
            response = await client.chat_completion(
                messages=messages,
                model=model,
                provider=provider
            )
            
            progress.remove_task(task)
            
            assistant_message = response["choices"][0]["message"]["content"]
            provider_used = response["provider"]
            
            console.print(f"[bold green]Assistant[/bold green] ([dim]{provider_used}[/dim]):")
            console.print(assistant_message)


async def list_models_command(client: LLMClient):
    """List available models."""
    
    try:
        models_data = await client.list_models()
        models = models_data["data"]
        
        # Group by provider
        providers = {}
        for model in models:
            provider = model["owned_by"]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)
        
        for provider, provider_models in providers.items():
            table = Table(title=f"{provider.title()} Models")
            table.add_column("Model ID", style="cyan")
            table.add_column("Display Name", style="green")
            table.add_column("Capabilities", style="yellow")
            table.add_column("Context", style="blue")
            table.add_column("Free", style="magenta")
            
            for model in provider_models:
                capabilities = ", ".join(model.get("capabilities", []))
                context = str(model.get("context_length", "N/A"))
                is_free = "✅" if model.get("is_free", False) else "❌"
                
                table.add_row(
                    model["id"],
                    model.get("display_name", model["id"]),
                    capabilities,
                    context,
                    is_free
                )
            
            console.print(table)
            console.print()
    
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")


async def status_command(client: LLMClient):
    """Show provider status."""
    
    try:
        # Get provider status
        provider_status = await client.get_provider_status()
        
        table = Table(title="Provider Status")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Available", style="yellow")
        table.add_column("Models", style="blue")
        table.add_column("Accounts", style="magenta")
        table.add_column("Success Rate", style="white")
        
        for provider, status in provider_status.items():
            available = "✅" if status["available"] else "❌"
            metrics = status.get("metrics", {})
            
            # Calculate success rate
            total_requests = metrics.get("total_requests", 0)
            successful_requests = metrics.get("successful_requests", 0)
            success_rate = f"{(successful_requests / total_requests * 100):.1f}%" if total_requests > 0 else "N/A"
            
            table.add_row(
                provider,
                status["status"],
                available,
                str(status["models_count"]),
                str(status["credentials_count"]),
                success_rate
            )
        
        console.print(table)
        
        # Show health check
        console.print("\n[bold]Health Check:[/bold]")
        health = await client.health_check()
        for provider, is_healthy in health["providers"].items():
            status_icon = "✅" if is_healthy else "❌"
            console.print(f"  {provider}: {status_icon}")
    
    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")


async def stats_command(client: LLMClient):
    """Show usage statistics."""
    
    try:
        stats = await client.get_usage_stats()
        
        # Account usage
        console.print("[bold]Account Usage:[/bold]")
        account_usage = stats.get("account_usage", {})
        
        for provider, usage_data in account_usage.items():
            console.print(f"\n[cyan]{provider}:[/cyan]")
            console.print(f"  Total Usage: {usage_data['total_usage']}")
            console.print(f"  Active Accounts: {usage_data['active_accounts']}")
            
            for account, count in usage_data.get("account_usage", {}).items():
                console.print(f"    {account}: {count} requests")
        
        # Provider scores
        console.print("\n[bold]Provider Performance Scores:[/bold]")
        provider_scores = stats.get("provider_scores", {})
        
        if provider_scores:
            for provider, score in provider_scores.items():
                console.print(f"  {provider}: {score:.2f}")
        else:
            console.print("  No performance data yet")
    
    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")


async def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(description="LLM API Aggregator CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start chat session")
    chat_parser.add_argument("--model", default="auto", help="Model to use")
    chat_parser.add_argument("--provider", help="Specific provider to use")
    chat_parser.add_argument("--message", help="Single message (non-interactive)")
    chat_parser.add_argument("--stream", action="store_true", help="Stream response")
    
    # Models command
    subparsers.add_parser("models", help="List available models")
    
    # Status command
    subparsers.add_parser("status", help="Show provider status")
    
    # Stats command
    subparsers.add_parser("stats", help="Show usage statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create client
    client = LLMClient(args.url)
    
    try:
        if args.command == "chat":
            if args.message:
                await chat_single(client, args.message, args.model, args.provider, args.stream)
            else:
                await chat_interactive(client, args.model, args.provider)
        
        elif args.command == "models":
            await list_models_command(client)
        
        elif args.command == "status":
            await status_command(client)
        
        elif args.command == "stats":
            await stats_command(client)
    
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())