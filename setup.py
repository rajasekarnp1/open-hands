#!/usr/bin/env python3
"""
Setup and configuration script for the LLM API Aggregator.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from getpass import getpass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.account_manager import AccountManager


async def setup_credentials():
    """Interactive setup for provider credentials."""
    
    print("🚀 LLM API Aggregator Setup")
    print("=" * 50)
    
    # Initialize account manager
    account_manager = AccountManager()
    
    providers = {
        "openrouter": {
            "name": "OpenRouter",
            "signup_url": "https://openrouter.ai/",
            "description": "50+ free models including DeepSeek R1, Llama 3.3 70B, Qwen 2.5 72B"
        },
        "groq": {
            "name": "Groq",
            "signup_url": "https://console.groq.com/",
            "description": "Ultra-fast inference with Llama, Gemma, and DeepSeek models"
        },
        "cerebras": {
            "name": "Cerebras",
            "signup_url": "https://cloud.cerebras.ai/",
            "description": "Fast inference with Llama and Qwen models (8K context limit on free tier)"
        },
        "together": {
            "name": "Together AI",
            "signup_url": "https://together.ai/",
            "description": "Free tier + $1 credit with payment method"
        },
        "cohere": {
            "name": "Cohere",
            "signup_url": "https://cohere.com/",
            "description": "Command models with 1,000 requests/month free"
        },
        "anthropic": {
            "name": "Anthropic",
            "signup_url": "https://console.anthropic.com/",
            "description": "Claude 3 models (Haiku, Sonnet, Opus)"
        }
    }
    
    print("\nAvailable Providers:")
    for provider_id, info in providers.items():
        print(f"\n📡 {info['name']}")
        print(f"   {info['description']}")
        print(f"   Sign up: {info['signup_url']}")
    
    print("\n" + "=" * 50)
    print("Let's add your API credentials!")
    print("You can add multiple accounts per provider for better rate limits.")
    
    for provider_id, info in providers.items():
        print(f"\n🔧 Setting up {info['name']}")
        
        while True:
            add_account = input(f"Add an account for {info['name']}? (y/n): ").lower().strip()
            if add_account in ['n', 'no']:
                break
            elif add_account not in ['y', 'yes']:
                print("Please enter 'y' or 'n'")
                continue
            
            account_id = input(f"Account ID/Name for {info['name']}: ").strip()
            if not account_id:
                print("Account ID cannot be empty")
                continue
            
            api_key = getpass(f"API Key for {info['name']}: ").strip()
            if not api_key:
                print("API Key cannot be empty")
                continue
            
            # Add additional headers if needed
            additional_headers = {}
            if provider_id == "openrouter":
                app_name = input("App name for OpenRouter (optional): ").strip()
                if app_name:
                    additional_headers["HTTP-Referer"] = f"https://{app_name}"
                    additional_headers["X-Title"] = app_name
            
            try:
                await account_manager.add_credentials(
                    provider=provider_id,
                    account_id=account_id,
                    api_key=api_key,
                    additional_headers=additional_headers if additional_headers else None
                )
                print(f"✅ Added credentials for {info['name']}:{account_id}")
                
                # Ask if they want to add another account for this provider
                another = input(f"Add another account for {info['name']}? (y/n): ").lower().strip()
                if another not in ['y', 'yes']:
                    break
                    
            except Exception as e:
                print(f"❌ Error adding credentials: {e}")
    
    print("\n🎉 Setup complete!")
    print("\nYou can now start the server with:")
    print("  python main.py")
    print("\nOr with custom settings:")
    print("  python main.py --host 0.0.0.0 --port 8000")
    
    # Show summary
    credentials = await account_manager.list_credentials()
    if credentials:
        print(f"\n📊 Summary:")
        for provider, accounts in credentials.items():
            print(f"  {provider}: {len(accounts)} account(s)")


def create_example_config():
    """Create example configuration files."""
    
    # Create example environment file
    env_content = """# LLM API Aggregator Configuration

# Server settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Security
ENCRYPTION_KEY=your-encryption-key-here

# Rate limiting
GLOBAL_REQUESTS_PER_MINUTE=100
USER_REQUESTS_PER_MINUTE=10
MAX_CONCURRENT_REQUESTS=50

# Provider settings
DEFAULT_PROVIDER=auto
ENABLE_CACHING=true
CACHE_TTL=3600
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    print("📝 Created .env.example file")


def main():
    """Main setup function."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "configure":
        # Interactive credential setup
        asyncio.run(setup_credentials())
    else:
        # Create example files
        create_example_config()
        
        print("🚀 LLM API Aggregator Setup")
        print("=" * 50)
        print("\nTo configure your API credentials, run:")
        print("  python setup.py configure")
        print("\nTo start the server:")
        print("  python main.py")
        print("\nFor help:")
        print("  python main.py --help")


if __name__ == "__main__":
    main()