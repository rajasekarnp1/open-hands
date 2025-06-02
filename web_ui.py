#!/usr/bin/env python3
"""
Streamlit web interface for the LLM API Aggregator.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional

import httpx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# Page config
st.set_page_config(
    page_title="LLM API Aggregator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


class APIClient:
    """Client for interacting with the LLM API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    async def chat_completion(self, messages: List[Dict], model: str = "auto", provider: Optional[str] = None, **kwargs):
        """Send chat completion request."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            request_data = {
                "messages": messages,
                "model": model,
                **kwargs
            }
            if provider:
                request_data["provider"] = provider
            
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=request_data)
            response.raise_for_status()
            return response.json()
    
    async def list_models(self):
        """List available models."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
    
    async def get_provider_status(self):
        """Get provider status."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/admin/providers")
            response.raise_for_status()
            return response.json()
    
    async def get_usage_stats(self):
        """Get usage statistics."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/admin/usage-stats")
            response.raise_for_status()
            return response.json()
    
    async def health_check(self):
        """Perform health check."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
    
    async def add_credentials(self, provider: str, account_id: str, api_key: str, additional_headers: Optional[Dict] = None):
        """Add credentials."""
        async with httpx.AsyncClient() as client:
            data = {
                "provider": provider,
                "account_id": account_id,
                "api_key": api_key
            }
            if additional_headers:
                data["additional_headers"] = additional_headers
            
            response = await client.post(f"{self.base_url}/admin/credentials", json=data)
            response.raise_for_status()
            return response.json()


def run_async(coro):
    """Run async function in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


def main():
    """Main Streamlit app."""
    
    st.title("🤖 LLM API Aggregator")
    st.markdown("Multi-provider LLM API with intelligent routing")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # API URL configuration
    api_url = st.sidebar.text_input(
        "API URL",
        value="http://localhost:8000",
        help="URL of the LLM API Aggregator server"
    )
    
    client = APIClient(api_url)
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to",
        ["💬 Chat", "📊 Dashboard", "🔧 Settings", "📈 Analytics"]
    )
    
    if page == "💬 Chat":
        chat_page(client)
    elif page == "📊 Dashboard":
        dashboard_page(client)
    elif page == "🔧 Settings":
        settings_page(client)
    elif page == "📈 Analytics":
        analytics_page(client)


def chat_page(client: APIClient):
    """Chat interface page."""
    
    st.header("💬 Chat Interface")
    
    # Chat configuration
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        model = st.selectbox(
            "Model",
            ["auto"] + get_available_models(client),
            help="Select a specific model or 'auto' for automatic selection"
        )
    
    with col2:
        provider = st.selectbox(
            "Provider",
            ["auto"] + get_available_providers(client),
            help="Force a specific provider or 'auto' for intelligent routing"
        )
    
    with col3:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses"
        )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "provider" in message:
                st.caption(f"Provider: {message['provider']}")
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_async(client.chat_completion(
                        messages=st.session_state.messages,
                        model=model if model != "auto" else "auto",
                        provider=provider if provider != "auto" else None,
                        temperature=temperature
                    ))
                    
                    assistant_message = response["choices"][0]["message"]["content"]
                    provider_used = response["provider"]
                    
                    st.markdown(assistant_message)
                    st.caption(f"Provider: {provider_used}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_message,
                        "provider": provider_used
                    })
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


def dashboard_page(client: APIClient):
    """Dashboard page."""
    
    st.header("📊 Dashboard")
    
    try:
        # Get data
        provider_status = run_async(client.get_provider_status())
        health_data = run_async(client.health_check())
        
        # Provider status overview
        st.subheader("Provider Status")
        
        status_data = []
        for provider, status in provider_status.items():
            metrics = status.get("metrics", {})
            total_requests = metrics.get("total_requests", 0)
            successful_requests = metrics.get("successful_requests", 0)
            success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
            
            status_data.append({
                "Provider": provider,
                "Status": status["status"],
                "Available": "✅" if status["available"] else "❌",
                "Models": status["models_count"],
                "Accounts": status["credentials_count"],
                "Success Rate": f"{success_rate:.1f}%",
                "Total Requests": total_requests
            })
        
        df = pd.DataFrame(status_data)
        st.dataframe(df, use_container_width=True)
        
        # Health status
        st.subheader("Health Status")
        
        health_cols = st.columns(len(health_data["providers"]))
        for i, (provider, is_healthy) in enumerate(health_data["providers"].items()):
            with health_cols[i]:
                status_color = "green" if is_healthy else "red"
                status_icon = "✅" if is_healthy else "❌"
                st.metric(
                    label=provider.title(),
                    value=status_icon,
                    delta="Healthy" if is_healthy else "Unhealthy"
                )
        
        # Request distribution
        if any(data["Total Requests"] > 0 for data in status_data):
            st.subheader("Request Distribution")
            
            fig = px.pie(
                df,
                values="Total Requests",
                names="Provider",
                title="Requests by Provider"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")


def settings_page(client: APIClient):
    """Settings page."""
    
    st.header("🔧 Settings")
    
    # Credentials management
    st.subheader("API Credentials")
    
    with st.expander("Add New Credentials"):
        provider_options = [
            "openrouter",
            "groq", 
            "cerebras",
            "together",
            "cohere",
            "huggingface",
            "nvidia"
        ]
        
        provider = st.selectbox("Provider", provider_options)
        account_id = st.text_input("Account ID/Name")
        api_key = st.text_input("API Key", type="password")
        
        # Additional headers for specific providers
        additional_headers = {}
        if provider == "openrouter":
            app_name = st.text_input("App Name (optional)")
            if app_name:
                additional_headers["HTTP-Referer"] = f"https://{app_name}"
                additional_headers["X-Title"] = app_name
        
        if st.button("Add Credentials"):
            if provider and account_id and api_key:
                try:
                    result = run_async(client.add_credentials(
                        provider=provider,
                        account_id=account_id,
                        api_key=api_key,
                        additional_headers=additional_headers if additional_headers else None
                    ))
                    st.success(f"Added credentials for {provider}:{account_id}")
                except Exception as e:
                    st.error(f"Error adding credentials: {e}")
            else:
                st.error("Please fill in all required fields")
    
    # Provider configuration
    st.subheader("Provider Configuration")
    
    provider_info = {
        "openrouter": {
            "name": "OpenRouter",
            "description": "50+ free models including DeepSeek R1, Llama 3.3 70B",
            "signup_url": "https://openrouter.ai/"
        },
        "groq": {
            "name": "Groq",
            "description": "Ultra-fast inference with Llama, Gemma models",
            "signup_url": "https://console.groq.com/"
        },
        "cerebras": {
            "name": "Cerebras",
            "description": "Fast inference (8K context limit on free tier)",
            "signup_url": "https://cloud.cerebras.ai/"
        }
    }
    
    for provider_id, info in provider_info.items():
        with st.expander(f"{info['name']} - {info['description']}"):
            st.markdown(f"**Sign up:** [{info['signup_url']}]({info['signup_url']})")
            st.markdown(f"**Description:** {info['description']}")


def analytics_page(client: APIClient):
    """Analytics page."""
    
    st.header("📈 Analytics")
    
    try:
        # Get usage statistics
        stats = run_async(client.get_usage_stats())
        
        # Account usage
        st.subheader("Account Usage")
        
        account_usage = stats.get("account_usage", {})
        if account_usage:
            usage_data = []
            for provider, usage_info in account_usage.items():
                for account, count in usage_info.get("account_usage", {}).items():
                    usage_data.append({
                        "Provider": provider,
                        "Account": account,
                        "Requests": count
                    })
            
            if usage_data:
                df = pd.DataFrame(usage_data)
                
                # Usage by provider
                provider_usage = df.groupby("Provider")["Requests"].sum().reset_index()
                fig1 = px.bar(
                    provider_usage,
                    x="Provider",
                    y="Requests",
                    title="Total Requests by Provider"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Usage by account
                fig2 = px.bar(
                    df,
                    x="Account",
                    y="Requests",
                    color="Provider",
                    title="Requests by Account"
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Provider performance scores
        st.subheader("Provider Performance")
        
        provider_scores = stats.get("provider_scores", {})
        if provider_scores:
            scores_df = pd.DataFrame([
                {"Provider": provider, "Score": score}
                for provider, score in provider_scores.items()
            ])
            
            fig3 = px.bar(
                scores_df,
                x="Provider",
                y="Score",
                title="Provider Performance Scores",
                color="Score",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No performance data available yet")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")


@st.cache_data(ttl=60)
def get_available_models(client: APIClient) -> List[str]:
    """Get list of available models."""
    try:
        models_data = run_async(client.list_models())
        return [model["id"] for model in models_data["data"]]
    except:
        return []


@st.cache_data(ttl=60)
def get_available_providers(client: APIClient) -> List[str]:
    """Get list of available providers."""
    try:
        provider_status = run_async(client.get_provider_status())
        return list(provider_status.keys())
    except:
        return []


if __name__ == "__main__":
    main()