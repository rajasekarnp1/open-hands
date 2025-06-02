# Critical Fixes Implementation

## 🚨 Fix 1: Make PyTorch Optional

The main blocker is the hard dependency on PyTorch. Here's the fix:

### Modified meta_controller.py
```python
"""
Meta-Model Controller for Intelligent Model Selection (Optional PyTorch)
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import asyncio
from collections import defaultdict
import sqlite3
import pickle

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass

from ..models import ChatCompletionRequest, ModelInfo, ModelCapability

class MetaModelController:
    """Meta-controller with optional PyTorch support."""
    
    def __init__(self, enable_ml_features: bool = None):
        if enable_ml_features is None:
            enable_ml_features = TORCH_AVAILABLE
            
        self.ml_enabled = enable_ml_features and TORCH_AVAILABLE
        
        if not self.ml_enabled:
            print("Warning: PyTorch not available. Using fallback model selection.")
            
    def select_model(self, request: ChatCompletionRequest, available_models: List[ModelInfo]) -> ModelInfo:
        """Select best model with or without ML features."""
        if self.ml_enabled:
            return self._ml_model_selection(request, available_models)
        else:
            return self._fallback_model_selection(request, available_models)
    
    def _fallback_model_selection(self, request: ChatCompletionRequest, available_models: List[ModelInfo]) -> ModelInfo:
        """Simple rule-based model selection."""
        # Prioritize by capabilities and context length
        scored_models = []
        
        for model in available_models:
            score = 0
            
            # Prefer free models
            if model.is_free:
                score += 10
                
            # Prefer models with higher context length
            score += min(model.context_length / 1000, 10)
            
            # Check for specific capabilities
            message_content = " ".join([msg.get("content", "") for msg in request.messages])
            
            if "code" in message_content.lower():
                if ModelCapability.CODE_GENERATION in model.capabilities:
                    score += 5
                    
            if "image" in message_content.lower() or "vision" in message_content.lower():
                if ModelCapability.VISION in model.capabilities:
                    score += 5
                    
            scored_models.append((score, model))
        
        # Return highest scoring model
        scored_models.sort(key=lambda x: x[0], reverse=True)
        return scored_models[0][1] if scored_models else available_models[0]
```

## 🔒 Fix 2: Security Improvements

### Environment Configuration (.env.example)
```env
# Security
ADMIN_TOKEN=your-secure-admin-token-here
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Database
DATABASE_URL=sqlite:///./app.db
REDIS_URL=redis://localhost:6379

# Encryption
ENCRYPTION_KEY=your-32-byte-encryption-key-here
```

### Updated server.py security
```python
import os
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Security configuration
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

if not ADMIN_TOKEN:
    raise ValueError("ADMIN_TOKEN environment variable is required")

# Updated CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Admin authentication
async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return credentials.credentials

# Apply to admin endpoints
@app.post("/admin/credentials")
async def add_credentials(
    provider: str,
    account_id: str,
    api_key: str,
    additional_headers: Optional[Dict[str, str]] = None,
    _: str = Depends(verify_admin_token)  # Require admin auth
):
    # ... existing code
```

## 🧪 Fix 3: Test Configuration

### pytest.ini
```ini
[tool:pytest]
asyncio_default_fixture_loop_scope = function
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

### conftest.py
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def mock_aggregator():
    """Mock LLM Aggregator for testing."""
    aggregator = AsyncMock()
    aggregator.chat_completion.return_value = {
        "id": "test-123",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Test response"}}]
    }
    return aggregator
```

## 🔧 Fix 4: Error Handling

### Enhanced aggregator.py
```python
import logging
from typing import Optional, List
from ..providers.base import ProviderError, RateLimitError, AuthenticationError

logger = logging.getLogger(__name__)

class LLMAggregator:
    async def chat_completion(self, request: ChatCompletionRequest, user_id: Optional[str] = None):
        """Chat completion with comprehensive error handling."""
        
        # Get available providers
        try:
            available_providers = await self._get_available_providers(request)
        except Exception as e:
            logger.error(f"Failed to get available providers: {e}")
            raise HTTPException(status_code=503, detail="No providers available")
        
        if not available_providers:
            raise HTTPException(status_code=503, detail="No providers available")
        
        # Try providers in order
        last_error = None
        for provider_name in available_providers:
            try:
                provider = self.providers[provider_name]
                
                # Check rate limits
                if not await self.rate_limiter.check_rate_limit(user_id, provider_name):
                    logger.warning(f"Rate limit exceeded for {provider_name}")
                    continue
                
                # Get credentials
                credentials = await self.account_manager.get_credentials(provider_name)
                if not credentials:
                    logger.warning(f"No credentials available for {provider_name}")
                    continue
                
                # Make request
                response = await provider.chat_completion(request, credentials)
                
                # Update rate limiter
                await self.rate_limiter.record_request(user_id, provider_name)
                
                # Update provider scores
                self.router.update_provider_score(provider_name, success=True)
                
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
                logger.warning(f"Provider {provider_name} failed: {e}")
                self.router.update_provider_score(provider_name, success=False)
                last_error = e
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error with {provider_name}: {e}")
                last_error = e
                continue
        
        # All providers failed
        error_msg = f"All providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)
```

## 📦 Fix 5: Dependency Management

### Updated requirements.txt
```txt
# Core dependencies
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
httpx==0.25.2
aiohttp==3.9.1
asyncio-throttle==1.0.2

# Database and storage
sqlalchemy==2.0.23
alembic==1.13.1
redis==5.0.1

# Security and encryption
cryptography==41.0.8
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Configuration and environment
python-dotenv==1.0.0
pyyaml==6.0.1
click==8.1.7

# Monitoring and logging
prometheus-client==0.19.0
structlog==23.2.0
rich==13.7.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0
httpx-mock==0.7.0

# Development tools
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1

# Optional: Web UI dependencies
streamlit==1.28.2
plotly==5.17.0
pandas==2.1.4

# Auto-updater dependencies
beautifulsoup4>=4.12.0
playwright>=1.40.0
lxml>=4.9.0

# Optional: Enhanced research features (install separately if needed)
# torch>=2.0.0
# numpy>=1.24.0
```

### requirements-ml.txt (Optional ML features)
```txt
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
transformers>=4.30.0
```

## 🚀 Quick Setup Script

### setup_fixes.py
```python
#!/usr/bin/env python3
"""
Quick setup script to apply critical fixes
"""

import os
import shutil
import subprocess
import sys

def apply_fixes():
    print("🔧 Applying critical fixes to OpenHands...")
    
    # 1. Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""# Security
ADMIN_TOKEN=change-this-secure-token
ALLOWED_ORIGINS=http://localhost:3000

# Database
DATABASE_URL=sqlite:///./app.db
REDIS_URL=redis://localhost:6379

# Encryption
ENCRYPTION_KEY=change-this-32-byte-encryption-key
""")
        print("✅ Created .env file - PLEASE UPDATE THE TOKENS!")
    
    # 2. Create pytest.ini
    print("📝 Creating pytest.ini...")
    with open('pytest.ini', 'w') as f:
        f.write("""[tool:pytest]
asyncio_default_fixture_loop_scope = function
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
""")
    
    # 3. Remove credentials.json from git tracking
    if os.path.exists('credentials.json'):
        print("🔒 Moving credentials.json to credentials.json.backup...")
        shutil.move('credentials.json', 'credentials.json.backup')
        
    # 4. Create .gitignore entries
    gitignore_entries = [
        "# Security",
        ".env",
        "credentials.json",
        "*.key",
        "*.pem",
        "",
        "# Database",
        "*.db",
        "*.sqlite",
        "",
        "# Logs",
        "*.log",
        "logs/",
        "",
        "# Cache",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        "",
        "# ML Models",
        "models/",
        "checkpoints/",
    ]
    
    with open('.gitignore', 'a') as f:
        f.write('\n'.join(gitignore_entries))
    
    print("✅ Applied critical fixes!")
    print("\n🚨 IMPORTANT: Please update the tokens in .env file before running!")
    print("🚨 IMPORTANT: Install dependencies with: pip install -r requirements.txt")
    print("🚨 OPTIONAL: For ML features, install: pip install -r requirements-ml.txt")

if __name__ == "__main__":
    apply_fixes()
```

## 🏃‍♂️ Quick Start After Fixes

```bash
# 1. Apply fixes
python setup_fixes.py

# 2. Update .env file with your tokens
nano .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: Install ML features
pip install -r requirements-ml.txt

# 5. Run tests
python -m pytest

# 6. Start the server
python main.py --port 8000
```

These fixes address the critical blocking issues and make the application functional while maintaining security best practices.