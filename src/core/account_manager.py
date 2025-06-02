"""
Account and credentials management system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from cryptography.fernet import Fernet
import json
import os

from ..models import AccountCredentials


logger = logging.getLogger(__name__)


class AccountManager:
    """Manages API credentials and accounts for different providers."""
    
    def __init__(self, encryption_key: Optional[str] = None, storage_path: str = "credentials.json"):
        self.storage_path = storage_path
        self.credentials: Dict[str, List[AccountCredentials]] = {}
        self.usage_tracking: Dict[str, Dict[str, int]] = {}
        
        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Generate a new key if none provided
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logger.warning(f"Generated new encryption key: {key.decode()}")
        
        # Load existing credentials
        asyncio.create_task(self.load_credentials())
    
    async def add_credentials(
        self,
        provider: str,
        account_id: str,
        api_key: str,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> AccountCredentials:
        """Add new credentials for a provider."""
        
        credentials = AccountCredentials(
            provider=provider,
            account_id=account_id,
            api_key=api_key,
            additional_headers=additional_headers or {},
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        if provider not in self.credentials:
            self.credentials[provider] = []
        
        self.credentials[provider].append(credentials)
        
        # Initialize usage tracking
        if provider not in self.usage_tracking:
            self.usage_tracking[provider] = {}
        self.usage_tracking[provider][account_id] = 0
        
        await self.save_credentials()
        logger.info(f"Added credentials for {provider}:{account_id}")
        
        return credentials
    
    async def get_credentials(self, provider: str) -> Optional[AccountCredentials]:
        """Get active credentials for a provider using round-robin selection."""
        
        if provider not in self.credentials:
            return None
        
        active_creds = [
            cred for cred in self.credentials[provider] 
            if cred.is_active and not self._is_rate_limited(cred)
        ]
        
        if not active_creds:
            return None
        
        # Simple round-robin selection based on usage count
        selected_cred = min(active_creds, key=lambda c: self.usage_tracking[provider].get(c.account_id, 0))
        
        return selected_cred
    
    async def update_usage(self, credentials: AccountCredentials):
        """Update usage statistics for credentials."""
        
        provider = credentials.provider
        account_id = credentials.account_id
        
        if provider in self.usage_tracking and account_id in self.usage_tracking[provider]:
            self.usage_tracking[provider][account_id] += 1
        
        # Update last used timestamp
        credentials.last_used = datetime.utcnow()
        credentials.usage_count += 1
        
        await self.save_credentials()
    
    async def mark_credentials_invalid(self, provider: str, account_id: Optional[str] = None):
        """Mark credentials as invalid (e.g., due to auth failure)."""
        
        if provider not in self.credentials:
            return
        
        for cred in self.credentials[provider]:
            if account_id is None or cred.account_id == account_id:
                cred.is_active = False
                logger.warning(f"Marked credentials as invalid: {provider}:{cred.account_id}")
        
        await self.save_credentials()
    
    async def set_rate_limit_reset(self, provider: str, account_id: str, reset_time: datetime):
        """Set rate limit reset time for credentials."""
        
        if provider not in self.credentials:
            return
        
        for cred in self.credentials[provider]:
            if cred.account_id == account_id:
                cred.rate_limit_reset = reset_time
                break
        
        await self.save_credentials()
    
    def _is_rate_limited(self, credentials: AccountCredentials) -> bool:
        """Check if credentials are currently rate limited."""
        
        if not credentials.rate_limit_reset:
            return False
        
        return datetime.utcnow() < credentials.rate_limit_reset
    
    async def list_credentials(self) -> Dict[str, List[Dict[str, any]]]:
        """List all credentials (without sensitive data)."""
        
        result = {}
        for provider, creds in self.credentials.items():
            result[provider] = []
            for cred in creds:
                result[provider].append({
                    "account_id": cred.account_id,
                    "is_active": cred.is_active,
                    "created_at": cred.created_at.isoformat(),
                    "last_used": cred.last_used.isoformat() if cred.last_used else None,
                    "usage_count": cred.usage_count,
                    "rate_limit_reset": cred.rate_limit_reset.isoformat() if cred.rate_limit_reset else None
                })
        
        return result
    
    async def remove_credentials(self, provider: str, account_id: str) -> bool:
        """Remove credentials for a specific account."""
        
        if provider not in self.credentials:
            return False
        
        original_count = len(self.credentials[provider])
        self.credentials[provider] = [
            cred for cred in self.credentials[provider] 
            if cred.account_id != account_id
        ]
        
        removed = len(self.credentials[provider]) < original_count
        
        if removed:
            # Clean up usage tracking
            if provider in self.usage_tracking and account_id in self.usage_tracking[provider]:
                del self.usage_tracking[provider][account_id]
            
            await self.save_credentials()
            logger.info(f"Removed credentials for {provider}:{account_id}")
        
        return removed
    
    async def rotate_credentials(self, provider: str):
        """Rotate to next available credentials for a provider."""
        
        if provider not in self.credentials:
            return
        
        active_creds = [cred for cred in self.credentials[provider] if cred.is_active]
        if len(active_creds) <= 1:
            return
        
        # Reset usage counts to force rotation
        for cred in active_creds:
            self.usage_tracking[provider][cred.account_id] = 0
        
        logger.info(f"Rotated credentials for {provider}")
    
    async def save_credentials(self):
        """Save credentials to encrypted storage."""
        
        try:
            # Prepare data for serialization
            data = {
                "credentials": {},
                "usage_tracking": self.usage_tracking
            }
            
            for provider, creds in self.credentials.items():
                data["credentials"][provider] = []
                for cred in creds:
                    cred_data = {
                        "account_id": cred.account_id,
                        "api_key": cred.api_key,
                        "additional_headers": cred.additional_headers,
                        "is_active": cred.is_active,
                        "created_at": cred.created_at.isoformat(),
                        "last_used": cred.last_used.isoformat() if cred.last_used else None,
                        "usage_count": cred.usage_count,
                        "rate_limit_reset": cred.rate_limit_reset.isoformat() if cred.rate_limit_reset else None
                    }
                    data["credentials"][provider].append(cred_data)
            
            # Encrypt and save
            json_data = json.dumps(data)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            
            with open(self.storage_path, "wb") as f:
                f.write(encrypted_data)
            
            logger.debug("Credentials saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    async def load_credentials(self):
        """Load credentials from encrypted storage."""
        
        if not os.path.exists(self.storage_path):
            logger.info("No existing credentials file found")
            return
        
        try:
            with open(self.storage_path, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt and parse
            json_data = self.cipher.decrypt(encrypted_data).decode()
            data = json.loads(json_data)
            
            # Load credentials
            self.credentials = {}
            for provider, creds_data in data.get("credentials", {}).items():
                self.credentials[provider] = []
                for cred_data in creds_data:
                    cred = AccountCredentials(
                        provider=provider,
                        account_id=cred_data["account_id"],
                        api_key=cred_data["api_key"],
                        additional_headers=cred_data.get("additional_headers", {}),
                        is_active=cred_data.get("is_active", True),
                        created_at=datetime.fromisoformat(cred_data["created_at"]),
                        last_used=datetime.fromisoformat(cred_data["last_used"]) if cred_data.get("last_used") else None,
                        usage_count=cred_data.get("usage_count", 0),
                        rate_limit_reset=datetime.fromisoformat(cred_data["rate_limit_reset"]) if cred_data.get("rate_limit_reset") else None
                    )
                    self.credentials[provider].append(cred)
            
            # Load usage tracking
            self.usage_tracking = data.get("usage_tracking", {})
            
            logger.info(f"Loaded credentials for {len(self.credentials)} providers")
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            self.credentials = {}
            self.usage_tracking = {}
    
    async def get_usage_stats(self) -> Dict[str, Dict[str, any]]:
        """Get usage statistics for all providers."""
        
        stats = {}
        for provider, accounts in self.usage_tracking.items():
            total_usage = sum(accounts.values())
            active_accounts = len([
                cred for cred in self.credentials.get(provider, [])
                if cred.is_active
            ])
            
            stats[provider] = {
                "total_usage": total_usage,
                "active_accounts": active_accounts,
                "account_usage": accounts
            }
        
        return stats