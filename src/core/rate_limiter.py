"""
Rate limiting system for the LLM aggregator.
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter with per-user and global limits."""
    
    def __init__(
        self,
        global_requests_per_minute: int = 100,
        global_requests_per_hour: int = 1000,
        user_requests_per_minute: int = 10,
        user_requests_per_hour: int = 100,
        max_concurrent_requests: int = 50
    ):
        self.global_requests_per_minute = global_requests_per_minute
        self.global_requests_per_hour = global_requests_per_hour
        self.user_requests_per_minute = user_requests_per_minute
        self.user_requests_per_hour = user_requests_per_hour
        self.max_concurrent_requests = max_concurrent_requests
        
        # Track requests with sliding window
        self.global_requests_minute = deque()
        self.global_requests_hour = deque()
        self.user_requests_minute: Dict[str, deque] = defaultdict(deque)
        self.user_requests_hour: Dict[str, deque] = defaultdict(deque)
        
        # Semaphore for concurrent request limiting
        self.concurrent_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def acquire(self, user_id: Optional[str] = None) -> bool:
        """Acquire rate limit permission."""
        
        async with self.lock:
            current_time = time.time()
            
            # Clean old entries
            self._clean_old_entries(current_time)
            
            # Check global limits
            if not self._check_global_limits(current_time):
                raise RateLimitExceeded("Global rate limit exceeded")
            
            # Check user limits if user_id provided
            if user_id and not self._check_user_limits(user_id, current_time):
                raise RateLimitExceeded(f"User rate limit exceeded for {user_id}")
            
            # Record the request
            self._record_request(user_id, current_time)
        
        # Acquire concurrent request semaphore
        await self.concurrent_semaphore.acquire()
        
        return True
    
    def release(self, user_id: Optional[str] = None):
        """Release rate limit permission."""
        self.concurrent_semaphore.release()
    
    def _clean_old_entries(self, current_time: float):
        """Remove old entries from tracking queues."""
        
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        # Clean global queues
        while self.global_requests_minute and self.global_requests_minute[0] < minute_cutoff:
            self.global_requests_minute.popleft()
        
        while self.global_requests_hour and self.global_requests_hour[0] < hour_cutoff:
            self.global_requests_hour.popleft()
        
        # Clean user queues
        for user_id in list(self.user_requests_minute.keys()):
            user_minute_queue = self.user_requests_minute[user_id]
            while user_minute_queue and user_minute_queue[0] < minute_cutoff:
                user_minute_queue.popleft()
            
            # Remove empty queues
            if not user_minute_queue:
                del self.user_requests_minute[user_id]
        
        for user_id in list(self.user_requests_hour.keys()):
            user_hour_queue = self.user_requests_hour[user_id]
            while user_hour_queue and user_hour_queue[0] < hour_cutoff:
                user_hour_queue.popleft()
            
            # Remove empty queues
            if not user_hour_queue:
                del self.user_requests_hour[user_id]
    
    def _check_global_limits(self, current_time: float) -> bool:
        """Check if global rate limits allow the request."""
        
        # Check minute limit
        if len(self.global_requests_minute) >= self.global_requests_per_minute:
            return False
        
        # Check hour limit
        if len(self.global_requests_hour) >= self.global_requests_per_hour:
            return False
        
        return True
    
    def _check_user_limits(self, user_id: str, current_time: float) -> bool:
        """Check if user rate limits allow the request."""
        
        # Check minute limit
        user_minute_queue = self.user_requests_minute[user_id]
        if len(user_minute_queue) >= self.user_requests_per_minute:
            return False
        
        # Check hour limit
        user_hour_queue = self.user_requests_hour[user_id]
        if len(user_hour_queue) >= self.user_requests_per_hour:
            return False
        
        return True
    
    def _record_request(self, user_id: Optional[str], current_time: float):
        """Record a request in the tracking queues."""
        
        # Record global request
        self.global_requests_minute.append(current_time)
        self.global_requests_hour.append(current_time)
        
        # Record user request if user_id provided
        if user_id:
            self.user_requests_minute[user_id].append(current_time)
            self.user_requests_hour[user_id].append(current_time)
    
    def get_rate_limit_status(self, user_id: Optional[str] = None) -> Dict[str, any]:
        """Get current rate limit status."""
        
        current_time = time.time()
        
        # Clean old entries first
        self._clean_old_entries(current_time)
        
        status = {
            "global": {
                "requests_per_minute": {
                    "current": len(self.global_requests_minute),
                    "limit": self.global_requests_per_minute,
                    "remaining": self.global_requests_per_minute - len(self.global_requests_minute)
                },
                "requests_per_hour": {
                    "current": len(self.global_requests_hour),
                    "limit": self.global_requests_per_hour,
                    "remaining": self.global_requests_per_hour - len(self.global_requests_hour)
                },
                "concurrent_requests": {
                    "current": self.max_concurrent_requests - self.concurrent_semaphore._value,
                    "limit": self.max_concurrent_requests,
                    "remaining": self.concurrent_semaphore._value
                }
            }
        }
        
        if user_id:
            user_minute_count = len(self.user_requests_minute.get(user_id, []))
            user_hour_count = len(self.user_requests_hour.get(user_id, []))
            
            status["user"] = {
                "user_id": user_id,
                "requests_per_minute": {
                    "current": user_minute_count,
                    "limit": self.user_requests_per_minute,
                    "remaining": self.user_requests_per_minute - user_minute_count
                },
                "requests_per_hour": {
                    "current": user_hour_count,
                    "limit": self.user_requests_per_hour,
                    "remaining": self.user_requests_per_hour - user_hour_count
                }
            }
        
        return status
    
    def reset_user_limits(self, user_id: str):
        """Reset rate limits for a specific user."""
        
        if user_id in self.user_requests_minute:
            del self.user_requests_minute[user_id]
        
        if user_id in self.user_requests_hour:
            del self.user_requests_hour[user_id]
        
        logger.info(f"Reset rate limits for user: {user_id}")
    
    def get_user_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all users."""
        
        current_time = time.time()
        self._clean_old_entries(current_time)
        
        stats = {}
        
        # Get all unique user IDs
        all_users = set(self.user_requests_minute.keys()) | set(self.user_requests_hour.keys())
        
        for user_id in all_users:
            minute_count = len(self.user_requests_minute.get(user_id, []))
            hour_count = len(self.user_requests_hour.get(user_id, []))
            
            stats[user_id] = {
                "requests_last_minute": minute_count,
                "requests_last_hour": hour_count
            }
        
        return stats


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class ProviderRateLimiter:
    """Rate limiter specific to individual providers."""
    
    def __init__(self, provider_name: str, requests_per_minute: int, requests_per_hour: int):
        self.provider_name = provider_name
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        self.requests_minute = deque()
        self.requests_hour = deque()
        self.lock = asyncio.Lock()
    
    async def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding limits."""
        
        async with self.lock:
            current_time = time.time()
            self._clean_old_entries(current_time)
            
            # Check limits
            minute_ok = len(self.requests_minute) < self.requests_per_minute
            hour_ok = len(self.requests_hour) < self.requests_per_hour
            
            return minute_ok and hour_ok
    
    async def record_request(self):
        """Record a request."""
        
        async with self.lock:
            current_time = time.time()
            self.requests_minute.append(current_time)
            self.requests_hour.append(current_time)
    
    def _clean_old_entries(self, current_time: float):
        """Remove old entries."""
        
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        
        while self.requests_minute and self.requests_minute[0] < minute_cutoff:
            self.requests_minute.popleft()
        
        while self.requests_hour and self.requests_hour[0] < hour_cutoff:
            self.requests_hour.popleft()
    
    def get_status(self) -> Dict[str, any]:
        """Get current status."""
        
        current_time = time.time()
        self._clean_old_entries(current_time)
        
        return {
            "provider": self.provider_name,
            "requests_per_minute": {
                "current": len(self.requests_minute),
                "limit": self.requests_per_minute,
                "remaining": self.requests_per_minute - len(self.requests_minute)
            },
            "requests_per_hour": {
                "current": len(self.requests_hour),
                "limit": self.requests_per_hour,
                "remaining": self.requests_per_hour - len(self.requests_hour)
            }
        }