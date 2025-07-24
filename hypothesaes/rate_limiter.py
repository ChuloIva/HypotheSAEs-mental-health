"""Rate limiting utilities for API calls."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import threading


@dataclass
class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    max_requests_per_minute: float = 500.0  # For completions
    max_tokens_per_minute: float = 30000.0  # For completions
    max_embedding_requests_per_minute: float = 3000.0  # For embeddings
    max_embedding_tokens_per_minute: float = 10000000.0  # For embeddings
    
    def __post_init__(self):
        self._lock = threading.Lock()
        self._available_request_capacity = self.max_requests_per_minute
        self._available_token_capacity = self.max_tokens_per_minute
        self._last_update_time = time.time()
    
    def _update_capacity(self):
        """Update available capacity based on time elapsed."""
        current_time = time.time()
        seconds_since_update = current_time - self._last_update_time
        
        self._available_request_capacity = min(
            self._available_request_capacity + 
            self.max_requests_per_minute * seconds_since_update / 60.0,
            self.max_requests_per_minute,
        )
        self._available_token_capacity = min(
            self._available_token_capacity + 
            self.max_tokens_per_minute * seconds_since_update / 60.0,
            self.max_tokens_per_minute,
        )
        self._last_update_time = current_time
    
    def wait_for_capacity(self, tokens_needed: int = 1) -> None:
        """Wait until sufficient capacity is available for the request."""
        while True:
            with self._lock:
                self._update_capacity()
                
                if (self._available_request_capacity >= 1 and 
                    self._available_token_capacity >= tokens_needed):
                    # Reserve capacity
                    self._available_request_capacity -= 1
                    self._available_token_capacity -= tokens_needed
                    return
            
            # Sleep briefly and try again
            time.sleep(0.1)


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter(
    max_requests_per_minute: float = 500.0,
    max_tokens_per_minute: float = 30000.0
) -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_rate_limiter
    
    with _limiter_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = RateLimiter(
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute
            )
    
    return _global_rate_limiter


# Global embedding rate limiter instance
_global_embedding_rate_limiter: Optional[RateLimiter] = None
_embedding_limiter_lock = threading.Lock()

def get_embedding_rate_limiter(
    max_embedding_requests_per_minute: float = 3000.0,
    max_embedding_tokens_per_minute: float = 10000000.0
) -> RateLimiter:
    """Get or create the global embedding rate limiter instance."""
    global _global_embedding_rate_limiter
    with _embedding_limiter_lock:
        if _global_embedding_rate_limiter is None:
            _global_embedding_rate_limiter = RateLimiter(
                max_requests_per_minute=max_embedding_requests_per_minute,
                max_tokens_per_minute=max_embedding_tokens_per_minute
            )
    return _global_embedding_rate_limiter


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for a text string."""
    try:
        import tiktoken
        
        # Use appropriate encoding based on model
        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "embedding" in model:
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except ImportError:
        # Fallback estimation if tiktoken not available
        return len(text.split()) * 1.3  # Rough approximation