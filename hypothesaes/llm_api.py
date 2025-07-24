"""LLM API utilities for HypotheSAEs."""

import os
import time
import openai
from .rate_limiter import get_rate_limiter, estimate_tokens

"""
These model IDs point to the latest versions of the models as of 2025-05-04.
We point to a specific version for reproducibility, but feel free to update them as necessary.
Note that o-series models (o1, o1-mini, o3-mini) are also supported by get_completion().
We don't point these models to a specific version, so passing in these model names will use the latest version.

2025-05-04:
- Removed gpt-4 (deprecated by gpt-4o, will be removed from API soon)
- Added gpt-4.1 models (not used by HypotheSAEs paper, but potentially of interest)

2025-03-12:
- First version of this file: supports gpt-4o, gpt-4o-mini, gpt-4
"""
model_abbrev_to_id = {
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',

    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
}

def get_client():
    """Get the OpenAI client, initializing it if necessary."""
    api_key = os.environ.get('OPENAI_KEY_SAE')
    if api_key is None or '...' in api_key:
        raise ValueError("Please set the OPENAI_KEY_SAE environment variable before using functions which require the OpenAI API.")
    
    return openai.OpenAI(api_key=api_key)

def get_completion(
    prompt: str,
    model: str = "gpt-4o",
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    use_rate_limiter: bool = True,
    **kwargs
) -> str:
    """
    Get completion from OpenAI API with retry logic, timeout, and rate limiting.
    
    Args:
        prompt: The prompt to send
        model: Model to use
        max_retries: Maximum number of retries on rate limit
        backoff_factor: Factor to multiply backoff time by after each retry
        timeout: Timeout for the request
        use_rate_limiter: Whether to use rate limiting (default: True)
        **kwargs: Additional arguments to pass to the OpenAI API; max_tokens, temperature, etc.
    Returns:
        Generated completion text
    
    Raises:
        Exception: If all retries fail
    """
    client = get_client()
    model_id = model_abbrev_to_id.get(model, model)
    
    # Estimate tokens for rate limiting
    if use_rate_limiter:
        estimated_tokens = estimate_tokens(prompt, model_id)
        max_completion_tokens = kwargs.get('max_tokens', 1000)
        total_tokens = estimated_tokens + max_completion_tokens
        
        rate_limiter = get_rate_limiter()
        rate_limiter.wait_for_capacity(total_tokens)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
                **kwargs
            )
            return response.choices[0].message.content
            
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            wait_time = timeout * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)