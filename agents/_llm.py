"""
FinSight AI — Shared LLM Helper
Structured Gemini calls with retry, caching, and schema support.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from cache.disk_cache import TTL_MEDIUM, cache_get, cache_set

logger = logging.getLogger(__name__)
settings = get_settings()


def _configure() -> None:
    genai.configure(api_key=settings.GEMINI_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_gemini_structured(
    system_prompt: str,
    user_prompt: str,
    response_schema: Any,
) -> dict[str, Any]:
    """
    Call Gemini Flash with structured JSON output.
    Retries up to 3 times with exponential backoff.
    """
    _configure()
    model = genai.GenerativeModel(
        settings.GEMINI_MODEL,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1,
            max_output_tokens=2048,
        ),
    )
    response = model.generate_content(user_prompt)
    return json.loads(response.text)


def get_cached_or_call(
    system_prompt: str,
    user_prompt: str,
    response_schema: Any,
    ttl: int = TTL_MEDIUM,
) -> dict[str, Any]:
    """
    Check cache before calling Gemini. Cache the result on success.
    """
    combined = f"{system_prompt}||{user_prompt}"
    cache_key = hashlib.sha256(combined.encode()).hexdigest()

    cached = cache_get(cache_key)
    if cached is not None:
        logger.debug("LLM cache hit for key %s", cache_key[:8])
        return cached

    result = call_gemini_structured(system_prompt, user_prompt, response_schema)
    cache_set(cache_key, result, ttl=ttl)
    return result
