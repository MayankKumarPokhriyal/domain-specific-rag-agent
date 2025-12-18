"""Lightweight client for Ollama HTTP API."""
from typing import Dict, Optional

import requests

from utils.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Simple wrapper around Ollama's local HTTP API."""

    def __init__(
        self,
        api_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        payload: Dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        logger.info("Calling Ollama model %s", self.model)
        response = requests.post(self.api_url, json=payload, timeout=60)
        response.raise_for_status()
        data: Dict[str, object] = response.json()
        text = data.get("response") or data.get("text")
        if not isinstance(text, str):
            raise ValueError("Unexpected Ollama response payload")
        return text
