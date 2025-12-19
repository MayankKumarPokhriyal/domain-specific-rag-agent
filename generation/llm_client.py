"""Lightweight client for interacting with a local Ollama model."""

from typing import Dict

import requests

from utils.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """
    Minimal HTTP client for Ollama's generate endpoint.

    Assumes Ollama is running locally.
    """

    def __init__(
        self,
        api_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 60,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Generate text from the Ollama model using the given prompt.
        """
        payload: Dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }

        logger.info("Sending prompt to Ollama model '%s'", self.model)

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError("Failed to communicate with Ollama") from exc

        data = response.json()

        # Ollama typically returns the text under 'response'
        text = data.get("response") or data.get("text")
        if not isinstance(text, str) or not text.strip():
            logger.error("Unexpected Ollama response payload: %s", data)
            raise ValueError("Ollama returned an empty or invalid response")

        return text.strip()