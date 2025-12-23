"""
Robust client for interacting with a local Ollama model.

This implementation:
- Uses the correct /api/generate endpoint
- Handles large prompts safely
- Uses modern Ollama payload format
- Avoids premature timeouts
"""

from __future__ import annotations

from typing import Any, Dict

import requests

from utils.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """
    HTTP client for Ollama's /api/generate endpoint.

    Assumes Ollama is running locally.
    """

    def __init__(
        self,
        api_url: str,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 300,
    ) -> None:
        # Normalize API URL
        api_url = api_url.rstrip("/")
        if not api_url.endswith("/api/generate"):
            api_url = f"{api_url}/api/generate"

        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        logger.info(
            "Initialized OllamaClient | model=%s | endpoint=%s",
            self.model,
            self.api_url,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text from the Ollama model using the given prompt.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        logger.info("Sending prompt to Ollama (len=%d chars)", len(prompt))

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=(10, self.timeout),  # (connect timeout, read timeout)
            )
            response.raise_for_status()

        except requests.exceptions.ReadTimeout as exc:
            logger.error(
                "Ollama timed out after %s seconds (model may be cold)",
                self.timeout,
            )
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s. "
                "Try warming the model or reducing context size."
            ) from exc

        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError("Failed to communicate with Ollama") from exc

        data = response.json()

        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid Ollama response payload: %s", data)
            raise ValueError("Ollama returned an empty or invalid response")

        return text.strip()