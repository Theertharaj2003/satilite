from __future__ import annotations

import json
from typing import Any


class OllamaClient:
    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.url = url.rstrip("/")
        self.model = model

    def available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.url}/api/tags", timeout=2)
            return r.ok
        except Exception:
            return False

    def generate(self, prompt: str) -> str:
        try:
            import requests
            r = requests.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return str(data.get("response", "")).strip()
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc
