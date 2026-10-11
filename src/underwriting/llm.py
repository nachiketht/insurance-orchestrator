"""Model client. Agents depend on LLMClient, not on Ollama directly."""

import json
import os
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import BaseModel

from underwriting.errors import (
    AgentOutputError,
    LLMRequestError,
    LLMTimeoutError,
    LLMTransientError,
    ModelNotFoundError,
)

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://localhost:11434"
SEED = 42


@dataclass
class LLMResponse:
    data: dict
    input_tokens: int
    output_tokens: int
    model: str
    prompt_cached: bool = False


class LLMClient(Protocol):
    def complete(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> LLMResponse:
        """Return structured data and the token counts for one call."""


class OllamaClient:
    """POST /api/chat. The schema is sent as format so the reply is JSON."""

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        http: httpx.Client | None = None,
        seed: int = SEED,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("UNDERWRITING_MODEL", DEFAULT_MODEL)
        self.host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.seed = seed
        self._owns_http = http is None
        self.http = http or httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    def close(self) -> None:
        if self._owns_http:
            self.http.close()

    def complete(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> LLMResponse:
        del agent  # the protocol carries it so fakes can answer per agent
        body = self._request(
            "POST",
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": schema.model_json_schema(),
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "seed": self.seed,
                    "num_predict": max_tokens,
                },
            },
        )
        if body.get("done_reason") == "length":
            raise AgentOutputError("truncated at num_predict")
        content = body.get("message", {}).get("content", "")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AgentOutputError("model did not return JSON") from exc
        if not isinstance(data, dict):
            raise AgentOutputError("model JSON was not an object")
        cached = "prompt_eval_count" not in body
        return LLMResponse(
            data=data,
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            model=body.get("model", self.model),
            prompt_cached=cached,
        )

    def preflight(self) -> None:
        """Fail fast when Ollama is down or the model was never pulled."""
        try:
            body = self._request("GET", "/api/tags")
        except LLMTimeoutError as exc:
            raise LLMTransientError(f"Ollama is not reachable at {self.host}") from exc
        names = [item.get("name", "") for item in body.get("models", [])]
        installed = any(name == self.model or name.startswith(self.model + ":") for name in names)
        if not installed:
            raise ModelNotFoundError(
                f"model {self.model} is not pulled; run: ollama pull {self.model}"
            )

    def warm_up(self) -> float:
        """Load the model and return how many seconds that took."""
        body = self._request(
            "POST",
            "/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 1},
            },
        )
        return body.get("load_duration", 0) / 1_000_000_000

    def _request(self, method: str, path: str, json_body: dict | None = None) -> dict:
        try:
            response = self.http.request(method, f"{self.host}{path}", json=json_body)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except httpx.ConnectError as exc:
            raise LLMTransientError(f"Ollama is not reachable at {self.host}") from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            detail = exc.response.text
            if code == 404:
                raise ModelNotFoundError(
                    f"model {self.model} is not pulled; run: ollama pull {self.model}"
                ) from exc
            if code >= 500:
                raise LLMTransientError(f"Ollama returned {code}: {detail}") from exc
            raise LLMRequestError(f"Ollama returned {code}: {detail}") from exc
        return response.json()
