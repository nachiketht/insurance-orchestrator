"""Stand-ins for the model client. Tests and the offline pipeline use these."""

import time

from pydantic import BaseModel

from underwriting.errors import LLMTimeoutError
from underwriting.llm import LLMResponse


class ScriptedClient:
    """Returns canned JSON. A list of dicts is handed out one call at a time."""

    def __init__(self, responses: dict[str, dict | list[dict]]) -> None:
        self._responses = {
            agent: (list(payloads) if isinstance(payloads, list) else [payloads])
            for agent, payloads in responses.items()
        }
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []

    def complete(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> LLMResponse:
        del system, schema, max_tokens
        self.calls.append(agent)
        self.prompts.append((agent, user))
        queue = self._responses[agent]
        if not queue:
            raise AssertionError(f"no scripted response left for {agent}")
        return LLMResponse(
            data=queue.pop(0),
            input_tokens=120,
            output_tokens=40,
            model="scripted",
        )


class FaultInjectingLLM:
    """Raise on the first N calls to one agent, then pass through."""

    def __init__(
        self,
        inner,
        *,
        agent: str,
        failures: int,
        error: type[Exception] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.inner = inner
        self.agent = agent
        self.failures = failures
        self.error = error
        self.delay = delay
        self.seen = 0

    def complete(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> LLMResponse:
        if agent == self.agent and self.seen < self.failures:
            self.seen += 1
            if self.delay:
                time.sleep(self.delay)
            kind = self.error or LLMTimeoutError
            raise kind(f"injected {kind.__name__}")
        return self.inner.complete(
            agent=agent,
            system=system,
            user=user,
            schema=schema,
            max_tokens=max_tokens,
        )
