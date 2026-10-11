"""One record per agent step: input, output, duration, and token usage."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Iterator

from underwriting.llm import LLMClient, LLMResponse

_current: ContextVar["Span | None"] = ContextVar("current_span", default=None)


@dataclass
class Span:
    step: int
    agent: str
    timestamp: str
    input: dict
    input_source: list[str]
    output: dict | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    status: str = "ok"
    error: str | None = None
    calls: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "agent": self.agent,
            "timestamp": self.timestamp,
            "input": self.input,
            "input_source": self.input_source,
            "output": self.output,
            "duration_ms": round(self.duration_ms, 1),
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "error": self.error,
            "model_calls": self.calls,
        }


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def step(
        self,
        step: int,
        agent: str,
        input_source: list[str],
        received: dict,
    ) -> Iterator[Span]:
        span = Span(
            step=step,
            agent=agent,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input=received,
            input_source=input_source,
        )
        token = _current.set(span)
        started = perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = (perf_counter() - started) * 1000
            span.tokens_used = sum(
                call["input_tokens"] + call["output_tokens"] for call in span.calls
            )
            span.cost_usd = 0.0
            _current.reset(token)
            self.spans.append(span)

    def dump(self, case_id: str, final_case: dict) -> dict:
        return {
            "case_id": case_id,
            "cost_usd": 0.0,
            "spans": [span.as_dict() for span in self.spans],
            "final_case": final_case,
        }


class TracedLLM:
    """Records each model call, including a repair, on the open agent step."""

    def __init__(self, inner: LLMClient, tracer: Tracer) -> None:
        del tracer
        self.inner = inner

    def complete(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type,
        max_tokens: int,
    ) -> LLMResponse:
        span = _current.get()
        try:
            response = self.inner.complete(
                agent=agent,
                system=system,
                user=user,
                schema=schema,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if span is not None:
                span.calls.append(
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "status": "error",
                        "error": type(exc).__name__,
                    }
                )
            raise
        if span is not None:
            span.calls.append(
                {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "status": "ok",
                }
            )
        return response
