"""Retry a failed step, then stop. The pipeline decides what happens after that."""

import logging
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from underwriting.errors import RetryableError
from underwriting.tracing import Tracer

log = logging.getLogger("underwriting")


class StepFailed(Exception):
    """A step will not be tried again."""

    def __init__(self, agent: str, attempts: int, cause: Exception) -> None:
        self.agent = agent
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"{agent} failed after {attempts} attempts: {cause}")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def delay(self, attempt: int) -> float:
        """Wait after this failed attempt. At least half the cap, at most the cap."""
        cap = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return cap / 2 + self.rng.uniform(0, cap / 2)


def run_step(
    call: Callable,
    *,
    policy: RetryPolicy,
    tracer: Tracer | None,
    step: int,
    agent: str,
    source: list[str],
    received: dict,
):
    """Call once per attempt. Retry only RetryableError. Anything else stops immediately."""
    caught: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            with _span(tracer, step, agent, source, received) as span:
                span.attempt = attempt
                result = call()
                span.output = result.model_dump(mode="json")
            _log(span)
            return result
        except RetryableError as exc:
            caught = exc
            closed = _last(tracer)
            if attempt < policy.max_attempts:
                wait = policy.delay(attempt)
                if closed is not None:
                    closed.will_retry = True
                    closed.backoff_ms = round(wait * 1000, 1)
                    _log(closed)
                policy.sleep(wait)
                continue
            if closed is not None:
                _log(closed)
            raise StepFailed(agent, attempt, exc) from exc
        except Exception as exc:
            closed = _last(tracer)
            if closed is not None:
                _log(closed)
            raise StepFailed(agent, attempt, exc) from exc
    raise StepFailed(agent, policy.max_attempts, caught or RuntimeError("no attempt ran"))


def _last(tracer: Tracer | None):
    if tracer is None or not tracer.spans:
        return None
    return tracer.spans[-1]


def _log(span) -> None:
    log.info(
        "agent=%s attempt=%s status=%s err=%s dur_ms=%.0f will_retry=%s backoff_ms=%s",
        span.agent,
        span.attempt,
        span.status,
        span.error,
        span.duration_ms,
        span.will_retry,
        span.backoff_ms,
    )


class _Note:
    def __init__(self) -> None:
        self.output = None
        self.attempt = 1
        self.agent = ""
        self.status = "ok"
        self.error = None
        self.duration_ms = 0.0
        self.will_retry = False
        self.backoff_ms = 0.0


@contextmanager
def _span(
    tracer: Tracer | None,
    step: int,
    agent: str,
    source: list[str],
    received: dict,
) -> Iterator:
    if tracer is None:
        yield _Note()
        return
    with tracer.step(step, agent, source, received) as span:
        yield span
