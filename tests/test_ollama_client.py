import json

import httpx

from underwriting.errors import (
    AgentOutputError,
    LLMRequestError,
    LLMTimeoutError,
    LLMTransientError,
    ModelNotFoundError,
)
from underwriting.llm import OllamaClient
from underwriting.models import IntakeResult


def _client(handler) -> OllamaClient:
    return OllamaClient(
        host="http://ollama.test",
        model="qwen3:8b",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _chat_response(**extra) -> httpx.Response:
    body = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": json.dumps({"ok": True})},
        "prompt_eval_count": 11,
        "eval_count": 3,
        "done_reason": "stop",
        "load_duration": 2_000_000_000,
    }
    body.update(extra)
    return httpx.Response(200, json=body)


def test_complete_sends_schema_and_reads_token_counts() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _chat_response()

    result = _client(handler).complete(
        agent="intake",
        system="return json",
        user="{}",
        schema=IntakeResult,
        max_tokens=800,
    )

    body = seen["body"]
    assert body["think"] is False
    assert body["stream"] is False
    assert body["format"]["title"] == "IntakeResult"
    assert body["options"]["temperature"] == 0
    assert body["options"]["seed"] == 42
    assert body["options"]["num_predict"] == 800
    assert result.data == {"ok": True}
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.prompt_cached is False


def test_missing_prompt_eval_count_counts_as_cached() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(_chat_response().content)
        del body["prompt_eval_count"]
        return httpx.Response(200, json=body)

    result = _client(handler).complete(
        agent="intake", system="s", user="u", schema=IntakeResult, max_tokens=50
    )
    assert result.input_tokens == 0
    assert result.prompt_cached is True


def test_timeout_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    try:
        _client(handler).complete(
            agent="intake", system="s", user="u", schema=IntakeResult, max_tokens=50
        )
    except LLMTimeoutError:
        return
    raise AssertionError("expected LLMTimeoutError")


def test_connection_refused_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    try:
        _client(handler).preflight()
    except LLMTransientError as exc:
        assert "not reachable" in str(exc)
        return
    raise AssertionError("expected LLMTransientError")


def test_server_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    try:
        _client(handler).complete(
            agent="intake", system="s", user="u", schema=IntakeResult, max_tokens=50
        )
    except LLMTransientError:
        return
    raise AssertionError("expected LLMTransientError")


def test_missing_model_is_not_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    try:
        _client(handler).complete(
            agent="intake", system="s", user="u", schema=IntakeResult, max_tokens=50
        )
    except ModelNotFoundError:
        return
    raise AssertionError("expected ModelNotFoundError")


def test_bad_request_is_not_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad format")

    try:
        _client(handler).complete(
            agent="intake", system="s", user="u", schema=IntakeResult, max_tokens=50
        )
    except LLMRequestError:
        return
    raise AssertionError("expected LLMRequestError")


def test_truncated_json_is_an_output_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(done_reason="length")

    try:
        _client(handler).complete(
            agent="intake", system="s", user="u", schema=IntakeResult, max_tokens=50
        )
    except AgentOutputError:
        return
    raise AssertionError("expected AgentOutputError")


def test_preflight_accepts_an_installed_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    _client(handler).preflight()


def test_preflight_rejects_a_model_that_is_not_pulled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})

    try:
        _client(handler).preflight()
    except ModelNotFoundError as exc:
        assert "ollama pull qwen3:8b" in str(exc)
        return
    raise AssertionError("expected ModelNotFoundError")


def test_warm_up_reports_load_seconds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["keep_alive"] == "30m"
        assert body["options"]["num_predict"] == 1
        return _chat_response()

    assert _client(handler).warm_up() == 2.0
