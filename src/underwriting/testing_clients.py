"""Stand-ins for the model client. Tests and the offline pipeline use these."""

from pydantic import BaseModel

from underwriting.llm import LLMResponse


class ScriptedClient:
    """Returns canned JSON. A list of dicts is handed out one call at a time."""

    def __init__(self, responses: dict[str, dict | list[dict]]) -> None:
        self._responses = {
            agent: (list(payloads) if isinstance(payloads, list) else [payloads])
            for agent, payloads in responses.items()
        }
        self.calls: list[str] = []

    def complete(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> LLMResponse:
        del system, user, schema, max_tokens
        self.calls.append(agent)
        queue = self._responses[agent]
        if not queue:
            raise AssertionError(f"no scripted response left for {agent}")
        return LLMResponse(
            data=queue.pop(0),
            input_tokens=120,
            output_tokens=40,
            model="scripted",
        )
