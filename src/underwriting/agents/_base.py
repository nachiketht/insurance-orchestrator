"""One model call, then a Pydantic check, then a single repair call."""

import json

from pydantic import BaseModel, ValidationError

from underwriting.errors import AgentOutputError
from underwriting.llm import LLMClient


def call_structured(
    llm: LLMClient,
    *,
    agent: str,
    system: str,
    user: str,
    model: type[BaseModel],
    max_tokens: int,
) -> BaseModel:
    first = llm.complete(
        agent=agent, system=system, user=user, schema=model, max_tokens=max_tokens
    )
    try:
        return model.model_validate(first.data)
    except ValidationError as exc:
        repair = (
            f"{user}\n\nPrevious JSON:\n{json.dumps(first.data)}\n\n"
            f"Validation error:\n{exc}\nReturn corrected JSON only."
        )
        second = llm.complete(
            agent=agent, system=system, user=repair, schema=model, max_tokens=max_tokens
        )
        try:
            return model.model_validate(second.data)
        except ValidationError as again:
            raise AgentOutputError(str(again)) from again
