"""Hosted-model prices for the same token counts a local run produced.

Local Ollama charges no API fee. These rates are only a comparison.
Claude Haiku 5.5, prompts up to 100,000 tokens: $0.10 per million input
tokens and $0.50 per million output tokens.
Source: https://platform.claude.com/docs/en/about-claude/pricing
List prices dated 2026-10-07, checked 2026-10-11.
"""

HOSTED_MODEL = "claude-haiku-5.5"
INPUT_USD_PER_MTOK = 0.10
OUTPUT_USD_PER_MTOK = 0.50
SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
CHECKED = "2026-10-11"

REALTIME_SECONDS = 2.0
BATCH_CASES = 10_000
BATCH_WINDOW_SECONDS = 8 * 60 * 60


def hosted_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_MTOK + output_tokens * OUTPUT_USD_PER_MTOK
    ) / 1_000_000
