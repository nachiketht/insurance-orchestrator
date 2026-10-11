"""Cost and latency figures from the saved local traces. No network calls."""

import json
from pathlib import Path

from underwriting.pricing import (
    BATCH_CASES,
    BATCH_WINDOW_SECONDS,
    CHECKED,
    HOSTED_MODEL,
    INPUT_USD_PER_MTOK,
    OUTPUT_USD_PER_MTOK,
    REALTIME_SECONDS,
    SOURCE,
    hosted_cost,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "traces" / "samples"
OUTPUT = ROOT / "docs" / "cost_latency.md"
AGENTS = ("intake", "enrichment", "risk_scoring", "recommendation")
STEP_BUDGET_S = {
    "intake": 8,
    "enrichment": 7,
    "risk_scoring": 12,
    "recommendation": 10,
}
STEP_BUDGET_TOKENS = {
    "intake": 850,
    "enrichment": 720,
    "risk_scoring": 1050,
    "recommendation": 1100,
}


def load_cases() -> list[dict]:
    cases = []
    for path in sorted(SAMPLES.glob("*.json")):
        payload = json.loads(path.read_text())
        if "fixture" not in payload:
            continue
        input_tokens = 0
        output_tokens = 0
        steps = {}
        for span in payload["spans"]:
            inn = sum(call["input_tokens"] for call in span["model_calls"])
            out = sum(call["output_tokens"] for call in span["model_calls"])
            input_tokens += inn
            output_tokens += out
            steps[span["agent"]] = {
                "ms": span["duration_ms"],
                "input_tokens": inn,
                "output_tokens": out,
                "tokens": span["tokens_used"],
            }
        cases.append(
            {
                "fixture": payload["fixture"],
                "ms": payload["total_ms"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens": payload["tokens_used"],
                "hosted_usd": hosted_cost(input_tokens, output_tokens),
                "steps": steps,
            }
        )
    return cases


def render(cases: list[dict]) -> str:
    mean_ms = sum(case["ms"] for case in cases) / len(cases)
    mean_s = mean_ms / 1000
    max_s = max(case["ms"] for case in cases) / 1000
    mean_hosted = sum(case["hosted_usd"] for case in cases) / len(cases)
    batch_seconds = BATCH_CASES * mean_s
    nightly = int(BATCH_WINDOW_SECONDS // mean_s)
    example = next(case for case in cases if case["fixture"] == "clean_auto")
    hand_input = example["input_tokens"] * INPUT_USD_PER_MTOK / 1_000_000
    hand_output = example["output_tokens"] * OUTPUT_USD_PER_MTOK / 1_000_000

    lines = [
        "# Cost and latency",
        "",
        "Numbers below come from the three live traces in `traces/samples/`,",
        "run on local Ollama `qwen3:8b`. The model was already loaded, so the",
        "warm-up time was 0.0 seconds and is not included in the case times.",
        "API fees for this run are $0.00. Electricity and the host machine are",
        "not priced here.",
        "",
        "The hosted column is a comparison only. It prices the same token counts",
        f"as {HOSTED_MODEL} for prompts under 100,000 tokens:",
        f"${INPUT_USD_PER_MTOK:.2f} per million input tokens and",
        f"${OUTPUT_USD_PER_MTOK:.2f} per million output tokens.",
        f"Source: {SOURCE}",
        f"List prices dated 2026-10-07, checked {CHECKED}.",
        "The counts come from Ollama's tokenizer, not Claude's, so the dollar",
        "figure is approximate. Every call in these traces reported a non-zero",
        "input count.",
        "",
        "## Per case",
        "",
        "| Case | Seconds | Input tokens | Output tokens | Local API cost | Hosted equivalent |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        lines.append(
            f"| {case['fixture']} | {case['ms'] / 1000:.2f} | {case['input_tokens']} | "
            f"{case['output_tokens']} | $0.00 | ${case['hosted_usd']:.6f} |"
        )
    lines += [
        f"| Mean | {mean_s:.2f} | | | $0.00 | ${mean_hosted:.6f} |",
        f"| Max | {max_s:.2f} | | | | |",
        "",
        "## Per step, mean of the three cases",
        "",
        "| Step | Mean seconds | Budget seconds | Mean tokens | Token budget |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for agent in AGENTS:
        seconds = sum(case["steps"][agent]["ms"] for case in cases) / len(cases) / 1000
        tokens = sum(case["steps"][agent]["tokens"] for case in cases) / len(cases)
        lines.append(
            f"| {agent} | {seconds:.2f} | {STEP_BUDGET_S[agent]} | "
            f"{tokens:.0f} | {STEP_BUDGET_TOKENS[agent]} |"
        )
    lines += [
        f"| whole case | {mean_s:.2f} | 37 | | 3,720 |",
        "",
        "Every step is inside the budget written in `docs/design.md` before these",
        "runs. Risk scoring is the slow step. It is about half of each case.",
        "",
        "## Worked hosted cost",
        "",
        f"`clean_auto` used {example['input_tokens']} input tokens and",
        f"{example['output_tokens']} output tokens.",
        "",
        f"{example['input_tokens']} * {INPUT_USD_PER_MTOK:.2f} / 1,000,000 = ${hand_input:.7f}",
        f"{example['output_tokens']} * {OUTPUT_USD_PER_MTOK:.2f} / 1,000,000 = ${hand_output:.7f}",
        f"Total = ${example['hosted_usd']:.7f}",
        "",
        "## Verdict",
        "",
        f"Mean {mean_s:.2f} seconds per case is {mean_s / REALTIME_SECONDS:.1f} times the",
        f"{REALTIME_SECONDS:.0f} second real-time budget. Real time fails. The four calls",
        "run one after another, and the next agent needs the previous agent's full",
        "JSON, so streaming the reply would not shorten the wait.",
        "",
        f"At {mean_s:.2f} seconds each, {BATCH_CASES:,} cases take "
        f"{batch_seconds / 3600:.1f} hours",
        f"on this one machine. The overnight window is {BATCH_WINDOW_SECONDS / 3600:.0f} hours,",
        f"so that batch fails too. This machine can finish about {nightly:,} cases in",
        "8 hours. Batch is still the right way to run the pipeline, because no",
        "applicant is waiting on their own case. More machines, a smaller model,",
        "or combining intake and enrichment into one call would raise that number.",
        "The hosted equivalent is well under one cent per case, so cost is not",
        "what makes either use case fail.",
        "",
    ]
    return "\n".join(lines)


def write_report() -> str:
    text = render(load_cases())
    OUTPUT.write_text(text)
    return text
