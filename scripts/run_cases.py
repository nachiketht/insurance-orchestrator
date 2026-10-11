"""Run the three sample applications and write a trace for each."""

import json
import sys
from pathlib import Path

from underwriting.fixtures import CASES
from underwriting.llm import OllamaClient
from underwriting.pipeline import run_case
from underwriting.tracing import Tracer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.show_trace import render

    out = ROOT / "traces" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    client = OllamaClient()
    try:
        client.preflight()
        load_s = client.warm_up()
        print(f"model loaded in {load_s:.1f}s")
        for name, raw in CASES.items():
            tracer = Tracer()
            case = run_case(raw, llm=client, tracer=tracer)
            payload = tracer.dump(case.case_id, case.model_dump(mode="json"))
            payload["fixture"] = name
            payload["total_ms"] = round(sum(span["duration_ms"] for span in payload["spans"]), 1)
            payload["tokens_used"] = sum(span["tokens_used"] for span in payload["spans"])
            path = out / f"{name}.json"
            path.write_text(json.dumps(payload, indent=2))
            decision = case.recommendation.decision if case.recommendation else None
            score = case.risk.score if case.risk else None
            print(f"\n{name}: decision={decision} score={score}")
            print(render(payload))
            print(f"wrote {path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
