"""Run the sample applications and write a trace for each."""

import argparse
import json
import random
import sys
from pathlib import Path

from underwriting.fixtures import CASES
from underwriting.llm import OllamaClient
from underwriting.pipeline import run_case
from underwriting.retry import RetryPolicy
from underwriting.testing_clients import FaultInjectingLLM, ScriptedClient
from underwriting.tracing import Tracer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.offline:
        run_offline(args.out or ROOT / "traces" / "ci")
        return
    run_live(args.out or ROOT / "traces" / "samples")


def run_offline(out: Path) -> None:
    """Full pipeline with canned model replies. No Ollama."""
    out.mkdir(parents=True, exist_ok=True)
    client = _scripted()
    for name, raw in CASES.items():
        _write(out, name, raw, client, RetryPolicy())
    timeout = FaultInjectingLLM(
        _scripted_one(CASES["clean_auto"]),
        agent="risk_scoring",
        failures=99,
    )
    policy = RetryPolicy(sleep=lambda _seconds: None, rng=random.Random(0))
    _write(out, "timeout_case", CASES["clean_auto"], timeout, policy)


def run_live(out: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.show_trace import render

    out.mkdir(parents=True, exist_ok=True)
    client = OllamaClient()
    try:
        client.preflight()
        load_s = client.warm_up()
        print(f"model loaded in {load_s:.1f}s")
        for name, raw in CASES.items():
            path = _write(out, name, raw, client, RetryPolicy())
            payload = json.loads(path.read_text())
            decision = payload["final_case"]["recommendation"]["decision"]
            score = (payload["final_case"].get("risk") or {}).get("score")
            print(f"\n{name}: decision={decision} score={score}")
            print(render(payload))
            print(f"wrote {path}")
    finally:
        client.close()


def _write(out: Path, name: str, raw, client, policy: RetryPolicy) -> Path:
    tracer = Tracer()
    case = run_case(raw, llm=client, tracer=tracer, policy=policy)
    payload = tracer.dump(case.case_id, case.model_dump(mode="json"))
    payload["fixture"] = name
    payload["total_ms"] = round(sum(span["duration_ms"] for span in payload["spans"]), 1)
    payload["tokens_used"] = sum(span["tokens_used"] for span in payload["spans"])
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"{name}: status={case.status} decision={case.recommendation.decision}")
    return path


def _scripted() -> ScriptedClient:
    return ScriptedClient(
        {
            "intake": [
                _intake("auto", "1990-12-10", 250000),
                _intake("home", "1970-01-01", 600000),
                _intake("life", "1988-06-23", 500000),
            ],
            "enrichment": [_enrichment(), _enrichment(), _enrichment()],
            "risk_scoring": [_risk(10), _risk(30, 25, 20, 15), _risk(20, name="missing_data")],
            "recommendation": [_approve(), _approve(), _approve()],
        }
    )


def _scripted_one(raw) -> ScriptedClient:
    del raw
    return ScriptedClient(
        {
            "intake": _intake("auto", "1990-12-10", 250000),
            "enrichment": _enrichment(),
            "risk_scoring": _risk(10),
            "recommendation": _approve(),
        }
    )


def _intake(product: str, dob: str, coverage: int) -> dict:
    return {
        "applicant_id": "from-model",
        "applicant_name": "from-model",
        "date_of_birth": dob,
        "address": "from-model",
        "product": product,
        "coverage_amount": coverage,
        "stated_income": 80000,
        "claims_history_text": "none",
        "missing_fields": [],
    }


def _enrichment() -> dict:
    return {
        "credit_band": "a",
        "prior_claims_count": 0,
        "asset_age_years": 1,
        "flags": [],
        "notes": "Copied from the bureau.",
    }


def _risk(*points: int, name: str = "prior_claims") -> dict:
    if len(points) == 1:
        factors = [{"name": name, "points": points[0], "detail": "scripted"}]
    else:
        names = ["prior_claims", "credit", "coverage_income", "asset_age"]
        factors = [
            {"name": factor, "points": value, "detail": "scripted"}
            for factor, value in zip(names, points, strict=True)
        ]
    return {"factors": factors, "score": 0, "rationale": "Scripted."}


def _approve() -> dict:
    return {
        "decision": "approve",
        "reasons": ["Scripted."],
        "conditions": [],
        "source": "agent",
    }


if __name__ == "__main__":
    main()
