"""Print a trace as a short timeline."""

import json
import sys
from pathlib import Path


def render(payload: dict) -> str:
    final = payload.get("final_case") or {}
    recommendation = final.get("recommendation") or {}
    decision = recommendation.get("decision", "-")
    lines = [
        f"fixture={payload.get('fixture', '-')}  decision={decision}  "
        f"spans={len(payload['spans'])}  cost_usd={payload['cost_usd']:.2f}",
    ]
    for span in payload["spans"]:
        lines.append(
            f"[{span['step']}] {span['agent']}  {span['status']}  "
            f"{span['duration_ms']} ms  tokens_used={span['tokens_used']}  "
            f"cost_usd={span['cost_usd']:.2f}  model_calls={len(span['model_calls'])}"
        )
        lines.append(f"    input_source: {', '.join(span['input_source'])}")
        lines.append(f"    received: {_brief(span['input'])}")
        lines.append(f"    produced: {_brief(span['output'])}")
        if span["error"]:
            retry = ""
            if span.get("will_retry"):
                retry = f" will_retry=True backoff_ms={span.get('backoff_ms')}"
            lines.append(f"    error: {span['error']} attempt={span.get('attempt', 1)}{retry}")
    return "\n".join(lines)


def _brief(value: dict | None) -> str:
    if not value:
        return "-"
    preferred = (
        "product",
        "applicant_id",
        "coverage_amount",
        "credit_band",
        "prior_claims_count",
        "flags",
        "score",
        "decision",
    )
    parts = []
    for key in preferred:
        if key in value and value[key] not in (None, [], ""):
            parts.append(f"{key}={value[key]}")
    if "intake" in value:
        parts.append(f"intake.product={value['intake'].get('product')}")
    if "bureau" in value:
        parts.append(f"bureau.credit_band={value['bureau'].get('credit_band')}")
    if "enrichment" in value:
        band = value["enrichment"].get("credit_band")
        claims = value["enrichment"].get("prior_claims_count")
        parts.append(f"enrichment.credit_band={band}")
        parts.append(f"enrichment.prior_claims_count={claims}")
    if "risk" in value:
        parts.append(f"risk.score={value['risk'].get('score')}")
    return ", ".join(parts) if parts else ", ".join(list(value)[:4])


def main() -> None:
    path = Path(sys.argv[1])
    print(render(json.loads(path.read_text())))


if __name__ == "__main__":
    main()
