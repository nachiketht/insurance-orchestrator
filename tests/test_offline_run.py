import json
import subprocess
import sys
from pathlib import Path


def test_offline_run_covers_the_pipeline_and_a_timeout(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/run_cases.py", "--offline", "--out", str(tmp_path)],
        check=True,
    )
    written = sorted(path.name for path in tmp_path.glob("*.json"))
    assert written == [
        "clean_auto.json",
        "high_risk_home.json",
        "no_hit_life.json",
        "timeout_case.json",
    ]
    clean = json.loads((tmp_path / "clean_auto.json").read_text())
    home = json.loads((tmp_path / "high_risk_home.json").read_text())
    life = json.loads((tmp_path / "no_hit_life.json").read_text())
    timeout = json.loads((tmp_path / "timeout_case.json").read_text())
    assert clean["final_case"]["recommendation"]["decision"] == "approve"
    assert home["final_case"]["recommendation"]["decision"] == "deny"
    assert life["final_case"]["recommendation"]["decision"] == "refer"
    assert timeout["final_case"]["status"] == "escalated"
    assert timeout["final_case"]["recommendation"]["source"] == "system_escalation"
    assert timeout["final_case"]["risk"] is None
