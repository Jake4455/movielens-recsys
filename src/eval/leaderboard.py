import csv
import os
from datetime import datetime, timezone
from pathlib import Path

LEADERBOARD_COLUMNS = [
    "run_id",
    "model",
    "recall_at10",
    "ndcg_at10",
    "lift_vs_pop",
    "ci_low",
    "throughput_rps",
    "latency_ms",
    "seed",
    "config_path",
    "timestamp",
]


def append_run(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            if header != LEADERBOARD_COLUMNS:
                raise ValueError(
                    "leaderboard header mismatch: expected "
                    f"{LEADERBOARD_COLUMNS}, found {header}"
                )
            rows = list(reader)
    normalized = {column: row.get(column, "") for column in LEADERBOARD_COLUMNS}
    if not normalized.get("timestamp"):
        normalized["timestamp"] = datetime.now(timezone.utc).isoformat()
    rows = [existing for existing in rows if existing.get("run_id") != normalized["run_id"]]
    rows.append(normalized)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)
    return path
