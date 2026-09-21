import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import schema


def top_k_table(candidates, scores, k=10, columns=None):
    frame = candidates[[schema.USER, schema.MOVIE]].copy()
    frame["_score"] = scores
    frame = frame.sort_values(
        [schema.USER, "_score", schema.MOVIE],
        ascending=[True, False, True],
        kind="mergesort",
    )
    frame["rank"] = frame.groupby(schema.USER).cumcount() + 1
    frame = frame[frame["rank"] <= k].copy()
    frame = frame.rename(columns={"_score": schema.SCORE})
    frame["rank"] = frame["rank"].astype("int64")
    if columns is None:
        columns = [schema.USER, "rank", schema.MOVIE, schema.SCORE]
    return frame[columns].reset_index(drop=True)


def write_top10(frame, csv_path, meta_path, meta):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    payload = dict(meta)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["n_rows"] = int(len(frame))
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return csv_path
