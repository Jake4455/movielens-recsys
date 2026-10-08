import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data import schema
from src.utils.config import get


def _rating_distribution(ratings):
    counts = ratings[schema.RATING].value_counts().sort_index()
    return {f"{float(k):.1f}": int(v) for k, v in counts.items()}


def validate_ratings(ratings, movies=None, tags=None):
    ts = ratings[schema.TIMESTAMP]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ratings": {
            "n_rows": int(len(ratings)),
            "n_users": int(ratings[schema.USER].nunique()),
            "n_movies": int(ratings[schema.MOVIE].nunique()),
            "n_duplicate_user_movie": int(
                ratings.duplicated(subset=[schema.USER, schema.MOVIE]).sum()
            ),
            "n_missing": {c: int(ratings[c].isna().sum()) for c in ratings.columns},
            "rating_min": float(ratings[schema.RATING].min()),
            "rating_max": float(ratings[schema.RATING].max()),
            "rating_values": sorted(ratings[schema.RATING].unique().tolist()),
            "rating_distribution": _rating_distribution(ratings),
            "timestamp_min": int(ts.min()),
            "timestamp_max": int(ts.max()),
            "timestamp_min_utc": datetime.fromtimestamp(
                int(ts.min()), tz=timezone.utc
            ).isoformat(),
            "timestamp_max_utc": datetime.fromtimestamp(
                int(ts.max()), tz=timezone.utc
            ).isoformat(),
            "n_timestamp_out_of_range": int((ts < 0).sum() + (ts > 2_147_483_647).sum()),
        },
    }
    checks = []
    r = report["ratings"]
    checks.append(("duplicates", r["n_duplicate_user_movie"] == 0, r["n_duplicate_user_movie"]))
    checks.append(("rating_range", r["rating_min"] >= 0.5 and r["rating_max"] <= 5.0, [r["rating_min"], r["rating_max"]]))
    checks.append(("timestamp_range", r["n_timestamp_out_of_range"] == 0, r["n_timestamp_out_of_range"]))
    checks.append(("no_missing_ratings", all(v == 0 for v in r["n_missing"].values()), r["n_missing"]))
    if movies is not None:
        missing = set(ratings[schema.MOVIE].unique()) - set(movies[schema.MOVIE].unique())
        report["movies"] = {
            "n_rows": int(len(movies)),
            "n_missing_title": int(movies[schema.TITLE].isna().sum()),
            "n_missing_genres": int(movies[schema.GENRES].isna().sum()),
        }
        report["coverage"] = {"n_movies_in_ratings_not_in_movies": len(missing)}
        checks.append(("movie_coverage", len(missing) == 0, len(missing)))
    if tags is not None:
        report["tags"] = {
            "n_rows": int(len(tags)),
            "n_users": int(tags[schema.USER].nunique()),
            "n_movies": int(tags[schema.MOVIE].nunique()),
            "n_missing_tag": int(tags[schema.TAG].isna().sum()),
        }
    report["checks"] = [
        {"name": name, "passed": bool(passed), "detail": detail}
        for name, passed, detail in checks
    ]
    report["all_passed"] = all(bool(passed) for _, passed, _ in checks)
    return report


def write_report(report, json_path, md_path):
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    lines = ["# Data Quality Report", ""]
    lines.append(f"- generated_at: {report['generated_at']}")
    lines.append(f"- all_passed: {report['all_passed']}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | passed | detail |")
    lines.append("|---|---|---|")
    for check in report["checks"]:
        lines.append(f"| {check['name']} | {check['passed']} | {check['detail']} |")
    lines.append("")
    lines.append("## Ratings")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|---|---|")
    for key, value in report["ratings"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    if "movies" in report:
        lines.append("## Movies")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for key, value in report["movies"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    if "tags" in report:
        lines.append("## Tags")
        lines.append("")
        lines.append("| key | value |")
        lines.append("|---|---|")
        for key, value in report["tags"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
