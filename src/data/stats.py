from datetime import datetime, timezone

import pandas as pd

from src.data import schema
from src.utils.config import get


def _percentiles(values):
    series = pd.Series(values)
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    return {f"p{int(q * 100)}": float(series.quantile(q)) for q in quantiles}


def describe_splits(splits, movies=None, threshold=4.0):
    train = splits[schema.TRAIN]
    n_users = int(train[schema.USER].nunique())
    n_movies = int(train[schema.MOVIE].nunique())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "positive_threshold": float(threshold),
        "n_ratings_total": int(sum(len(frame) for frame in splits.values())),
        "n_users": n_users,
        "n_movies": n_movies,
        "splits": {},
    }
    for name, frame in splits.items():
        n = len(frame)
        positives = int((frame[schema.RATING] >= threshold).sum())
        report["splits"][name] = {
            "n_ratings": int(n),
            "ratio": float(n / report["n_ratings_total"]) if report["n_ratings_total"] else 0.0,
            "n_users": int(frame[schema.USER].nunique()),
            "n_movies": int(frame[schema.MOVIE].nunique()),
            "n_positives": positives,
            "positive_ratio": float(positives / n) if n else 0.0,
            "mean_rating": float(frame[schema.RATING].mean()),
        }
    if movies is not None:
        report["n_movies_catalog"] = int(len(movies))
        report["sparsity"] = float(
            report["n_ratings_total"] / (n_users * int(len(movies)))
        )
    report["per_user_ratings"] = _percentiles(train.groupby(schema.USER).size())
    report["per_movie_ratings"] = _percentiles(train.groupby(schema.MOVIE).size())
    counts = train.groupby(schema.MOVIE).size()
    report["long_tail"] = {
        "n_movies_lt_5_ratings": int((counts < 5).sum()),
        "n_movies_lt_20_ratings": int((counts < 20).sum()),
        "share_movies_lt_5_ratings": float((counts < 5).mean()),
    }
    return report


def write_stats(report, json_path, md_path):
    import json

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    lines = ["# Dataset Statistics", ""]
    lines.append(f"- generated_at: {report['generated_at']}")
    lines.append(f"- positive_threshold: {report['positive_threshold']}")
    lines.append(f"- n_ratings_total: {report['n_ratings_total']}")
    lines.append(f"- n_users: {report['n_users']}")
    lines.append(f"- n_movies: {report['n_movies']}")
    if "sparsity" in report:
        lines.append(f"- sparsity: {report['sparsity']:.6f}")
    lines.append("")
    lines.append("## Splits")
    lines.append("")
    lines.append("| split | n_ratings | ratio | n_users | n_movies | n_positives | positive_ratio | mean_rating |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, entry in report["splits"].items():
        lines.append(
            f"| {name} | {entry['n_ratings']} | {entry['ratio']:.4f} | {entry['n_users']} | "
            f"{entry['n_movies']} | {entry['n_positives']} | {entry['positive_ratio']:.4f} | "
            f"{entry['mean_rating']:.4f} |"
        )
    lines.append("")
    for key in ["per_user_ratings", "per_movie_ratings"]:
        lines.append(f"## {key}")
        lines.append("")
        lines.append("| " + " | ".join(report[key].keys()) + " |")
        lines.append("|" + "---|" * len(report[key]))
        lines.append("| " + " | ".join(f"{v:.1f}" for v in report[key].values()) + " |")
        lines.append("")
    lines.append("## Long Tail")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|---|---|")
    for key, value in report["long_tail"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
