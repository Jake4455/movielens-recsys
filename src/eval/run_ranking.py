import json
from pathlib import Path

import pandas as pd

from src.data import schema
from src.eval.leaderboard import append_run
from src.eval.ndcg import evaluate_candidates
from src.eval.significance import paired_bootstrap
from src.serving.write_top10 import top_k_table, write_top10
from src.utils.config import ensure_dirs, get, path_of


def _significance_vs_baseline(cfg, per_user, run_id, metrics_dir):
    baseline_path = metrics_dir / "R00_pop_per_user.parquet"
    if run_id == "R00_pop" or not baseline_path.exists():
        return None, None
    baseline = pd.read_parquet(baseline_path)
    merged = per_user[[schema.USER, "ndcg"]].merge(
        baseline[[schema.USER, "ndcg"]],
        on=schema.USER,
        suffixes=("", "_base"),
        how="inner",
    )
    if len(merged) != len(per_user) or len(merged) != len(baseline):
        raise ValueError(
            "paired significance requires identical user sets: "
            f"model={len(per_user)} baseline={len(baseline)} merged={len(merged)}"
        )
    diffs = merged["ndcg"].to_numpy() - merged["ndcg_base"].to_numpy()
    bootstrap_cfg = get(cfg, "eval.bootstrap", {}) or {}
    result = paired_bootstrap(
        diffs,
        n_boot=int(bootstrap_cfg.get("n_boot", 1000)),
        seed=int(bootstrap_cfg.get("seed", get(cfg, "seed", 20260907))),
        chunk=int(bootstrap_cfg.get("chunk", 25)),
    )
    base_mean = float(merged["ndcg_base"].mean())
    model_mean = float(merged["ndcg"].mean())
    result["baseline_ndcg"] = base_mean
    result["model_ndcg"] = model_mean
    result["lift"] = model_mean / base_mean if base_mean > 0 else 0.0
    return result, result["lift"]


def rank_and_report(
    cfg, candidates, scores, run_id, model_name, elapsed_s, extra=None, infer_s=None
):
    """Persist metrics/artifacts for one ranking arm.

    ``elapsed_s`` is the whole stage wall clock (pipeline proxy); ``infer_s`` is the
    scoring-only time used for the serving latency/throughput reported in the
    Top-10 metadata. Keeping both avoids labelling pipeline time as serving latency.
    """
    outputs = path_of(cfg, "paths.outputs_dir")
    metrics_dir = outputs / "metrics"
    top10_dir = outputs / "top10"
    ensure_dirs(metrics_dir, top10_dir)
    k = int(get(cfg, "eval.k", 10))
    metrics, per_user = evaluate_candidates(
        candidates,
        scores,
        k=k,
        expected_group_size=int(get(cfg, "data.candidates.num_negatives", 100)) + 1,
    )
    top10 = top_k_table(
        candidates,
        scores,
        k=int(get(cfg, "serving.top_k", 10)),
        columns=get(cfg, "serving.columns"),
    )
    n_users = metrics["n_users"]
    metrics.update(
        {
            "run_id": run_id,
            "model": model_name,
            "seed": int(get(cfg, "seed", 20260907)),
            "pipeline_s": float(elapsed_s),
            "throughput_rps": float(len(candidates) / elapsed_s) if elapsed_s > 0 else 0.0,
            "latency_ms": float(elapsed_s / n_users * 1000.0) if n_users else 0.0,
        }
    )
    serving_latency = metrics["latency_ms"]
    serving_rps = metrics["throughput_rps"]
    serving_scope = "pipeline_proxy"
    if infer_s is not None and infer_s > 0:
        serving_latency = float(infer_s / n_users * 1000.0) if n_users else 0.0
        serving_rps = float(len(candidates) / infer_s)
        serving_scope = "inference_only"
        metrics.update(
            {
                "infer_s": float(infer_s),
                "infer_latency_ms": serving_latency,
                "infer_throughput_rps": serving_rps,
            }
        )
    if extra:
        metrics.update(extra)
    significance, lift = _significance_vs_baseline(cfg, per_user, run_id, metrics_dir)
    if significance is not None:
        metrics["significance"] = significance
    per_user.to_parquet(metrics_dir / f"{run_id}_per_user.parquet", index=False)
    with open(metrics_dir / f"{run_id}.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=str)
    write_top10(
        top10,
        top10_dir / f"{run_id}.csv",
        top10_dir / f"{run_id}_meta.json",
        {
            "run_id": run_id,
            "model": model_name,
            "seed": int(get(cfg, "seed", 20260907)),
            "ndcg_at10": metrics["ndcg_at_k"],
            "latency_ms": serving_latency,
            "throughput_rps": serving_rps,
            "latency_scope": serving_scope,
        },
    )
    append_run(
        outputs / "leaderboard.csv",
        {
            "run_id": run_id,
            "model": model_name,
            "recall_at10": f"{metrics['recall_at_k']:.6f}",
            "ndcg_at10": f"{metrics['ndcg_at_k']:.6f}",
            "lift_vs_pop": f"{lift:.4f}" if lift is not None else "",
            "ci_low": (
                f"{significance['ci_low']:.6f}" if significance is not None else ""
            ),
            "throughput_rps": f"{metrics['throughput_rps']:.2f}",
            "latency_ms": f"{metrics['latency_ms']:.4f}",
            "seed": int(get(cfg, "seed", 20260907)),
            "config_path": cfg["_config_path"],
        },
    )
    return metrics, per_user, significance
