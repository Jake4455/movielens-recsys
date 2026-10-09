"""保真性自检：把 leaderboard/文档里报出的每个数字，回溯到最底层的产物重算一遍。

用法：python tools/audit_fidelity.py

依赖的产物（其中 *_per_user.parquet 与 candidates.parquet 被 .gitignore 排除，需在本机跑过
流水线才有）：
  outputs/leaderboard.csv, outputs/metrics/*.json, outputs/metrics/*_per_user.parquet,
  outputs/reports/seed_robustness_<run>.json, outputs/top10/submission.csv,
  data/processed/candidates.parquet

检查内容：leaderboard / metrics JSON / 逐用户重算 三者一致；NDCG 是否满足
1/log2(rank+1) 公式；lift 是否可由逐用户均值反推；配对 bootstrap 的 Delta 与 CI 下界是否可复现；
sigma_seed 是否可由五套种子反推；提交文件是否与最终模型 Top-10 逐行一致；候选集是否 1 正 + 100 负。
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(".").resolve()
sys.path.insert(0, str(ROOT))
from src.eval.significance import paired_bootstrap  # noqa: E402

checks = []


def check(name, expected, actual, tol=5e-7):
    ok = abs(float(expected) - float(actual)) <= tol
    checks.append((name, expected, actual, ok))
    return ok


for required in (ROOT / "outputs" / "leaderboard.csv", ROOT / "outputs" / "metrics" / "R00_pop_per_user.parquet"):
    if not required.exists():
        raise SystemExit(f"缺少产物 {required}；请先在本机运行流水线（python run_all.py --config conf/stream.yaml）")

lb = pd.read_csv(ROOT / "outputs" / "leaderboard.csv")
per_user = {}
for run in lb["run_id"]:
    per_user[run] = pd.read_parquet(ROOT / "outputs" / "metrics" / f"{run}_per_user.parquet")

# ---- 1) leaderboard / JSON / 逐用户重算 三者一致
for _, row in lb.iterrows():
    run = row["run_id"]
    frame = per_user[run]
    meta = json.loads((ROOT / "outputs" / "metrics" / f"{run}.json").read_text(encoding="utf-8"))
    check(f"{run} leaderboard.ndcg vs per-user", row["ndcg_at10"], frame["ndcg"].mean())
    check(f"{run} json.ndcg vs per-user", meta["ndcg_at_k"], frame["ndcg"].mean())
    check(f"{run} leaderboard.recall vs per-user HR", row["recall_at10"], (frame["rank"] <= 10).mean())
    check(f"{run} json.recall vs per-user HR", meta["recall_at_k"], (frame["rank"] <= 10).mean())
    # ndcg 与 rank 必须满足协议公式
    expected_ndcg = np.where(frame["rank"] <= 10, 1.0 / np.log2(frame["rank"] + 1.0), 0.0)
    check(f"{run} ndcg == 1/log2(rank+1) 公式", 0.0, np.abs(frame["ndcg"].to_numpy() - expected_ndcg).max())

# ---- 2) lift 可由逐用户均值反推
base = per_user["R00_pop"]["ndcg"].mean()
for _, row in lb.iterrows():
    if row["run_id"] == "R00_pop":
        continue
    check(f"{row['run_id']} lift 可反推", row["lift_vs_pop"], per_user[row["run_id"]]["ndcg"].mean() / base, 5e-5)

# ---- 3) 配对 bootstrap 的 CI 下界可复现（B=1000, seed=20260907）
for run in ("R06_ltr_f1", "R10_ltr_stream", "R13_ltr_itemcf_noes"):
    merged = per_user[run][["userId", "ndcg"]].merge(
        per_user["R00_pop"][["userId", "ndcg"]], on="userId", suffixes=("", "_base")
    )
    result = paired_bootstrap(
        merged["ndcg"].to_numpy() - merged["ndcg_base"].to_numpy(),
        n_boot=1000, seed=20260907, chunk=25,
    )
    stored = json.loads((ROOT / "outputs" / "metrics" / f"{run}.json").read_text(encoding="utf-8"))["significance"]
    check(f"{run} CI 下界可复现", stored["ci_low"], result["ci_low"], 1e-9)
    check(f"{run} Delta 可复现", stored["mean_diff"], result["mean_diff"], 1e-9)

# ---- 4) σ_seed 可由五套种子反推
sr = json.loads((ROOT / "outputs" / "reports" / "seed_robustness_R13_ltr_itemcf_noes.json").read_text(encoding="utf-8"))
ndcgs = np.asarray([e["ndcg"] for e in sr["entries"]])
lifts = np.asarray([e["ndcg"] / e["popularity_ndcg"] for e in sr["entries"]])
check("R13 sigma_seed 可反推", sr["ndcg_std_seed"], ndcgs.std(ddof=1), 1e-12)
check("R13 min_lift 可反推", sr["min_lift"], lifts.min(), 1e-12)
check("R13 share>=1.1 可反推", sr["share_lift_ge_1_1"], float((lifts >= 1.1).mean()), 1e-12)
check("R13 五套种子数", 5, len(sr["entries"]))

# ---- 5) 提交文件与最终模型 Top-10 一致
sub = pd.read_csv(ROOT / "outputs" / "top10" / "submission.csv")
best = pd.read_csv(ROOT / "outputs" / "top10" / "R13_ltr_itemcf_noes.csv")
check("submission 行数", len(best), len(sub))
check("submission 用户数", best["userId"].nunique(), sub["userId"].nunique())
check("submission 与 R13 逐行一致(movieId)", 0, int((sub["movieId"].to_numpy() != best["movieId"].to_numpy()).sum()))

# ---- 6) 候选集协议（1 正 + 100 负）
cand = pd.read_parquet(ROOT / "data" / "processed" / "candidates.parquet", columns=["userId", "label"])
sizes = cand.groupby("userId").size()
check("候选集每用户 101 行", 101, int(sizes.max()))
check("候选集用户数", 187278, cand["userId"].nunique())
check("每用户恰 1 正样本", 1, int(cand.groupby("userId")["label"].sum().max()))

passed = sum(1 for _, _, _, ok in checks if ok)
print(f"{'检查项':<52} {'期望':>14} {'实测':>14}  结果")
print("-" * 96)
for name, expected, actual, ok in checks:
    print(f"{name:<52} {float(expected):>14.9f} {float(actual):>14.9f}  {'PASS' if ok else 'FAIL'}")
print("-" * 96)
print(f"通过 {passed}/{len(checks)}")
failed = [c[0] for c in checks if not c[3]]
if failed:
    print("未通过:", failed)
