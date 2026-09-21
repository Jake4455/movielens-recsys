import numpy as np


def paired_bootstrap(diffs, n_boot=1000, seed=20260907, chunk=25, alpha=0.05):
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.size
    if n == 0:
        raise ValueError("diffs is empty")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for start in range(0, int(n_boot), int(chunk)):
        size = min(int(chunk), int(n_boot) - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start : start + size] = diffs[idx].mean(axis=1)
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    p_value = float(((means <= 0.0).sum() + 1) / (int(n_boot) + 1))
    se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean_diff": float(diffs.mean()),
        "se": se,
        "ci_low": lower,
        "ci_high": upper,
        "p_one_sided": p_value,
        "n_users": int(n),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def seed_variance(ndcgs, lifts, target_lift=1.1):
    ndcgs = np.asarray(ndcgs, dtype=np.float64)
    lifts = np.asarray(lifts, dtype=np.float64)
    return {
        "n_seeds": int(ndcgs.size),
        "ndcg_mean": float(ndcgs.mean()),
        "sigma_seed": float(ndcgs.std(ddof=1)) if ndcgs.size > 1 else 0.0,
        "min_lift": float(lifts.min()) if lifts.size else 0.0,
        "share_lift_ge_target": float((lifts >= target_lift).mean()) if lifts.size else 0.0,
        "target_lift": float(target_lift),
    }
