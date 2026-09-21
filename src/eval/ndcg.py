import numpy as np
import pandas as pd

from src.data import schema


def evaluate_candidates(candidates, scores, k=10):
    user = candidates[schema.USER].to_numpy()
    movie = candidates[schema.MOVIE].to_numpy()
    label = candidates[schema.LABEL].to_numpy()
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size != label.size:
        raise ValueError("scores and candidates length mismatch")
    codes, uniques = pd.factorize(user, sort=True)
    n_users = len(uniques)
    positive_mask = label == 1
    positives_per_user = np.bincount(codes[positive_mask], minlength=n_users)
    if not np.all(positives_per_user == 1):
        bad = int((positives_per_user != 1).sum())
        raise ValueError(f"protocol requires exactly 1 positive per user, violated for {bad} users")
    positive_scores = np.zeros(n_users, dtype=np.float64)
    positive_scores[codes[positive_mask]] = scores[positive_mask]
    positive_movies = np.zeros(n_users, dtype=movie.dtype)
    positive_movies[codes[positive_mask]] = movie[positive_mask]
    negatives = ~positive_mask
    neg_user = codes[negatives]
    neg_scores = scores[negatives]
    neg_movies = movie[negatives]
    better = (neg_scores > positive_scores[neg_user]) | (
        (neg_scores == positive_scores[neg_user]) & (neg_movies < positive_movies[neg_user])
    )
    rank = 1 + np.bincount(neg_user[better], minlength=n_users)
    hit = rank <= k
    ndcg = np.where(hit, 1.0 / np.log2(rank + 1.0), 0.0)
    per_user = pd.DataFrame(
        {
            schema.USER: uniques,
            "rank": rank.astype(np.int64),
            "ndcg": ndcg.astype(np.float64),
            "hit": hit,
        }
    )
    metrics = {
        "k": int(k),
        "n_users": int(n_users),
        "ndcg_at_k": float(ndcg.mean()),
        "hr_at_k": float(hit.mean()),
        "recall_at_k": float(hit.mean()),
        "mrr_at_k": float(np.where(hit, 1.0 / rank, 0.0).mean()),
        "mean_rank": float(rank.mean()),
    }
    return metrics, per_user
