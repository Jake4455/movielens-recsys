import numpy as np
import pandas as pd

from src.data import schema
from src.data.candidates import _sample_negatives, _user_code_arrays
from src.utils.config import get


def build_groups(
    cfg,
    positives,
    exclude,
    movie_ids,
    popularity=None,
    seed=0,
    max_users=None,
    hard_ratio=None,
):
    n_negatives = int(get(cfg, "ltr.num_negatives", 100))
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    if hard_ratio is None:
        hard_ratio = float(get(cfg, "ltr.hard_ratio", 0.0))
    hard_ratio = float(hard_ratio)
    hard_source = str(get(cfg, "ltr.hard_negative_source", "popular"))
    hard_pool_size = int(get(cfg, "ltr.hard_pool_size", 5000))
    rng = np.random.default_rng(int(seed))

    ordered = positives.sort_values(
        [schema.USER, schema.TIMESTAMP, schema.MOVIE], kind="mergesort"
    )
    key_series = pd.Series(rng.random(len(ordered)), index=ordered.index)
    picked = key_series.groupby(ordered[schema.USER]).idxmax()
    chosen = ordered.loc[picked.values].sort_values(schema.USER)
    if max_users is not None:
        users = chosen[schema.USER].unique()
        if users.size > int(max_users):
            keep = rng.choice(users, size=int(max_users), replace=False)
            chosen = chosen[chosen[schema.USER].isin(set(keep.tolist()))].sort_values(
                schema.USER
            )
    users = np.sort(chosen[schema.USER].unique())

    index = pd.Index(movie_ids)
    n_movies = int(len(movie_ids))
    exclude_subset = exclude[exclude[schema.USER].isin(set(users.tolist()))]
    exclude_map = _user_code_arrays(exclude_subset, index)

    n_hard = int(round(n_negatives * hard_ratio))
    n_easy = n_negatives - n_hard
    top_pool = None
    if n_hard > 0 and hard_source == "popular" and popularity is not None:
        order = np.argsort(-np.asarray(popularity, dtype=np.float64))
        top_pool = order[: min(int(hard_pool_size), n_movies)]
    real_negative_map = {}
    if n_hard > 0 and hard_source == "real":
        negatives = exclude[exclude[schema.RATING] < threshold]
        negatives = negatives[negatives[schema.USER].isin(set(users.tolist()))]
        real_negative_map = _user_code_arrays(negatives, index)
    if hard_source not in {"popular", "real"}:
        raise ValueError(f"unknown hard_negative_source: {hard_source}")

    positive_movie = dict(zip(chosen[schema.USER].tolist(), chosen[schema.MOVIE].tolist()))
    positive_ts = dict(
        zip(chosen[schema.USER].tolist(), chosen[schema.TIMESTAMP].tolist())
    )
    total = users.size * (n_negatives + 1)
    out_users = np.empty(total, dtype=np.int64)
    out_movies = np.empty(total, dtype=np.int64)
    out_labels = np.zeros(total, dtype=np.int8)
    out_target_ts = np.zeros(total, dtype=np.int64)
    blocked = np.zeros(n_movies, dtype=bool)
    cursor = 0
    for user in users:
        user = int(user)
        block = exclude_map.get(user)
        if block is not None and block.size:
            blocked[block] = True
        positive_id = int(positive_movie[user])
        positive_code = int(index.get_loc(positive_id))
        if n_hard > 0 and real_negative_map:
            pool = real_negative_map.get(user)
            if pool is not None and pool.size:
                take = min(n_hard, int(pool.size))
                hard = rng.choice(pool, size=take, replace=False)
                if take < n_hard:
                    extra = _sample_negatives(rng, blocked, n_hard - take, n_movies)
                    hard = np.concatenate([hard, extra])
            else:
                hard = _sample_negatives(rng, blocked, n_hard, n_movies)
        elif n_hard > 0 and top_pool is not None:
            allowed = top_pool[~blocked[top_pool]]
            if allowed.size >= n_hard:
                hard = rng.choice(allowed, size=n_hard, replace=False)
            else:
                hard = _sample_negatives(rng, blocked, n_hard, n_movies)
        else:
            hard = np.empty(0, dtype=np.int64)
        if hard.size:
            blocked[hard] = True
        if n_easy > 0:
            easy = _sample_negatives(rng, blocked, n_easy, n_movies)
        else:
            easy = np.empty(0, dtype=np.int64)
        if hard.size:
            blocked[hard] = False
        if block is not None and block.size:
            blocked[block] = False
        target_ts = int(positive_ts[user])
        out_users[cursor] = user
        out_movies[cursor] = positive_id
        out_labels[cursor] = 1
        out_target_ts[cursor] = target_ts
        cursor += 1
        for source in (easy, hard):
            if source.size:
                out_users[cursor : cursor + source.size] = user
                out_movies[cursor : cursor + source.size] = movie_ids[source]
                out_target_ts[cursor : cursor + source.size] = target_ts
                cursor += source.size
        if positive_code < 0 or positive_id not in positive_movie.values():
            raise RuntimeError("invalid positive mapping")
    return pd.DataFrame(
        {
            schema.USER: out_users,
            schema.MOVIE: out_movies,
            schema.LABEL: out_labels,
            schema.TARGET_TS: out_target_ts,
        }
    )


def group_sizes(frame, n_negatives=100):
    counts = frame.groupby(schema.USER, sort=False).size().to_numpy()
    expected = int(n_negatives) + 1
    if not np.all(counts == expected):
        bad = int((counts != expected).sum())
        raise ValueError(f"group size mismatch for {bad} users (expected {expected})")
    return counts.astype(np.int32)
