import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data import schema
from src.utils.config import get


def _user_code_arrays(frame, index, user_col=schema.USER, movie_col=schema.MOVIE):
    if frame.empty:
        return {}
    codes = index.get_indexer(frame[movie_col].to_numpy())
    users = frame[user_col].to_numpy()
    order = np.argsort(users, kind="stable")
    users_sorted = users[order]
    codes_sorted = codes[order].astype(np.int64)
    uniq, first = np.unique(users_sorted, return_index=True)
    boundaries = np.append(first, len(users_sorted))
    return {
        int(user): codes_sorted[boundaries[i] : boundaries[i + 1]]
        for i, user in enumerate(uniq)
    }


def _pick_positive(test_pos, pick, rng):
    ordered = test_pos.sort_values(
        [schema.USER, schema.TIMESTAMP, schema.MOVIE], kind="mergesort"
    )
    if pick == "earliest":
        return ordered.groupby(schema.USER, sort=True).head(1)
    if pick == "latest":
        return ordered.groupby(schema.USER, sort=True).tail(1)
    if pick != "random":
        raise ValueError(f"unknown positive_pick: {pick}")
    scored = ordered.copy()
    scored["_key"] = rng.random(len(scored))
    index = scored.groupby(schema.USER, sort=True)["_key"].idxmax()
    return scored.loc[index.values].sort_values(schema.USER)


def _sample_negatives(rng, blocked, n_negatives, n_movies):
    """Uniformly sample ``n_negatives`` distinct codes from the non-blocked catalog.

    ``np.unique`` sorts its result ascending, so truncating it directly would keep
    the *smallest* codes of the draw instead of a uniform subset (that made ~51% of
    the catalog unreachable and biased the negative pool towards popular low-id
    movies, inflating the reported lift). A uniform subsample of a uniform sample is
    itself uniform, hence the permutation below.
    """
    collected = np.empty(0, dtype=np.int64)
    attempts = 0
    while collected.size < n_negatives and attempts < 64:
        draw = rng.integers(0, n_movies, size=max(n_negatives * 2, 256))
        valid = draw[~blocked[draw]]
        if valid.size:
            collected = np.unique(np.concatenate([collected, valid]))
        attempts += 1
    if collected.size < n_negatives:
        allowed = np.flatnonzero(~blocked)
        collected = rng.choice(allowed, size=n_negatives, replace=False)
    else:
        collected = rng.permutation(collected)[:n_negatives]
    return collected


def build_candidates(splits, movies, cfg, seed):
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    n_negatives = int(get(cfg, "data.candidates.num_negatives", 100))
    pick = get(cfg, "data.candidates.positive_pick", "random")
    scope = get(cfg, "data.candidates.negative_scope", "per_user")
    if scope != "per_user":
        raise NotImplementedError(f"negative_scope not supported yet: {scope}")
    source = str(get(cfg, "data.candidates.source", "local"))
    teacher_file = get(cfg, "data.candidates.teacher_file")
    if source != "local":
        # The "teacher candidate file first" channel (PRD FR-1.5 / Q7) is not
        # implemented; failing loudly is better than silently emitting locally
        # reproduced candidates while recording a misleading provenance in the meta.
        raise NotImplementedError(
            f"data.candidates.source={source!r} is not supported yet "
            f"(teacher_file={teacher_file!r}); only 'local' is implemented"
        )
    exclude_positive = bool(get(cfg, "data.candidates.exclude.positive", True))
    exclude_train_val = bool(get(cfg, "data.candidates.exclude.user_train_val", True))
    exclude_test = bool(get(cfg, "data.candidates.exclude.user_test", True))

    test = splits[schema.TEST]
    test_pos = test[test[schema.RATING] >= threshold]
    if test_pos.empty:
        raise RuntimeError("no positive interactions in test split")
    rng = np.random.default_rng(int(seed))
    chosen = _pick_positive(test_pos, pick, rng)

    movie_ids = np.sort(movies[schema.MOVIE].unique())
    index = pd.Index(movie_ids)
    n_movies = int(movie_ids.size)

    eligible = chosen[schema.USER].to_numpy()
    train_val = pd.concat([splits[schema.TRAIN], splits[schema.VAL]])
    train_val = train_val[train_val[schema.USER].isin(set(eligible.tolist()))]
    test_only = test[test[schema.USER].isin(set(eligible.tolist()))]
    train_val_map = _user_code_arrays(train_val, index)
    test_map = _user_code_arrays(test_only, index)

    positive_movie = dict(zip(chosen[schema.USER].tolist(), chosen[schema.MOVIE].tolist()))
    total_rows = eligible.size * (n_negatives + 1)
    out_users = np.empty(total_rows, dtype=np.int64)
    out_movies = np.empty(total_rows, dtype=np.int64)
    out_labels = np.zeros(total_rows, dtype=np.int8)
    out_target_ts = np.zeros(total_rows, dtype=np.int64)
    positive_ts = dict(
        zip(chosen[schema.USER].tolist(), chosen[schema.TIMESTAMP].tolist())
    )

    blocked = np.zeros(n_movies, dtype=bool)
    cursor = 0
    for user in np.sort(eligible):
        user = int(user)
        blocks = []
        if exclude_train_val and user in train_val_map:
            blocks.append(train_val_map[user])
        if exclude_test and user in test_map:
            blocks.append(test_map[user])
        if blocks:
            blocked[np.concatenate(blocks)] = True
        positive_movie_id = int(positive_movie[user])
        positive_code = int(index.get_loc(positive_movie_id))
        if not exclude_positive:
            blocked[positive_code] = False
        n_blocked = int(blocked.sum())
        if n_movies - n_blocked < n_negatives:
            raise RuntimeError(
                f"user {user} has only {n_movies - n_blocked} allowed items "
                f"for {n_negatives} negatives"
            )
        negatives = _sample_negatives(rng, blocked, n_negatives, n_movies)
        if exclude_positive or blocks:
            blocked[np.concatenate(blocks)] = False
        target_ts = int(positive_ts[user])
        out_users[cursor] = user
        out_movies[cursor] = positive_movie_id
        out_labels[cursor] = 1
        out_target_ts[cursor] = target_ts
        cursor += 1
        out_users[cursor : cursor + n_negatives] = user
        out_movies[cursor : cursor + n_negatives] = movie_ids[negatives]
        out_target_ts[cursor : cursor + n_negatives] = target_ts
        cursor += n_negatives

    candidates = pd.DataFrame(
        {
            schema.USER: out_users,
            schema.MOVIE: out_movies,
            schema.LABEL: out_labels,
            schema.TARGET_TS: out_target_ts,
        }
    )
    candidates.insert(0, schema.CAND_ID, np.arange(len(candidates), dtype=np.int64))
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "protocol": "1 positive + K fixed negatives per test user",
        "num_negatives": n_negatives,
        "positive_pick": pick,
        "negative_scope": scope,
        "exclude": {
            "positive": exclude_positive,
            "user_train_val": exclude_train_val,
            "user_test": exclude_test,
        },
        "n_users": int(eligible.size),
        "n_rows": int(len(candidates)),
        "pool_size": n_movies,
        "positive_threshold": threshold,
        "source": source,
        "teacher_file": teacher_file if source != "local" else None,
    }
    return candidates, meta


def write_candidates(candidates, meta, parquet_path, meta_path):
    candidates.to_parquet(parquet_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
