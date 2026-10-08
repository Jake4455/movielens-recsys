import numpy as np
import pandas as pd
import pytest

from src.data import schema
from src.data.candidates import _sample_negatives, build_candidates
from src.data.split import frames_by_split, split_per_user
from src.eval.ndcg import evaluate_candidates
from src.eval.significance import paired_bootstrap, seed_variance


def make_ratings(n_users=2, n_per_user=10):
    rows = []
    for user in range(1, n_users + 1):
        for i in range(n_per_user):
            rows.append(
                {
                    schema.USER: user,
                    schema.MOVIE: user * 10 + i,
                    schema.RATING: 5.0 if i % 3 == 0 else 2.0,
                    schema.TIMESTAMP: 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows)


def make_movies(n_movies=60):
    return pd.DataFrame(
        {
            schema.MOVIE: np.arange(1, n_movies + 1, dtype=np.int64),
            schema.TITLE: [f"movie {i}" for i in range(1, n_movies + 1)],
            schema.GENRES: ["Drama"] * n_movies,
        }
    )


def make_cfg(n_negatives=100):
    return {
        "seed": 20260907,
        "data": {
            "positive_threshold": 4.0,
            "candidates": {
                "num_negatives": n_negatives,
                "positive_pick": "random",
                "negative_scope": "per_user",
                "exclude": {
                    "positive": True,
                    "user_train_val": True,
                    "user_test": True,
                },
            },
        },
    }


def test_split_is_temporal_per_user():
    frame = split_per_user(make_ratings(n_users=3, n_per_user=10))
    splits = frames_by_split(frame)
    assert len(splits[schema.TRAIN]) == 24
    assert len(splits[schema.VAL]) == 3
    assert len(splits[schema.TEST]) == 3
    for user in [1, 2, 3]:
        user_frame = frame[frame[schema.USER] == user]
        train_ts = user_frame[user_frame[schema.SPLIT] == schema.TRAIN][schema.TIMESTAMP].max()
        val_ts = user_frame[user_frame[schema.SPLIT] == schema.VAL][schema.TIMESTAMP].max()
        test_ts = user_frame[user_frame[schema.SPLIT] == schema.TEST][schema.TIMESTAMP].max()
        assert train_ts < val_ts < test_ts


def test_candidates_protocol_shape_and_exclusions():
    movies = make_movies(60)
    frame = split_per_user(make_ratings(n_users=2, n_per_user=10))
    splits = frames_by_split(frame)
    candidates, meta = build_candidates(splits, movies, make_cfg(n_negatives=50), seed=20260907)
    assert meta["n_users"] == 2
    grouped = candidates.groupby(schema.USER)
    assert set(grouped.size().unique()) == {51}
    assert set(grouped[schema.LABEL].sum().unique()) == {1}
    assert not candidates.duplicated(subset=[schema.USER, schema.MOVIE]).any()
    for user in [1, 2]:
        seen = set(
            splits[schema.TRAIN][splits[schema.TRAIN][schema.USER] == user][schema.MOVIE]
        ) | set(
            splits[schema.VAL][splits[schema.VAL][schema.USER] == user][schema.MOVIE]
        )
        test_movies = set(
            splits[schema.TEST][splits[schema.TEST][schema.USER] == user][schema.MOVIE]
        )
        positive = candidates[
            (candidates[schema.USER] == user) & (candidates[schema.LABEL] == 1)
        ][schema.MOVIE].iloc[0]
        negatives = set(
            candidates[
                (candidates[schema.USER] == user) & (candidates[schema.LABEL] == 0)
            ][schema.MOVIE]
        )
        assert positive in test_movies
        assert not (negatives & seen)
        assert not (negatives & test_movies)


def test_candidates_are_deterministic():
    movies = make_movies(60)
    frame = split_per_user(make_ratings(n_users=2, n_per_user=10))
    splits = frames_by_split(frame)
    first, _ = build_candidates(splits, movies, make_cfg(50), seed=7)
    second, _ = build_candidates(splits, movies, make_cfg(50), seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_ndcg_matches_position_table():
    rows = []
    scores = []
    for user in range(1, 4):
        for position in range(1, 14):
            rows.append(
                {
                    schema.USER: user,
                    schema.MOVIE: user * 1000 + position,
                    schema.LABEL: 1 if position == 1 else 0,
                }
            )
            if user == 1:
                scores.append(1.0 if position == 1 else 0.0)
            elif user == 2:
                scores.append(0.0 if position == 1 else (1.0 if position == 2 else 0.0))
            else:
                scores.append(0.0 if position == 1 else (1.0 if position <= 12 else 0.0))
    frame = pd.DataFrame(rows)
    scores = np.asarray(scores, dtype=np.float64)
    metrics, per_user = evaluate_candidates(frame, scores, k=10)
    by_user = per_user.set_index(schema.USER)
    assert abs(by_user.loc[1, "ndcg"] - 1.0) < 1e-12
    assert abs(by_user.loc[2, "ndcg"] - 0.6309297535714574) < 1e-9
    assert abs(by_user.loc[3, "ndcg"] - 0.0) < 1e-12
    expected = (1.0 + 0.6309297535714574 + 0.0) / 3
    assert abs(metrics["ndcg_at_k"] - expected) < 1e-9
    assert abs(metrics["hr_at_k"] - 2 / 3) < 1e-12


def test_significance_ci_and_seed_variance():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.05, 0.2, size=5000)
    result = paired_bootstrap(diffs, n_boot=200, seed=1)
    assert result["ci_low"] > 0
    assert result["p_one_sided"] < 0.05
    zero = paired_bootstrap(np.zeros(1000), n_boot=200, seed=1)
    assert abs(zero["ci_low"]) < 1e-12
    variance = seed_variance([0.22, 0.24, 0.23], [1.10, 1.20, 1.15])
    assert variance["n_seeds"] == 3
    assert variance["share_lift_ge_target"] == 1.0
    assert variance["sigma_seed"] > 0


def test_negative_sampling_is_uniform_over_the_whole_catalog():
    """Regression: truncating the sorted ``np.unique`` draw kept the smallest codes only.

    With the old code the retained codes were the 100 smallest of ~250 draws, so the
    pool never exceeded ~40% of the catalog and averaged ~20% of the id range.
    """
    n_movies, n_negatives = 1000, 100
    blocked = np.zeros(n_movies, dtype=bool)
    rng = np.random.default_rng(20260907)
    picks = np.concatenate(
        [_sample_negatives(rng, blocked, n_negatives, n_movies) for _ in range(50)]
    )
    assert picks.size == 5000
    assert picks.max() > 0.9 * n_movies, "negatives never reach the top of the id range"
    assert abs(picks.mean() - n_movies / 2) < 0.05 * n_movies, "negative codes are biased low"


def test_negative_sampling_respects_blocked_items():
    n_movies, n_negatives = 200, 50
    blocked = np.zeros(n_movies, dtype=bool)
    blocked[: n_movies - n_negatives] = True
    picks = _sample_negatives(np.random.default_rng(1), blocked, n_negatives, n_movies)
    assert sorted(picks.tolist()) == list(range(n_movies - n_negatives, n_movies))


def test_ndcg_enforces_expected_group_size():
    rows, scores = [], []
    for user in range(1, 3):
        for position in range(1, 52):
            rows.append(
                {
                    schema.USER: user,
                    schema.MOVIE: user * 1000 + position,
                    schema.LABEL: 1 if position == 1 else 0,
                }
            )
            scores.append(1.0 if position == 1 else 0.0)
    frame = pd.DataFrame(rows)
    scores = np.asarray(scores, dtype=np.float64)
    evaluate_candidates(frame, scores, k=10, expected_group_size=51)
    with pytest.raises(ValueError, match="candidates per user"):
        evaluate_candidates(frame, scores, k=10, expected_group_size=101)


def test_teacher_source_fails_loudly():
    movies = make_movies(60)
    frame = split_per_user(make_ratings(n_users=2, n_per_user=10))
    splits = frames_by_split(frame)
    cfg = make_cfg(50)
    cfg["data"]["candidates"]["source"] = "teacher"
    cfg["data"]["candidates"]["teacher_file"] = "some/teacher.csv"
    with pytest.raises(NotImplementedError, match="teacher"):
        build_candidates(splits, movies, cfg, seed=20260907)
