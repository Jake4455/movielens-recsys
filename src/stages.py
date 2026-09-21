import json
import time

import numpy as np
import pandas as pd

from src.baseline.popularity import popularity_scores
from src.data import schema
from src.data.candidates import build_candidates, write_candidates
from src.data.io import read_movies, read_ratings, read_tags
from src.data.split import frames_by_split, split_per_user
from src.data.stats import describe_splits, write_stats
from src.data.validate import validate_ratings, write_report
from src.eval.leaderboard import append_run
from src.eval.ndcg import evaluate_candidates
from src.eval.run_ranking import rank_and_report
from src.serving.write_top10 import top_k_table, write_top10
from src.utils.config import ensure_dirs, get, path_of
from src.utils.logging_utils import setup_logger


def _graph_columns(cfg):
    mapping = [
        ("graph_pagerank", "features.graph_pagerank"),
        ("graph_walk2", "features.graph_walk"),
        ("graph_community_affinity", "features.graph_community"),
    ]
    return [name for name, key in mapping if bool(get(cfg, key, True))]


def _raw_path(cfg, key):
    return path_of(cfg, "paths.raw_dir") / get(cfg, f"data.files.{key}")


def _load_splits(cfg):
    processed = path_of(cfg, "paths.processed_dir")
    return {
        name: pd.read_parquet(processed / f"{name}.parquet")
        for name in schema.SPLITS
    }


def stage_validate(cfg, force=False):
    logger = setup_logger("stage_validate", path_of(cfg, "paths.logs_dir"))
    reports = path_of(cfg, "paths.reports_dir")
    ensure_dirs(reports)
    json_path = reports / "data_quality.json"
    md_path = reports / "data_quality.md"
    if json_path.exists() and md_path.exists() and not force:
        logger.info("skip validate (cached)")
        return
    ratings = read_ratings(_raw_path(cfg, "ratings"), cfg)
    movies = read_movies(_raw_path(cfg, "movies"))
    tags = read_tags(_raw_path(cfg, "tags"), cfg)
    logger.info("loaded ratings=%d movies=%d tags=%d", len(ratings), len(movies), len(tags))
    report = validate_ratings(ratings, movies, tags)
    write_report(report, json_path, md_path)
    logger.info("all_passed=%s report=%s", report["all_passed"], md_path)


def stage_split(cfg, force=False):
    logger = setup_logger("stage_split", path_of(cfg, "paths.logs_dir"))
    processed = path_of(cfg, "paths.processed_dir")
    ensure_dirs(processed)
    targets = [processed / f"{name}.parquet" for name in schema.SPLITS]
    if all(path.exists() for path in targets) and not force:
        logger.info("skip split (cached)")
        return
    ratings = read_ratings(_raw_path(cfg, "ratings"), cfg)
    frame = split_per_user(
        ratings,
        train_ratio=float(get(cfg, "data.split.train_ratio", 0.8)),
        val_ratio=float(get(cfg, "data.split.val_ratio", 0.1)),
        test_ratio=float(get(cfg, "data.split.test_ratio", 0.1)),
    )
    frames = frames_by_split(frame)
    summary = {}
    for name, split_frame in frames.items():
        path = processed / f"{name}.parquet"
        split_frame.to_parquet(path, index=False)
        summary[name] = int(len(split_frame))
    with open(processed / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("split sizes: %s", summary)


def stage_stats(cfg, force=False):
    logger = setup_logger("stage_stats", path_of(cfg, "paths.logs_dir"))
    reports = path_of(cfg, "paths.reports_dir")
    ensure_dirs(reports)
    json_path = reports / "dataset_stats.json"
    md_path = reports / "dataset_stats.md"
    if json_path.exists() and md_path.exists() and not force:
        logger.info("skip stats (cached)")
        return
    splits = _load_splits(cfg)
    movies = read_movies(_raw_path(cfg, "movies"))
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    report = describe_splits(splits, movies, threshold=threshold)
    write_stats(report, json_path, md_path)
    logger.info("stats written to %s", md_path)


def stage_candidates(cfg, force=False):
    logger = setup_logger("stage_candidates", path_of(cfg, "paths.logs_dir"))
    processed = path_of(cfg, "paths.processed_dir")
    ensure_dirs(processed)
    parquet_path = processed / "candidates.parquet"
    meta_path = processed / "candidates_meta.json"
    if parquet_path.exists() and meta_path.exists() and not force:
        logger.info("skip candidates (cached)")
        return
    splits = _load_splits(cfg)
    movies = read_movies(_raw_path(cfg, "movies"))
    seed = int(get(cfg, "seed", 20260907))
    candidates, meta = build_candidates(splits, movies, cfg, seed)
    write_candidates(candidates, meta, parquet_path, meta_path)
    logger.info(
        "candidates users=%d rows=%d seed=%d negatives=%d",
        meta["n_users"],
        meta["n_rows"],
        meta["seed"],
        meta["num_negatives"],
    )


def stage_baseline(cfg, force=False):
    logger = setup_logger("stage_baseline", path_of(cfg, "paths.logs_dir"))
    outputs = path_of(cfg, "paths.outputs_dir")
    metrics_dir = outputs / "metrics"
    top10_dir = outputs / "top10"
    ensure_dirs(metrics_dir, top10_dir)
    run_id = "R00_pop"
    metrics_path = metrics_dir / f"{run_id}.json"
    if metrics_path.exists() and not force:
        logger.info("skip baseline (cached)")
        return
    candidates = pd.read_parquet(path_of(cfg, "paths.processed_dir") / "candidates.parquet")
    train = pd.read_parquet(path_of(cfg, "paths.processed_dir") / "train.parquet")
    k = int(get(cfg, "eval.k", 10))
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    positive_only = bool(get(cfg, "baseline.popularity.positive_only", False))
    columns = get(cfg, "serving.columns")
    start = time.perf_counter()
    scores = popularity_scores(train, candidates, positive_only=positive_only, threshold=threshold)
    metrics, per_user = evaluate_candidates(candidates, scores, k=k)
    top10 = top_k_table(candidates, scores, k=int(get(cfg, "serving.top_k", 10)), columns=columns)
    elapsed = time.perf_counter() - start
    n_users = metrics["n_users"]
    metrics.update(
        {
            "run_id": run_id,
            "model": "most_popular_train",
            "positive_only": positive_only,
            "throughput_rps": float(len(candidates) / elapsed) if elapsed > 0 else 0.0,
            "latency_ms": float(elapsed / n_users * 1000.0) if n_users else 0.0,
        }
    )
    per_user.to_parquet(metrics_dir / f"{run_id}_per_user.parquet", index=False)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    write_top10(
        top10,
        top10_dir / f"{run_id}.csv",
        top10_dir / f"{run_id}_meta.json",
        {
            "run_id": run_id,
            "model": metrics["model"],
            "seed": int(get(cfg, "seed", 20260907)),
            "ndcg_at10": metrics["ndcg_at_k"],
        },
    )
    append_run(
        outputs / "leaderboard.csv",
        {
            "run_id": run_id,
            "model": metrics["model"],
            "recall_at10": f"{metrics['recall_at_k']:.6f}",
            "ndcg_at10": f"{metrics['ndcg_at_k']:.6f}",
            "lift_vs_pop": "1.0000",
            "ci_low": "0.0000",
            "throughput_rps": f"{metrics['throughput_rps']:.2f}",
            "latency_ms": f"{metrics['latency_ms']:.4f}",
            "seed": int(get(cfg, "seed", 20260907)),
            "config_path": cfg["_config_path"],
        },
    )
    logger.info(
        "ndcg@%d=%.6f hr@%d=%.6f users=%d",
        k,
        metrics["ndcg_at_k"],
        k,
        metrics["hr_at_k"],
        n_users,
    )


def stage_als(cfg, force=False):
    logger = setup_logger("stage_als", path_of(cfg, "paths.logs_dir"))
    run_id = "R03_als"
    outputs = path_of(cfg, "paths.outputs_dir")
    metrics_path = outputs / "metrics" / f"{run_id}.json"
    if metrics_path.exists() and not force:
        logger.info("skip als (cached)")
        return
    from src.rank.als_model import align_scores, score_pairs, train_als
    from src.utils.spark import get_spark

    processed = path_of(cfg, "paths.processed_dir")
    spark = get_spark(cfg, "als")
    start = time.perf_counter()
    try:
        train = spark.read.parquet(str(processed / "train.parquet")).select(
            schema.USER, schema.MOVIE, schema.RATING
        )
        candidate_pairs = spark.read.parquet(
            str(processed / "candidates.parquet")
        ).select(schema.CAND_ID, schema.USER, schema.MOVIE)
        model = train_als(spark, cfg, train)
        model_path = outputs / "models" / run_id
        ensure_dirs(model_path.parent)
        model.write().overwrite().save(str(model_path))
        scores_frame = score_pairs(spark, model, candidate_pairs).toPandas()
        scores_path = outputs / "scores" / f"{run_id}.parquet"
        ensure_dirs(scores_path.parent)
        scores_frame.to_parquet(scores_path, index=False)
    finally:
        spark.stop()
    elapsed = time.perf_counter() - start
    candidates = pd.read_parquet(processed / "candidates.parquet")
    scores = align_scores(candidates, scores_frame, len(candidates))
    metrics, _, significance = rank_and_report(
        cfg,
        candidates,
        scores,
        run_id,
        "spark_als",
        elapsed,
        extra={
            "als": {
                "rank": int(get(cfg, "als.rank", 64)),
                "max_iter": int(get(cfg, "als.max_iter", 10)),
                "reg_param": float(get(cfg, "als.reg_param", 0.1)),
                "implicit_prefs": bool(get(cfg, "als.implicit_prefs", False)),
                "nonnegative": bool(get(cfg, "als.nonnegative", True)),
            }
        },
    )
    logger.info(
        "ndcg@%d=%.6f lift=%s",
        int(get(cfg, "eval.k", 10)),
        metrics["ndcg_at_k"],
        f"{significance['lift']:.4f}" if significance else "n/a",
    )


def stage_ltr(cfg, force=False):
    logger = setup_logger("stage_ltr", path_of(cfg, "paths.logs_dir"))
    run_id = get(cfg, "ltr.run_id", "R06_ltr_f1")
    model_name = get(cfg, "ltr.model_name", "lightgbm_lambdarank_f1")
    outputs = path_of(cfg, "paths.outputs_dir")
    metrics_path = outputs / "metrics" / f"{run_id}.json"
    if metrics_path.exists() and not force:
        logger.info("skip ltr (cached)")
        return
    from src.features.basic import FeatureContext
    from src.rank.build_training import build_groups, group_sizes
    from src.rank.ltr import predict_scores, train_ranker

    processed = path_of(cfg, "paths.processed_dir")
    splits = _load_splits(cfg)
    movies = read_movies(_raw_path(cfg, "movies"))
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    seed = int(get(cfg, "seed", 20260907))
    k = int(get(cfg, "eval.k", 10))
    n_negatives = int(get(cfg, "ltr.num_negatives", 100))
    als_model_dir = outputs / "models" / "R03_als"
    start = time.perf_counter()
    context = FeatureContext(
        splits,
        movies,
        als_model_dir=als_model_dir if als_model_dir.exists() else None,
        threshold=threshold,
        use_bayes_rating=bool(get(cfg, "features.bayes_rating", True)),
        use_weighted_genre=bool(get(cfg, "features.weighted_genre", True)),
        weighted_genre_threshold=float(
            get(cfg, "features.weighted_genre_threshold", 3.5)
        ),
    )
    train_positives = splits[schema.TRAIN][
        splits[schema.TRAIN][schema.RATING] >= threshold
    ]
    val_positives = splits[schema.VAL][splits[schema.VAL][schema.RATING] >= threshold]
    train_groups = build_groups(
        cfg,
        train_positives,
        splits[schema.TRAIN],
        context.movie_ids,
        popularity=context.movie_count,
        seed=seed,
        max_users=get(cfg, "ltr.max_train_users"),
    )
    val_exclude = pd.concat([splits[schema.TRAIN], splits[schema.VAL]])
    val_groups = build_groups(
        cfg,
        val_positives,
        val_exclude,
        context.movie_ids,
        popularity=context.movie_count,
        seed=seed + 1,
        max_users=get(cfg, "ltr.valid_users"),
        hard_ratio=0.0,
    )
    graph_context = None
    if bool(get(cfg, "features.graph", False)):
        from src.features.graph import load_or_build

        graph_context, _ = load_or_build(
            splits,
            processed / "graph",
            threshold=threshold,
            top_k=int(get(cfg, "graph.top_k", 50)),
            max_items=int(get(cfg, "graph.max_items", 20000)),
            pagerank_iters=int(get(cfg, "graph.pagerank_iters", 20)),
            damping=float(get(cfg, "graph.damping", 0.85)),
            lpa_iters=int(get(cfg, "graph.lpa_iters", 5)),
            community_method=str(get(cfg, "graph.community_method", "louvain")),
            community_resolution=float(
                get(cfg, "graph.community_resolution", 1.0)
            ),
            seed=seed,
        )
    stream_context = None
    ActivityContext = None
    stream_mode = str(get(cfg, "features.stream_mode", "absolute"))
    if bool(get(cfg, "features.stream", False)):
        from src.stream.features import ActivityContext as ActivityContextClass

        ActivityContext = ActivityContextClass
        stream_context = ActivityContext.load(processed / "stream")
    features_train = context.compute(train_groups, adjust_train=True)
    features_val = context.compute(val_groups)
    if graph_context is not None:
        graph_columns = _graph_columns(cfg)
        graph_train = graph_context.compute(train_groups, adjust_train=True)[
            graph_columns
        ]
        graph_val = graph_context.compute(val_groups)[graph_columns]
        features_train = pd.concat([features_train, graph_train], axis=1)
        features_val = pd.concat([features_val, graph_val], axis=1)
    if stream_context is not None:
        features_train = pd.concat(
            [features_train, stream_context.compute(train_groups, mode=stream_mode)],
            axis=1,
        )
        features_val = pd.concat(
            [features_val, stream_context.compute(val_groups, mode=stream_mode)],
            axis=1,
        )
    groups_train = group_sizes(train_groups, n_negatives)
    groups_val = group_sizes(val_groups, n_negatives)
    logger.info(
        "training groups: train=%d rows, val=%d rows, features=%d",
        len(train_groups),
        len(val_groups),
        features_train.shape[1],
    )
    model = train_ranker(
        cfg,
        features_train,
        train_groups[schema.LABEL].to_numpy(),
        groups_train,
        features_val,
        val_groups[schema.LABEL].to_numpy(),
        groups_val,
    )
    models_dir = outputs / "models"
    ensure_dirs(models_dir)
    model.save_model(str(models_dir / f"{run_id}.txt"))
    importance = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "gain": model.feature_importance("gain"),
            "split": model.feature_importance("split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(outputs / "metrics" / f"{run_id}_importance.csv", index=False)
    val_scores = predict_scores(model, features_val)
    val_metrics, _ = evaluate_candidates(val_groups, val_scores, k=k)
    logger.info("val ndcg@%d=%.6f", k, val_metrics["ndcg_at_k"])
    candidates = pd.read_parquet(processed / "candidates.parquet")
    als_scores = None
    als_scores_path = outputs / "scores" / "R03_als.parquet"
    if als_scores_path.exists():
        als_frame = pd.read_parquet(als_scores_path)[[schema.CAND_ID, schema.SCORE]]
        als_scores = (
            candidates[[schema.CAND_ID]]
            .merge(als_frame, on=schema.CAND_ID, how="left")[schema.SCORE]
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    features_test = context.compute(candidates, als_scores=als_scores)
    if graph_context is not None:
        graph_test = graph_context.compute(candidates)[_graph_columns(cfg)]
        features_test = pd.concat([features_test, graph_test], axis=1)
    if stream_context is not None:
        features_test = pd.concat(
            [features_test, stream_context.compute(candidates, mode=stream_mode)],
            axis=1,
        )
    test_scores = predict_scores(model, features_test)
    elapsed = time.perf_counter() - start
    metrics, _, significance = rank_and_report(
        cfg,
        candidates,
        test_scores,
        run_id,
        model_name,
        elapsed,
        extra={
            "val_ndcg_at_k": val_metrics["ndcg_at_k"],
            "n_features": int(features_train.shape[1]),
            "best_iteration": int(model.best_iteration or model.current_iteration()),
            "ltr": {
                "num_negatives": n_negatives,
                "hard_ratio": float(get(cfg, "ltr.hard_ratio", 0.0)),
                "hard_negative_source": str(
                    get(cfg, "ltr.hard_negative_source", "popular")
                ),
                "max_train_users": get(cfg, "ltr.max_train_users"),
                "train_groups": int(train_groups[schema.USER].nunique()),
                "val_groups": int(val_groups[schema.USER].nunique()),
            },
            "graph": graph_context.meta if graph_context is not None else None,
            "stream": (
                {"mode": stream_mode, "features": ActivityContext.feature_columns(stream_mode)}
                if stream_context is not None
                else None
            ),
        },
    )
    logger.info(
        "test ndcg@%d=%.6f lift=%s",
        k,
        metrics["ndcg_at_k"],
        f"{significance['lift']:.4f}" if significance else "n/a",
    )


def stage_graph(cfg, force=False):
    logger = setup_logger("stage_graph", path_of(cfg, "paths.logs_dir"))
    processed = path_of(cfg, "paths.processed_dir")
    graph_dir = processed / "graph"
    meta_path = graph_dir / "graph_meta.json"
    if meta_path.exists() and not force:
        logger.info("skip graph (cached)")
        return
    from src.features.graph import load_or_build

    splits = _load_splits(cfg)
    context, built = load_or_build(
        splits,
        graph_dir,
        force=force,
        threshold=float(get(cfg, "data.positive_threshold", 4.0)),
        top_k=int(get(cfg, "graph.top_k", 50)),
        max_items=int(get(cfg, "graph.max_items", 20000)),
        pagerank_iters=int(get(cfg, "graph.pagerank_iters", 20)),
        damping=float(get(cfg, "graph.damping", 0.85)),
        lpa_iters=int(get(cfg, "graph.lpa_iters", 5)),
        community_method=str(get(cfg, "graph.community_method", "louvain")),
        community_resolution=float(get(cfg, "graph.community_resolution", 1.0)),
        seed=int(get(cfg, "seed", 20260907)),
    )
    logger.info(
        "%s: items=%d topk_nnz=%d communities=%d coverage=%.4f",
        "built" if built else "loaded",
        context.meta["n_graph_items"],
        context.meta["topk_nnz"],
        context.meta["n_communities"],
        context.meta["graph_item_positive_coverage"],
    )


def stage_spark_features(cfg, force=False):
    logger = setup_logger("stage_spark_features", path_of(cfg, "paths.logs_dir"))
    reports = path_of(cfg, "paths.reports_dir")
    processed = path_of(cfg, "paths.processed_dir")
    ensure_dirs(reports)
    json_path = reports / "spark_features.json"
    md_path = reports / "spark_features.md"
    if json_path.exists() and md_path.exists() and not force:
        logger.info("skip spark features (cached)")
        return
    from src.features.basic import FeatureContext
    from src.features.distributed import compute_distributed_stats, write_report
    from src.utils.spark import get_spark

    splits = _load_splits(cfg)
    movies = read_movies(_raw_path(cfg, "movies"))
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    spark = get_spark(cfg, "spark_features")
    start = time.perf_counter()
    try:
        train = spark.read.parquet(str(processed / "train.parquet")).select(
            schema.USER, schema.MOVIE, schema.RATING
        )
        movies_sdf = spark.createDataFrame(
            movies[[schema.MOVIE, schema.GENRES]]
        )
        stats = compute_distributed_stats(
            spark, train, movies_sdf, processed / "spark_features"
        )
        movie_stats = spark.read.parquet(
            str(processed / "spark_features" / "movie_stats")
        ).toPandas()
        user_stats = spark.read.parquet(
            str(processed / "spark_features" / "user_stats")
        ).toPandas()
    finally:
        spark.stop()
    elapsed = time.perf_counter() - start

    context = FeatureContext(splits, movies, threshold=threshold)
    movie_frame = pd.DataFrame(
        {
            schema.MOVIE: context.movie_ids,
            "movie_count": context.movie_count,
        }
    )
    movie_merge = movie_frame.merge(
        movie_stats[[schema.MOVIE, "movie_count"]].rename(
            columns={"movie_count": "spark_count"}
        ),
        on=schema.MOVIE,
        how="left",
    )
    movie_merge["spark_count"] = movie_merge["spark_count"].fillna(0)
    movie_match = float(
        (movie_merge["movie_count"] == movie_merge["spark_count"]).mean()
    )
    user_frame = pd.DataFrame(
        {
            schema.USER: context.user_ids,
            "user_count": context.user_count,
        }
    )
    user_merge = user_frame.merge(
        user_stats[[schema.USER, "user_count"]].rename(
            columns={"user_count": "spark_count"}
        ),
        on=schema.USER,
        how="left",
    )
    user_merge["spark_count"] = user_merge["spark_count"].fillna(0)
    user_match = float(
        (user_merge["user_count"] == user_merge["spark_count"]).mean()
    )
    report = {
        "elapsed_s": float(elapsed),
        "stats": stats,
        "checks": {
            "movie_count": {
                "rows": int(len(movie_frame)),
                "match_share": movie_match,
            },
            "user_count": {
                "rows": int(len(user_frame)),
                "match_share": user_match,
            },
        },
    }
    write_report(report, json_path, md_path)
    logger.info(
        "spark stats done in %.1fs; movie match=%.6f user match=%.6f",
        elapsed,
        movie_match,
        user_match,
    )


def stage_scalability(cfg, force=False):
    logger = setup_logger("stage_scalability", path_of(cfg, "paths.logs_dir"))
    reports = path_of(cfg, "paths.reports_dir")
    processed = path_of(cfg, "paths.processed_dir")
    ensure_dirs(reports)
    json_path = reports / "scalability.json"
    md_path = reports / "scalability.md"
    png_path = reports / "scalability.png"
    if json_path.exists() and md_path.exists() and not force:
        logger.info("skip scalability (cached)")
        return
    import copy

    from src.rank.als_model import score_pairs, train_als
    from src.utils.spark import get_spark

    parallelism_list = [
        int(value) for value in get(cfg, "scalability.parallelism", [2, 4, 6, 8])
    ]
    sample_ratings = int(get(cfg, "scalability.sample_ratings", 6_000_000))
    sample_candidates = int(get(cfg, "scalability.sample_candidates", 5_000_000))
    max_iter = int(get(cfg, "scalability.max_iter", get(cfg, "als.max_iter", 10)))
    seed = int(get(cfg, "seed", 20260907))

    entries = []
    for parallelism in parallelism_list:
        local_cfg = copy.deepcopy(cfg)
        local_cfg.setdefault("spark", {})["master"] = f"local[{parallelism}]"
        local_cfg.setdefault("als", {})["max_iter"] = max_iter
        spark = get_spark(local_cfg, f"scalability-local{parallelism}")
        try:
            train = spark.read.parquet(str(processed / "train.parquet")).select(
                schema.USER, schema.MOVIE, schema.RATING
            )
            n_train = train.count()
            fraction = min(sample_ratings / max(n_train, 1), 1.0)
            sample = (
                train
                if fraction >= 1.0
                else train.sample(False, fraction, seed=seed)
            )
            sample = sample.cache()
            n_sample = sample.count()
            start = time.perf_counter()
            model = train_als(spark, local_cfg, sample)
            fit_s = time.perf_counter() - start
            candidates = spark.read.parquet(
                str(processed / "candidates.parquet")
            ).select(schema.CAND_ID, schema.USER, schema.MOVIE)
            n_candidates = candidates.count()
            candidate_fraction = min(sample_candidates / max(n_candidates, 1), 1.0)
            candidate_sample = (
                candidates
                if candidate_fraction >= 1.0
                else candidates.sample(False, candidate_fraction, seed=seed)
            )
            n_candidate_sample = candidate_sample.count()
            start = time.perf_counter()
            score_pairs(spark, model, candidate_sample).count()
            score_s = time.perf_counter() - start
            sample.unpersist()
        finally:
            spark.stop()
        entries.append(
            {
                "parallelism": int(parallelism),
                "train_rows": int(n_sample),
                "fit_s": float(fit_s),
                "fit_rps": float(n_sample / fit_s) if fit_s > 0 else 0.0,
                "candidate_rows": int(n_candidate_sample),
                "score_s": float(score_s),
                "score_rps": (
                    float(n_candidate_sample / score_s) if score_s > 0 else 0.0
                ),
            }
        )
        logger.info(
            "local[%d] fit=%.1fs (%.0f ratings/s) score=%.1fs (%.0f pairs/s)",
            parallelism,
            fit_s,
            entries[-1]["fit_rps"],
            score_s,
            entries[-1]["score_rps"],
        )

    base = entries[0]
    for entry in entries:
        entry["fit_speedup"] = base["fit_s"] / entry["fit_s"]
        entry["score_speedup"] = base["score_s"] / entry["score_s"]

    payload = {
        "max_iter": max_iter,
        "seed": seed,
        "sample_ratings": sample_ratings,
        "sample_candidates": sample_candidates,
        "entries": entries,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = ["# 任务三：Spark ALS 扩展性实验", ""]
    lines.append(
        f"- 固定负载：训练样本 {entries[0]['train_rows']} 条评分（上限 {sample_ratings}），"
        f"候选打分样本 {entries[0]['candidate_rows']} 对（上限 {sample_candidates}）"
    )
    lines.append(f"- ALS：rank={get(cfg, 'als.rank', 64)}，maxIter={max_iter}")
    lines.append("")
    lines.append(
        "| 并行度 | 训练耗时(s) | 训练吞吐(ratings/s) | 加速比 | 打分耗时(s) | 打分吞吐(pairs/s) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for entry in entries:
        lines.append(
            f"| local[{entry['parallelism']}] | {entry['fit_s']:.1f} | "
            f"{entry['fit_rps']:.0f} | {entry['fit_speedup']:.2f} | "
            f"{entry['score_s']:.1f} | {entry['score_rps']:.0f} |"
        )
    lines.append("")
    lines.append("> 加速比以最小并行度为基准。")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        par = [entry["parallelism"] for entry in entries]
        axes[0].plot(par, [entry["fit_s"] for entry in entries], marker="o")
        axes[0].set_xlabel("Spark parallelism (local[n])")
        axes[0].set_ylabel("ALS fit time (s)")
        axes[0].set_title("Fit time vs parallelism")
        axes[0].grid(alpha=0.3)
        axes[1].plot(par, [entry["fit_rps"] for entry in entries], marker="o", color="tab:orange")
        axes[1].set_xlabel("Spark parallelism (local[n])")
        axes[1].set_ylabel("fit throughput (ratings/s)")
        axes[1].set_title("Throughput vs parallelism")
        axes[1].grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(png_path, dpi=150)
        plt.close(figure)
    except Exception as error:
        logger.info("skip plot: %s", error)
    logger.info("scalability report written to %s", md_path)


def stage_seed(cfg, force=False):
    logger = setup_logger("stage_seed", path_of(cfg, "paths.logs_dir"))
    reports = path_of(cfg, "paths.reports_dir")
    outputs = path_of(cfg, "paths.outputs_dir")
    ensure_dirs(reports)
    json_path = reports / "seed_robustness.json"
    md_path = reports / "seed_robustness.md"
    if json_path.exists() and md_path.exists() and not force:
        logger.info("skip seed robustness (cached)")
        return
    import lightgbm as lgb

    from src.baseline.popularity import popularity_scores
    from src.data.candidates import build_candidates
    from src.eval.ndcg import evaluate_candidates
    from src.features.basic import FeatureContext
    from src.features.graph import load_or_build as load_graph
    from src.stream.features import ActivityContext

    run_id = get(cfg, "ltr.run_id", "R10_ltr_stream")
    model_path = outputs / "models" / f"{run_id}.txt"
    official_seed = int(get(cfg, "robustness.official_seed", get(cfg, "seed", 20260907)))
    positive_only = bool(get(cfg, "baseline.popularity.positive_only", False))
    seeds = [
        int(value)
        for value in get(
            cfg, "robustness.seeds", [20260908, 20261001, 20261002, 20261003]
        )
    ]
    processed = path_of(cfg, "paths.processed_dir")
    splits = _load_splits(cfg)
    movies = read_movies(_raw_path(cfg, "movies"))
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    k = int(get(cfg, "eval.k", 10))
    stream_mode = str(get(cfg, "features.stream_mode", "absolute"))

    context = FeatureContext(
        splits,
        movies,
        als_model_dir=outputs / "models" / "R03_als",
        threshold=threshold,
        use_bayes_rating=bool(get(cfg, "features.bayes_rating", True)),
        use_weighted_genre=bool(get(cfg, "features.weighted_genre", True)),
        weighted_genre_threshold=float(
            get(cfg, "features.weighted_genre_threshold", 3.5)
        ),
    )
    graph_context = None
    if bool(get(cfg, "features.graph", False)):
        graph_context, _ = load_graph(
            splits,
            processed / "graph",
            threshold=threshold,
            top_k=int(get(cfg, "graph.top_k", 50)),
            max_items=int(get(cfg, "graph.max_items", 20000)),
            pagerank_iters=int(get(cfg, "graph.pagerank_iters", 20)),
            damping=float(get(cfg, "graph.damping", 0.85)),
            lpa_iters=int(get(cfg, "graph.lpa_iters", 5)),
            community_method=str(get(cfg, "graph.community_method", "louvain")),
            community_resolution=float(get(cfg, "graph.community_resolution", 1.0)),
            seed=int(get(cfg, "seed", 20260907)),
        )
    stream_context = (
        ActivityContext.load(processed / "stream")
        if bool(get(cfg, "features.stream", False))
        else None
    )
    model = lgb.Booster(model_file=str(model_path))
    train = splits[schema.TRAIN]

    official_metrics = json.loads(
        (outputs / "metrics" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    entries = [
        {
            "seed": official_seed,
            "ndcg": float(official_metrics["ndcg_at_k"]),
            "hr": float(official_metrics["hr_at_k"]),
            "popularity_ndcg": float(official_metrics["significance"]["baseline_ndcg"]),
            "source": "official",
        }
    ]
    for seed in seeds:
        start = time.perf_counter()
        candidates, _ = build_candidates(splits, movies, cfg, seed)
        als_scores = context.als_score(
            candidates[schema.USER].to_numpy(), candidates[schema.MOVIE].to_numpy()
        )
        features = context.compute(candidates, als_scores=als_scores)
        if graph_context is not None:
            features = pd.concat(
                [features, graph_context.compute(candidates)[_graph_columns(cfg)]],
                axis=1,
            )
        if stream_context is not None:
            features = pd.concat(
                [features, stream_context.compute(candidates, mode=stream_mode)],
                axis=1,
            )
        scores = model.predict(features)
        metrics, _ = evaluate_candidates(candidates, scores, k=k)
        popularity = popularity_scores(
            train, candidates, positive_only=positive_only, threshold=threshold
        )
        pop_metrics, _ = evaluate_candidates(candidates, popularity, k=k)
        entries.append(
            {
                "seed": seed,
                "ndcg": float(metrics["ndcg_at_k"]),
                "hr": float(metrics["hr_at_k"]),
                "popularity_ndcg": float(pop_metrics["ndcg_at_k"]),
                "source": "resampled",
            }
        )
        logger.info(
            "seed=%d ndcg=%.6f pop=%.6f lift=%.4f users=%d (%.0fs)",
            seed,
            metrics["ndcg_at_k"],
            pop_metrics["ndcg_at_k"],
            metrics["ndcg_at_k"] / pop_metrics["ndcg_at_k"],
            metrics["n_users"],
            time.perf_counter() - start,
        )
        del features, scores, candidates, als_scores

    ndcgs = np.asarray([entry["ndcg"] for entry in entries], dtype=np.float64)
    lifts = np.asarray(
        [entry["ndcg"] / entry["popularity_ndcg"] for entry in entries],
        dtype=np.float64,
    )
    sigma_seed = float(ndcgs.std(ddof=1)) if ndcgs.size > 1 else 0.0
    delta = float(official_metrics["significance"]["mean_diff"])
    payload = {
        "run_id": run_id,
        "official_delta": delta,
        "sigma_user_se": float(official_metrics["significance"]["se"]),
        "n_seeds": int(ndcgs.size),
        "ndcg_mean": float(ndcgs.mean()),
        "ndcg_std_seed": sigma_seed,
        "min_lift": float(lifts.min()),
        "share_lift_ge_1_1": float((lifts >= 1.1).mean()),
        "entries": entries,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = ["# 候选集种子稳健性（σ_seed）", ""]
    lines.append(f"- 固定模型：`{run_id}`")
    lines.append(f"- 官方种子：{official_seed}；重采样种子：{seeds}")
    lines.append(f"- 官方用户配对 Δ = {delta:.4f}，SE(σ_user) = {payload['sigma_user_se']:.6f}")
    lines.append(
        f"- 各套 NDCG 均值 = {ndcgs.mean():.6f}，**σ_seed = {sigma_seed:.6f}**，最小 lift = {lifts.min():.4f}，"
        f"lift ≥ 1.1 占比 = {payload['share_lift_ge_1_1']:.2%}"
    )
    lines.append("")
    lines.append("| seed | NDCG@10 | HR@10 | 同套热门基线 | lift |")
    lines.append("|---|---|---|---|---|")
    for entry in entries:
        lines.append(
            f"| {entry['seed']}{' (官方)' if entry['source'] == 'official' else ''} | "
            f"{entry['ndcg']:.6f} | {entry['hr']:.6f} | {entry['popularity_ndcg']:.6f} | "
            f"{entry['ndcg'] / entry['popularity_ndcg']:.4f} |"
        )
    if sigma_seed < delta:
        verdict = "σ_seed 远小于 Δ，增益不是种子运气"
    else:
        verdict = "σ_seed 与 Δ 同量级，增益不稳健，需回到特征/模型层"
    lines.append("")
    lines.append(f"**判定**：{verdict}（σ_seed={sigma_seed:.6f}，Δ={delta:.4f}）。")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logger.info(
        "sigma_seed=%.6f min_lift=%.4f share=%.2f -> %s",
        sigma_seed,
        lifts.min(),
        payload["share_lift_ge_1_1"],
        md_path,
    )


def stage_stream(cfg, force=False):
    logger = setup_logger("stage_stream", path_of(cfg, "paths.logs_dir"))
    processed = path_of(cfg, "paths.processed_dir")
    stream_dir = processed / "stream"
    meta_path = stream_dir / "stream_meta.json"
    if meta_path.exists() and not force:
        logger.info("skip stream (cached)")
        return
    from src.stream.replay import build_meta, run_aggregates, write_events, write_meta
    from src.utils.spark import get_spark

    splits = _load_splits(cfg)
    start = time.perf_counter()
    rows, files = write_events(
        splits, stream_dir / "events", n_files=int(get(cfg, "stream.files", 24))
    )
    logger.info("replay files written: %d events in %d files", rows, len(files))
    spark = get_spark(cfg, "stream")
    try:
        run_aggregates(
            spark,
            stream_dir / "events",
            stream_dir,
            stream_dir / "checkpoint",
            files_per_trigger=int(get(cfg, "stream.files_per_trigger", 2)),
        )
    finally:
        spark.stop()
    elapsed = time.perf_counter() - start
    write_meta(
        build_meta(
            rows, len(files), int(get(cfg, "stream.files_per_trigger", 2)), elapsed
        ),
        meta_path,
    )
    user_daily = pd.read_parquet(stream_dir / "user_daily")
    movie_daily = pd.read_parquet(stream_dir / "movie_daily")
    logger.info(
        "window aggregates: user_daily=%d rows, movie_daily=%d rows, %.1fs",
        len(user_daily),
        len(movie_daily),
        elapsed,
    )


def stage_lsh(cfg, force=False):
    logger = setup_logger("stage_lsh", path_of(cfg, "paths.logs_dir"))
    reports = path_of(cfg, "paths.reports_dir")
    ensure_dirs(reports)
    json_path = reports / "recall_lsh.json"
    md_path = reports / "recall_lsh.md"
    png_path = reports / "recall_lsh.png"
    if json_path.exists() and md_path.exists() and not force:
        logger.info("skip lsh (cached)")
        return
    from src.recall.itemcf import (
        brute_force_queries,
        build_incidence,
        exact_topk,
        topk_rows,
    )
    from src.recall.simhash import SimHashIndex

    splits = _load_splits(cfg)
    threshold = float(get(cfg, "data.positive_threshold", 4.0))
    max_items = int(get(cfg, "recall.max_items", 20000))
    max_users = int(get(cfg, "recall.max_users", 10000))
    top_k = int(get(cfg, "recall.top_k", 50))
    query_sample = int(get(cfg, "recall.query_sample", 300))
    brute_sample = int(get(cfg, "recall.brute_force_sample", 150))
    bits_list = get(cfg, "recall.lsh_bits", [64, 128, 256])
    band_sizes = get(cfg, "recall.band_sizes", [4, 8, 16])
    seed = int(get(cfg, "seed", 20260907))

    incidence_full, item_ids, _ = build_incidence(splits, threshold)
    user_activity = np.asarray(incidence_full.sum(axis=1)).ravel()
    keep = np.sort(np.argsort(-user_activity)[:max_users])
    keep = keep[user_activity[keep] > 0]
    incidence_full = incidence_full[keep, :].tocsr()
    counts = np.asarray(incidence_full.sum(axis=0)).ravel()
    selected = np.sort(np.argsort(-counts)[:max_items])
    selected = selected[counts[selected] > 0]
    incidence = incidence_full[:, selected].tocsr()
    item_ids = item_ids[selected]
    n_items = int(len(selected))
    logger.info(
        "items=%d users=%d (positives cover %d items)",
        n_items,
        len(keep),
        int((counts > 0).sum()),
    )

    start = time.perf_counter()
    exact = exact_topk(incidence, top_k=top_k, block_size=500)
    exact_build_s = time.perf_counter() - start

    scale = 1.0 / np.sqrt(np.maximum(counts[selected], 1.0))
    normalized = incidence.multiply(scale[None, :]).tocsr()
    transposed = normalized.T.tocsr()

    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(np.diff(exact.indptr) >= 5)
    size = min(query_sample, eligible.size)
    query_codes = np.sort(rng.choice(eligible, size=size, replace=False))
    exact_neighbors = topk_rows(exact, query_codes, k=top_k)
    logger.info("exact top-%d built in %.1fs; queries=%d", top_k, exact_build_s, size)

    brute_codes = query_codes[: min(brute_sample, query_codes.size)]
    brute_latency, brute_results = brute_force_queries(normalized, brute_codes, k=top_k)
    reference = topk_rows(exact, brute_codes, k=20)
    agreement = float(
        np.mean(
            [
                len(set(a[:20].tolist()) & set(b[:20].tolist())) / 20.0
                for a, b in zip(brute_results, reference)
            ]
        )
    )

    configs = []
    for bits in bits_list:
        for band_size in band_sizes:
            if int(bits) % int(band_size) != 0:
                continue
            build_start = time.perf_counter()
            index = SimHashIndex(
                incidence, n_bits=int(bits), band_size=int(band_size), seed=seed
            )
            build_s = time.perf_counter() - build_start
            latencies = []
            candidates = []
            recall_10 = []
            recall_k = []
            for code, ref in zip(query_codes, exact_neighbors):
                result, latency, n_candidates = index.query(
                    int(code), transposed, k=top_k
                )
                latencies.append(latency)
                candidates.append(n_candidates)
                recall_10.append(
                    len(set(result[:10].tolist()) & set(ref[:10].tolist())) / 10.0
                )
                recall_k.append(
                    len(set(result.tolist()) & set(ref.tolist())) / float(top_k)
                )
            configs.append(
                {
                    "n_bits": int(bits),
                    "band_size": int(band_size),
                    "n_bands": index.n_bands,
                    "recall_at_10": float(np.mean(recall_10)),
                    "recall_at_50": float(np.mean(recall_k)),
                    "avg_candidates": float(np.mean(candidates)),
                    "candidate_ratio": float(np.mean(candidates) / n_items),
                    "query_ms": float(np.mean(latencies) * 1000.0),
                    "build_s": float(build_s),
                }
            )
            logger.info(
                "lsh bits=%d band=%d bands=%d recall@10=%.3f recall@50=%.3f candidates=%.0f query_ms=%.2f",
                bits,
                band_size,
                index.n_bands,
                configs[-1]["recall_at_10"],
                configs[-1]["recall_at_50"],
                configs[-1]["avg_candidates"],
                configs[-1]["query_ms"],
            )

    payload = {
        "n_items": n_items,
        "n_users": int(len(keep)),
        "top_k": top_k,
        "exact_build_s": float(exact_build_s),
        "brute_force_query_ms": float(brute_latency * 1000.0),
        "brute_force_agreement_at_20": agreement,
        "query_sample": int(size),
        "seed": seed,
        "configs": configs,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    lines = ["# 任务二：精确 ItemCF 与 SimHash LSH 对比", ""]
    lines.append(f"- 物品数（训练期正反馈 Top-{max_items}）：{n_items}")
    lines.append(f"- 用户侧（最活跃 Top-{max_users}）：{len(keep)}")
    lines.append(f"- 精确余弦 Top-{top_k} 构建耗时：{exact_build_s:.1f}s")
    lines.append(
        f"- 暴力精确查询延迟：{brute_latency * 1000.0:.2f} ms（与精确 Top-20 一致率 {agreement:.3f}）"
    )
    lines.append(f"- 查询样本：{size}，种子：{seed}")
    lines.append("")
    lines.append(
        "| bits | band_size | bands | recall@10 | recall@50 | 平均候选数 | 候选比例 | 查询延迟(ms) | 构建(s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for entry in configs:
        lines.append(
            f"| {entry['n_bits']} | {entry['band_size']} | {entry['n_bands']} | "
            f"{entry['recall_at_10']:.3f} | "
            f"{entry['recall_at_50']:.3f} | {entry['avg_candidates']:.0f} | "
            f"{entry['candidate_ratio']:.4f} | {entry['query_ms']:.2f} | {entry['build_s']:.1f} |"
        )
    lines.append("")
    lines.append("> 说明：LSH 仅作任务二的准确率—效率证据，不直接进入 NDCG。")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            [entry["query_ms"] for entry in configs],
            [entry["recall_at_50"] for entry in configs],
            s=60,
        )
        for entry in configs:
            axis.annotate(
                f"{entry['n_bits']}b/{entry['band_size']}bnd",
                (entry["query_ms"], entry["recall_at_50"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
        axis.axhline(
            1.0, color="tab:red", linestyle="--", linewidth=1, label="exact recall=1"
        )
        axis.set_xlabel("query latency (ms)")
        axis.set_ylabel("recall@50 vs exact")
        axis.set_title("SimHash LSH: accuracy vs latency")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(png_path, dpi=150)
        plt.close(figure)
    except Exception as error:
        logger.info("skip plot: %s", error)
    logger.info("report written to %s", md_path)


STAGES = {
    "validate": stage_validate,
    "split": stage_split,
    "stats": stage_stats,
    "candidates": stage_candidates,
    "baseline": stage_baseline,
    "als": stage_als,
    "graph": stage_graph,
    "stream": stage_stream,
    "spark_features": stage_spark_features,
    "scalability": stage_scalability,
    "lsh": stage_lsh,
    "ltr": stage_ltr,
    "seed": stage_seed,
}
