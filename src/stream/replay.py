"""任务四：按时间戳回放评分流 + Structured Streaming 窗口聚合 + 采样 + 增量用户画像。

三件事
------
1. **回放**：train+val 按时间戳全局排序后切成 N 个事件文件（file source 不保证行序，
   因此必须先落盘排序），再用 `maxFilesPerTrigger` 实现有序回放。
2. **窗口统计**：每个微批按 `window(1 day)` 聚合出 user_daily / movie_daily。
3. **采样 + 增量画像**（2026-10-08 补齐）：
   - 采样：每批按固定种子（`sample_seed + batch_id`）做伯努利采样，落盘 `sampled/`；
   - 增量画像：跨微批维护 `w(user, genre)`，每批先把历史状态按天数间隔做指数衰减，
     再累加本批正反馈事件的按事件时间加权的贡献（半衰期 `profile_half_life_days`）。
     状态只依赖此前（含本批）的事件，**不含任何未来信息**。
"""
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, LongType, StringType, StructField, StructType

from src.data import schema

PROFILE_PRUNE_THRESHOLD = 1e-6


def write_events(splits, directory, n_files=24):
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    columns = [schema.USER, schema.MOVIE, schema.RATING, schema.TIMESTAMP]
    frame = pd.concat(
        [splits[schema.TRAIN][columns], splits[schema.VAL][columns]]
    ).sort_values(
        [schema.TIMESTAMP, schema.USER, schema.MOVIE],
        kind="mergesort",
    )
    frame = frame.reset_index(drop=True)
    shard = max(math.ceil(len(frame) / n_files), 1)
    files = []
    for index, start in enumerate(range(0, len(frame), shard)):
        path = directory / f"events_{index:03d}.parquet"
        frame.iloc[start : start + shard].to_parquet(path, index=False)
        files.append(path.name)
    return len(frame), files


def top_active_users(events_dir, n=200):
    """回放事件里交互次数最多的 n 个用户（跨整条时间线，适合做画像演化演示）。"""
    events_dir = Path(events_dir)
    counts = pd.Series(dtype="int64")
    for path in sorted(events_dir.glob("events_*.parquet")):
        column = pd.read_parquet(path, columns=[schema.USER])[schema.USER]
        counts = counts.add(column.value_counts(), fill_value=0)
    counts = counts.astype("int64").sort_values(ascending=False)
    return sorted(int(value) for value in counts.head(int(n)).index)


def decay_factor(gap_days, half_life_days):
    """两次更新之间的状态衰减系数：gap 天之后保留 0.5 ** (gap / half_life)。"""
    return float(0.5 ** (max(float(gap_days), 0.0) / float(half_life_days)))


class ProfileState:
    """跨微批的增量用户-题材兴趣画像（指数衰减 + 累加）。"""

    def __init__(self, half_life_days=365.0):
        self.half_life_days = float(half_life_days)
        self.series = pd.Series(dtype="float64")  # MultiIndex(userId, genre) -> weight
        self.last_day = None
        self.batches = []
        self.pruned_total = 0

    def update(self, contributions, end_day, batch_id, extra=None):
        """contributions: DataFrame[userId, genre, weight]（本批按事件时间加权的贡献）。"""
        gap = 0 if self.last_day is None else max((pd.Timestamp(end_day) - pd.Timestamp(self.last_day)).days, 0)
        factor = decay_factor(gap, self.half_life_days)
        if factor != 1.0 and len(self.series):
            self.series = self.series * factor

        if contributions is not None and len(contributions):
            incoming = contributions.set_index([schema.USER, "genre"])["weight"].astype("float64")
            self.series = (
                self.series.add(incoming, fill_value=0.0)
                if len(self.series)
                else incoming
            )

        before = len(self.series)
        if before:
            self.series = self.series[self.series >= PROFILE_PRUNE_THRESHOLD]
        pruned = before - len(self.series)
        self.pruned_total += pruned
        self.last_day = pd.Timestamp(end_day) if self.last_day is None else max(
            pd.Timestamp(end_day), pd.Timestamp(self.last_day)
        )

        entry = {
            "batch_id": int(batch_id),
            "end_day": str(pd.Timestamp(end_day).date()),
            "gap_days": int(gap),
            "decay_factor": round(factor, 8),
            "contrib_pairs": int(0 if contributions is None else len(contributions)),
            "contrib_weight": round(float(0 if contributions is None or not len(contributions) else contributions["weight"].sum()), 4),
            "state_pairs": int(len(self.series)),
            "state_weight": round(float(self.series.sum()) if len(self.series) else 0.0, 4),
            "pruned_pairs": int(pruned),
        }
        if extra:
            entry.update(extra)
        if len(self.series):
            top = self.series.groupby(level=1).sum().idxmax()
            entry["top_genre"] = str(top)
        self.batches.append(entry)
        return entry

    def frame(self):
        if not len(self.series):
            return pd.DataFrame(columns=[schema.USER, "genre", "weight"])
        out = self.series.rename("weight").reset_index()
        out.columns = [schema.USER, "genre", "weight"]
        return out

    def top_for_users(self, user_ids, top_k=5):
        if not len(self.series):
            return pd.DataFrame(columns=[schema.USER, "genre", "weight", "rank"])
        subset = self.series[self.series.index.get_level_values(0).isin(list(user_ids))]
        if not len(subset):
            return pd.DataFrame(columns=[schema.USER, "genre", "weight", "rank"])
        frame = subset.rename("weight").reset_index()
        frame.columns = [schema.USER, "genre", "weight"]
        frame = frame.sort_values([schema.USER, "weight"], ascending=[True, False])
        frame["rank"] = frame.groupby(schema.USER).cumcount() + 1
        return frame[frame["rank"] <= top_k]


def _genre_frame(spark, movies):
    """movieId → genre 展开表（约 17 万行，广播用）。"""
    frame = movies[[schema.MOVIE, schema.GENRES]].copy()
    frame[schema.GENRES] = frame[schema.GENRES].fillna("")
    sdf = spark.createDataFrame(
        frame,
        schema=StructType(
            [
                StructField(schema.MOVIE, LongType()),
                StructField(schema.GENRES, StringType()),
            ]
        ),
    )
    return (
        sdf.select(
            schema.MOVIE,
            F.explode(F.split(F.col(schema.GENRES), r"\|")).alias("genre"),
        )
        .filter((F.col("genre") != "") & (F.col("genre") != "(no genres listed)"))
        .distinct()
    )


def run_aggregates(
    spark,
    events_dir,
    output_dir,
    checkpoint_dir,
    files_per_trigger=2,
    time_column="event_time",
    movies=None,
    positive_threshold=4.0,
    sample_fraction=0.01,
    sample_seed=20260907,
    profile_half_life_days=365.0,
    demo_users=200,
    demo_user_ids=None,
    snapshot_batches=(1, 4, 8, 11),
    progress_path=None,
):
    events_dir = Path(events_dir)
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)
    for path in (
        output_dir / "user_daily",
        output_dir / "movie_daily",
        output_dir / "sampled",
        output_dir / "user_profile",
        checkpoint_dir,
    ):
        if Path(path).exists():
            shutil.rmtree(path)

    event_schema = StructType(
        [
            StructField(schema.USER, LongType()),
            StructField(schema.MOVIE, LongType()),
            StructField(schema.RATING, FloatType()),
            StructField(schema.TIMESTAMP, LongType()),
        ]
    )
    stream = (
        spark.readStream.schema(event_schema)
        .option("maxFilesPerTrigger", int(files_per_trigger))
        .parquet(str(events_dir))
    )
    with_time = stream.withColumn(
        time_column, F.to_timestamp(F.col(schema.TIMESTAMP))
    )

    genre_frame = _genre_frame(spark, movies) if movies is not None else None
    state = ProfileState(profile_half_life_days) if genre_frame is not None else None
    demo_ids = sorted(int(v) for v in demo_user_ids) if demo_user_ids else []
    progress_file = Path(progress_path) if progress_path else (output_dir / "user_profile" / "progress.jsonl")
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    def _append_progress(entry):
        with open(progress_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def process(batch, batch_id):
        bid = int(batch_id) if batch_id is not None else -1
        batch = batch.cache()
        n_batch = batch.count()

        # ---------- 1) 采样（每批固定种子，可复现） ----------
        sampled = (
            batch.withColumn("batch_id", F.lit(bid))
            .sample(False, float(sample_fraction), seed=int(sample_seed) + bid)
        )
        n_sampled = sampled.count()
        sampled.write.mode("append").parquet(str(output_dir / "sampled"))

        # ---------- 2) 窗口统计（原有能力） ----------
        user_daily = (
            batch.groupBy(F.window(time_column, "1 day"), schema.USER)
            .agg(F.count("*").alias("count"))
            .select(F.col("window.start").alias("window_start"), schema.USER, "count")
        )
        user_daily.write.mode("append").parquet(str(output_dir / "user_daily"))
        movie_daily = (
            batch.groupBy(F.window(time_column, "1 day"), schema.MOVIE)
            .agg(F.count("*").alias("count"))
            .select(F.col("window.start").alias("window_start"), schema.MOVIE, "count")
        )
        movie_daily.write.mode("append").parquet(str(output_dir / "movie_daily"))

        # ---------- 3) 采样 + 增量画像 ----------
        entry = {"batch_id": bid, "events": int(n_batch), "sampled": int(n_sampled),
                 "sample_fraction": float(sample_fraction)}
        if genre_frame is not None:
            bounds = batch.agg(
                F.min(F.to_date(F.col(time_column))).alias("day_min"),
                F.max(F.to_date(F.col(time_column))).alias("day_max"),
            ).collect()[0]
            day_min, day_max = bounds["day_min"], bounds["day_max"]
            if not demo_ids:
                pool = np.sort(
                    np.asarray(
                        batch.select(schema.USER).distinct().limit(20000).toPandas()[schema.USER],
                        dtype=np.int64,
                    )
                )
                rng = np.random.default_rng(int(sample_seed))
                size = min(int(demo_users), pool.size)
                demo_ids.extend(sorted(int(v) for v in rng.choice(pool, size=size, replace=False)))
            weighted = (
                batch.filter(F.col(schema.RATING) >= float(positive_threshold))
                .join(F.broadcast(genre_frame), schema.MOVIE, "inner")
                .withColumn("event_day", F.to_date(F.col(time_column)))
                .withColumn(
                    "w",
                    F.pow(
                        F.lit(0.5),
                        F.datediff(F.lit(day_max), F.col("event_day")).cast("double")
                        / float(profile_half_life_days),
                    ),
                )
            )
            contributions = weighted.groupBy(schema.USER, "genre").agg(
                F.sum("w").alias("weight")
            ).toPandas()
            entry.update({"day_min": str(day_min), "day_max": str(day_max)})
            entry = state.update(contributions, day_max, bid, extra=entry)
            if bid in set(int(v) for v in snapshot_batches):
                snapshot = state.top_for_users(demo_ids, top_k=5)
                if len(snapshot):
                    snapshot["batch_id"] = bid
                    snapshot["end_day"] = str(pd.Timestamp(day_max).date())
                    snapshot.to_parquet(
                        output_dir / "user_profile" / f"demo_snapshot_batch{bid:03d}.parquet",
                        index=False,
                    )
        _append_progress(entry)
        batch.unpersist()

    query = (
        with_time.writeStream.foreachBatch(process)
        .option("checkpointLocation", str(checkpoint_dir))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()

    if state is not None:
        final = state.frame()
        if len(final):
            spark.createDataFrame(final).write.mode("overwrite").parquet(
                str(output_dir / "user_profile" / "final")
            )
        (output_dir / "user_profile" / "meta.json").write_text(
            json.dumps(
                {
                    "half_life_days": float(profile_half_life_days),
                    "positive_threshold": float(positive_threshold),
                    "sample_fraction": float(sample_fraction),
                    "sample_seed": int(sample_seed),
                    "demo_users": len(demo_ids),
                    "batches": state.batches,
                    "final_pairs": int(len(final)),
                    "pruned_total": int(state.pruned_total),
                    "note": "状态只使用该批及之前的事件；作为特征使用时须按批次时间切片",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return query


def build_meta(rows, n_files, files_per_trigger, elapsed_s, extra=None):
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": int(rows),
        "files": int(n_files),
        "files_per_trigger": int(files_per_trigger),
        "elapsed_s": float(elapsed_s),
    }
    if extra:
        meta.update(extra)
    return meta


def write_meta(meta, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
