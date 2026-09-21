import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, LongType, StructField, StructType

from src.data import schema


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


def run_aggregates(
    spark,
    events_dir,
    output_dir,
    checkpoint_dir,
    files_per_trigger=2,
    time_column="event_time",
):
    events_dir = Path(events_dir)
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)
    for path in (output_dir / "user_daily", output_dir / "movie_daily", checkpoint_dir):
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

    def process(batch, batch_id):
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

    query = (
        with_time.writeStream.foreachBatch(process)
        .option("checkpointLocation", str(checkpoint_dir))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    return query


def build_meta(rows, n_files, files_per_trigger, elapsed_s):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": int(rows),
        "files": int(n_files),
        "files_per_trigger": int(files_per_trigger),
        "elapsed_s": float(elapsed_s),
    }


def write_meta(meta, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
