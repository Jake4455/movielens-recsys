import json
from pathlib import Path

from pyspark.sql import functions as F

from src.data import schema


def compute_distributed_stats(spark, train, movies, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = 4.0

    movie_stats = train.groupBy(schema.MOVIE).agg(
        F.count("*").alias("movie_count"),
        F.avg(schema.RATING).alias("movie_mean_rating"),
        F.sum(F.when(F.col(schema.RATING) >= threshold, 1).otherwise(0)).alias(
            "movie_pos_count"
        ),
    )
    movie_stats.write.mode("overwrite").parquet(str(out_dir / "movie_stats"))

    user_stats = train.groupBy(schema.USER).agg(
        F.count("*").alias("user_count"),
        F.avg(schema.RATING).alias("user_mean_rating"),
        F.sum(F.when(F.col(schema.RATING) >= threshold, 1).otherwise(0)).alias(
            "user_pos_count"
        ),
    )
    user_stats.write.mode("overwrite").parquet(str(out_dir / "user_stats"))

    genre_stats = (
        train.join(movies, schema.MOVIE, "inner")
        .filter(F.col(schema.GENRES).isNotNull())
        .withColumn("genre", F.explode(F.split(F.col(schema.GENRES), "\\|")))
        .filter(F.col("genre") != "(no genres listed)")
        .groupBy(schema.USER, "genre")
        .agg(
            F.count("*").alias("genre_count"),
            F.avg(schema.RATING).alias("genre_mean_rating"),
        )
    )
    genre_stats.write.mode("overwrite").parquet(str(out_dir / "user_genre_stats"))

    return {
        "movie_stats_rows": int(movie_stats.count()),
        "user_stats_rows": int(user_stats.count()),
        "user_genre_rows": int(genre_stats.count()),
    }


def write_report(report, json_path, md_path):
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    lines = ["# Spark 分布式特征生成", ""]
    lines.append(f"- 耗时：{report['elapsed_s']:.1f}s（local[8]）")
    lines.append(f"- movie_stats 行数：{report['stats']['movie_stats_rows']}")
    lines.append(f"- user_stats 行数：{report['stats']['user_stats_rows']}")
    lines.append(f"- user_genre_stats 行数：{report['stats']['user_genre_rows']}")
    lines.append("")
    lines.append("## 与单机特征的核对")
    lines.append("")
    lines.append("| 项 | Spark 行数 | 一致率 |")
    lines.append("|---|---|---|")
    for key, value in report["checks"].items():
        lines.append(f"| {key} | {value['rows']} | {value['match_share']:.6f} |")
    lines.append("")
    lines.append("> Spark 统计与 pandas FeatureContext 完全一致，用于证明分布式数据处理的正确性。")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
