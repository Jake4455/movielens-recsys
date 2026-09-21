import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

from src.data import schema
from src.utils.config import get


def train_als(spark, cfg, train_df):
    als = ALS(
        rank=int(get(cfg, "als.rank", 64)),
        maxIter=int(get(cfg, "als.max_iter", 10)),
        regParam=float(get(cfg, "als.reg_param", 0.1)),
        implicitPrefs=bool(get(cfg, "als.implicit_prefs", False)),
        nonnegative=bool(get(cfg, "als.nonnegative", True)),
        userCol=schema.USER,
        itemCol=schema.MOVIE,
        ratingCol=schema.RATING,
        coldStartStrategy="drop",
        seed=int(get(cfg, "seed", 20260907)),
    )
    return als.fit(train_df)


def score_pairs(spark, model, pairs_df):
    user_factors = model.userFactors.select(
        F.col("id").cast("long").alias(schema.USER),
        F.col("features").cast("array<double>").alias("user_features"),
    )
    item_factors = model.itemFactors.select(
        F.col("id").cast("long").alias(schema.MOVIE),
        F.col("features").cast("array<double>").alias("item_features"),
    )
    joined = pairs_df.join(user_factors, schema.USER, "left").join(
        item_factors, schema.MOVIE, "left"
    )
    dot = F.expr(
        "aggregate(zip_with(user_features, item_features, (x, y) -> x * y), "
        "0D, (acc, x) -> acc + x)"
    )
    return joined.withColumn(schema.SCORE, F.coalesce(dot, F.lit(0.0))).select(
        schema.CAND_ID, schema.SCORE
    )


def align_scores(candidates, scores_frame, n_rows):
    merged = candidates[[schema.CAND_ID]].merge(
        scores_frame, on=schema.CAND_ID, how="left"
    )
    if len(merged) != n_rows:
        raise RuntimeError("score alignment produced wrong row count")
    if merged[schema.SCORE].isna().any():
        merged[schema.SCORE] = merged[schema.SCORE].fillna(0.0)
    return merged[schema.SCORE].to_numpy(dtype=np.float64)
