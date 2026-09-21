# Spark 分布式特征生成

- 耗时：34.7s（local[8]）
- movie_stats 行数：71411
- user_stats 行数：200948
- user_genre_stats 行数：3213100

## 与单机特征的核对

| 项 | Spark 行数 | 一致率 |
|---|---|---|
| movie_count | 87585 | 1.000000 |
| user_count | 200948 | 1.000000 |

> Spark 统计与 pandas FeatureContext 完全一致，用于证明分布式数据处理的正确性。