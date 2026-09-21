# Dataset Statistics

- generated_at: 2026-09-16T07:31:20.254578+00:00
- positive_threshold: 4.0
- n_ratings_total: 32000204
- n_users: 200948
- n_movies: 71411
- sparsity: 0.001818

## Splits

| split | n_ratings | ratio | n_users | n_movies | n_positives | positive_ratio | mean_rating |
|---|---|---|---|---|---|---|---|
| train | 25771222 | 0.8053 | 200948 | 71411 | 13048993 | 0.5063 | 3.5578 |
| val | 3114491 | 0.0973 | 200948 | 52139 | 1430609 | 0.4593 | 3.4593 |
| test | 3114491 | 0.0973 | 200948 | 55558 | 1458629 | 0.4683 | 3.4778 |

## per_user_ratings

| p0 | p25 | p50 | p75 | p90 | p99 | p100 |
|---|---|---|---|---|---|---|
| 16.0 | 30.0 | 59.0 | 135.0 | 292.0 | 1032.5 | 26666.0 |

## per_movie_ratings

| p0 | p25 | p50 | p75 | p90 | p99 | p100 |
|---|---|---|---|---|---|---|
| 1.0 | 2.0 | 5.0 | 25.0 | 262.0 | 8701.9 | 95832.0 |

## Long Tail

| key | value |
|---|---|
| n_movies_lt_5_ratings | 35418 |
| n_movies_lt_20_ratings | 51806 |
| share_movies_lt_5_ratings | 0.4959740096063632 |
