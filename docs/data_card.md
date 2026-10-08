# 数据卡（data card）

> 对应 PRD FR-1.1「校验 MovieLens 32M（MD5、行数、列名），记录数据版本」。
> 生成时间：2026-10-08T11:12:23.574389+00:00

## 1. 数据集

| 项 | 值 |
|---|---|
| 数据集 | MovieLens 32M（稳定基准） |
| 获取方式 | GroupLens 官网 `ml-32m.zip` 解压到 `data/ml-32m/`（原始数据不提交，见 `.gitignore`） |
| 引文 | Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. |
| 时间跨度 | 1995-01-09 至 2023-10-13（`data_quality.json::timestamp_max_utc`） |
| 评分范围 | 0.5 – 5.0（半星，10 档） |
| 正反馈定义 | rating ≥ 4 |

## 2. 文件级校验（MD5 与行数）

| 文件 | 字节数 | 数据行数（不含表头） | 实测 MD5 | `checksums.txt` | 一致 |
|---|---|---|---|---|---|
| `ratings.csv` | 877,076,222 | 32,000,204 | `CF12B74F9AD4B94A011F079E26D4270A` | `CF12B74F9AD4B94A011F079E26D4270A` | ✅ |
| `movies.csv` | 4,242,926 | 87,585 | `0DF90835C19151F9D819D0822E190797` | `0DF90835C19151F9D819D0822E190797` | ✅ |
| `tags.csv` | 72,353,890 | 2,000,072 | `963BF4FA4DE6B8901868FDDD3EB54567` | `963BF4FA4DE6B8901868FDDD3EB54567` | ✅ |
| `links.csv` | 1,950,748 | 87,585 | `8F033867BCB4E6BE8792B21468B4FA6E` | `8F033867BCB4E6BE8792B21468B4FA6E` | ✅ |

## 3. 列名与类型

| 文件 | 列 |
|---|---|
| ratings.csv | userId (int64), movieId (int64), rating (float32), timestamp (int64) |
| movies.csv | movieId (int64), title (str), genres (str) |
| tags.csv | userId, movieId, tag, timestamp |
| links.csv | movieId, imdbId, tmdbId |

## 4. 质量检查结论（`outputs/reports/data_quality.md`）

| 检查 | 结果 |
|---|---|
| 评分记录数 | 32,000,204 |
| 用户数 / 电影数（评分表） | 200,948 / 84,432 |
| 重复 (userId, movieId) | 0 |
| 缺失值 | 0 |
| 评分/时间戳越界 | 0 |
| 评分表电影在 movies.csv 覆盖率 | 100% |
| tags 缺失值 | 17 / 2,000,072 |

## 5. 派生数据版本

| 产物 | 行数 | 说明 |
|---|---|---|
| `data/processed/cleaned.parquet` | 32,000,204 | 清洗层（校验通过，无删除/修补；等于 train+val+test 并集） |
| `data/processed/train.parquet` | 25,771,222 | 每用户时间序前 80% |
| `data/processed/val.parquet` | 3,114,491 | 中 10% |
| `data/processed/test.parquet` | 3,114,491 | 后 10% |
| `data/processed/candidates.parquet` | 18,915,078 | 187,278 用户 × (1 正 + 100 均匀负)，种子 20260907 |

## 6. 可复现性

- 种子：20260907（候选集、ALS、LightGBM、bootstrap 统一使用，见 `conf/*.yaml` 的 `seed`）。
- 校验与划分脚本：`src/data/validate.py`、`src/data/split.py`、`src/data/candidates.py`。
- 校验产物：`outputs/reports/data_quality.json`、`outputs/reports/dataset_stats.json`。
- 复现命令：`python run_all.py --config conf/final.yaml --stages validate,split,stats,candidates`。
